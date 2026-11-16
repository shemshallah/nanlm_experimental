"""Audited frozen-donor -> live RSI predict/observe replay and restart test.

This bridge reads the already frozen distributions from the *unchanged* v102
donor and feeds them into StreamingRSICore in chronological order. Each byte's
target is passed only to observe(), after its probability was returned.
The archived frozen probabilities are independently checked against the
donor's own per-byte records before a replay can begin.

The bridge does not turn the frozen donor into a live training checkpoint or
claim that its source alone supplies a real-time probability socket. It is
an exact end-to-end integration contract for that future interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from .causal_echo import _data
from .feedback_core import FeedbackRSI, StreamingRSICore


def windows(work:Path,split:str):
    raw=(work/f'win_{split}.bin').read_bytes()
    count,skip=struct.unpack_from('<II',raw)
    pos=8
    for _ in range(count):
        if pos+4>len(raw):raise ValueError('Truncated window header')
        width=struct.unpack_from('<I',raw,pos)[0]
        pos+=4
        chunk=raw[pos:pos+width]
        pos+=width
        if len(chunk)!=width or not 0<skip<width:
            raise ValueError('Malformed donor book window')
        yield chunk,skip
    if pos!=len(raw):raise ValueError('Trailing donor window bytes')


def replay(work:Path,split:str,checkpoint:Path,*,resume_every=240):
    """Score all bytes strictly prequentially and audit batch equivalence."""
    allowed={'dev2','test2','test3','dev4','holdout4'}
    if split not in allowed:raise ValueError('Unapproved frozen replay split')
    if not isinstance(resume_every,int) or resume_every<0:
        raise ValueError('Invalid restart interval')
    source_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model=FeedbackRSI.from_checkpoint(checkpoint)
    probs,targets,per,classes,counts,orders=_data(work,split)
    expected=model.score(probs,targets,per,classes,counts,orders)
    stream=StreamingRSICore(model)
    start=time.monotonic()
    loss_sum=0.;top1=top5=0;scored=0;restarts=0
    book_bpc=[];book_receipts=[]
    for book, (window,skip) in enumerate(windows(work,split)):
        if len(window)-skip!=per:
            raise ValueError('Unexpected scored positions per book')
        stream.reset_book(f'{split}-book-{book:03d}')
        for ch in window[:skip]:
            stream.warmup(int(ch))
        book_loss=0.
        for i,target in enumerate(window[skip:]):
            at=scored
            if int(target)!=int(targets[at]):
                raise ValueError('Stream target differs from independently audited donor')
            predicted=stream.predict(probs[at])
            if not np.isfinite(predicted).all() or abs(float(predicted.sum())-1.)>1e-10:
                raise ValueError('Invalid live RSI distribution')
            loss=-math.log(max(float(predicted[int(target)]),1e-30))
            loss_sum+=loss;book_loss+=loss
            top1+=int(int(np.argmax(predicted))==int(target))
            top5+=int(int(target) in np.argpartition(predicted,-5)[-5:])
            stream.observe(int(target))
            scored+=1
            if resume_every and (i+1)%resume_every==0 and i+1<per:
                snap=stream.snapshot()
                stream=StreamingRSICore.from_snapshot(model,json.loads(json.dumps(snap)))
                if stream.snapshot()!=snap:
                    raise AssertionError('Streaming RSI restart altered a state')
                restarts+=1
        book_bpc.append(book_loss/(per*math.log(2)))
        book_receipts.append(stream.stream_digest)
    if scored!=len(targets):
        raise ValueError('Missing live-stream predictions')
    bpc=loss_sum/(scored*math.log(2))
    t1=top1/scored;t5=top5/scored
    if (abs(bpc-expected['bpc'])>1e-11 or
            abs(t1-expected['top1'])>1e-14 or
            abs(t5-expected['top5'])>1e-14 or
            max(abs(a-b) for a,b in zip(book_bpc,expected['per_book_bpc']))>1e-11):
        raise AssertionError('Live RSI disagrees with independently audited frozen batch')
    if source_digest!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError('Model checkpoint changed during read-only evaluation')
    elapsed=time.monotonic()-start
    report={'schema':'terminal-rsi-v102-live-feedback-bridge-v1',
            'split':split,'book_count':len(book_receipts),'scored_bytes':scored,
            'source_checkpoint_sha256':source_digest,
            'training_updates_during_replay':0,
            'bits_per_byte':bpc,'top1':t1,'top5':t5,
            'elapsed_seconds':round(elapsed,3),
            'bytes_per_second':round(scored/max(elapsed,1e-12),2),
            'checkpoint_resume_count':restarts,
            'exact_batch_metric_equivalence':True,
            'per_book_bpc':book_bpc,
            'book_receipt_sha256':book_receipts,
            'candidate_promoted':False,
            'caveat':('Previously inspected book set; not an independent promotion gate'
                      if split in ('dev2','test2','test3')
                      else 'Frozen donor must pass separate corpus independence audit')}
    (work/f'rsi_live_{split}_replay.json').write_text(
        json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=('dev2','test2','test3','dev4','holdout4'),required=True)
    p.add_argument('--resume-every',type=int,default=240)
    args=p.parse_args()
    print(json.dumps(replay(args.work,args.split,args.checkpoint,
                             resume_every=args.resume_every),sort_keys=True,indent=2))


if __name__=='__main__':main()
