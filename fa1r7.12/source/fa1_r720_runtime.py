#!/usr/bin/env python3
"""FA1 R7.20: frozen R7.18 full-model scoring with streaming 128K far lookup.

READ-ONLY: accepted R7.13, R7.16, and R7.18 parameters are not trained.
The exact_reference mode reproduces the R7.18 far-path history rescan.
The streaming_index mode replaces ONLY that far scan with R7.19 causal state.
All other retrieval and language-model computations use identical batch paths.

For reproducibility against historical R7.18 metrics, the underlying inherited
R7.12 matrix is re-initialized per batch from its historical 6144-word block
window; its alternative per-token recurrence is NOT equivalent to that path.
The 128K retrieval index alone persists across consecutive positions.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import time
import numpy as np
import torch
from r7_core import Corpus
from train_r7 import batch_data
from r74_stream import OUTER_CONTEXT
from r712_streaming import DirectedStreaming
from r712_validate import restore
from r710_checkpoint_integrity import sha256
from fa1_r716_router import TrustRouter, extract_paths, nll as near_nll, PARENT, RECENT
from fa1_r718_far_residual import (FarResidual, far_only, make_feature, loss_far,
                                  HORIZON, FAR_CHANNELS)
from fa1_r719_streaming_index import StreamingFarIndex
from fa1_r720_packed_index import PackedFarIndex

NEAR_SHA='436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c'
FAR_SHA='f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f'


def restore_frozen(root, near_path, far_path, threads=2):
    """Verify exact parent/adapter chain and five inherited optimizer histories."""
    root=pathlib.Path(root); near_path=pathlib.Path(near_path);far_path=pathlib.Path(far_path)
    torch.set_num_threads(threads)
    if sha256(root/'checkpoint/pal_active.pt') != PARENT:
        raise ValueError('accepted FA1 R7.13 checkpoint digest mismatch')
    if sha256(near_path)!=NEAR_SHA or sha256(far_path)!=FAR_SHA:
        raise ValueError('FA1 experimental overlay digest mismatch')
    model,opts,base,comp,_,_=restore(root,root/'adaptation/r712_gate_delta.pt')
    if len(opts)!=5:raise ValueError('original optimizer lineage unexpectedly changed')
    near_blob=torch.load(near_path,map_location='cpu',weights_only=False)
    far_blob=torch.load(far_path,map_location='cpu',weights_only=False)
    if near_blob['parent_sha256'] != PARENT or far_blob['parent_sha256']!=PARENT:
        raise ValueError('adapter checkpoint parent does not match')
    if far_blob['r716_overlay_sha256']!=NEAR_SHA:
        raise ValueError('far adapter near ancestor mismatch')
    near=TrustRouter();near.load_state_dict(near_blob['gate']);near.eval()
    far=FarResidual();far.load_state_dict(far_blob['far_gate']);far.eval()
    model.eval()
    for module in (model,near,far):
        for p in module.parameters():p.requires_grad_(False)
    return model,opts,base,comp,near,far


class FA1R720Runtime:
    """Causal corpus-bound execution. No document-spanning memory and no writes."""
    def __init__(self,root,model,base,comp,near,far,split='test',corpus_dir=None,
                 max_step=2048):
        self.root=pathlib.Path(root)
        self.corpus=Corpus(root,split,corpus_dir=corpus_dir)
        self.model=model;self.base=base;self.comp=comp;self.near=near;self.far=far
        self.stream=DirectedStreaming(model)
        self.index=None;self.max_step=int(max_step)
        self.stats={'resets':0,'incremental_steps':0}

    def _document_start(self,pos):
        if self.corpus.doc_start is None:return 0
        return int(self.corpus.doc_start[pos])

    def _far_records(self,pos,q,target,mode,reference_history=None):
        if mode=='exact_reference':
            if reference_history is None:raise ValueError('reference history required')
            return far_only(reference_history,q,target)
        if mode not in ('streaming_index','packed_index'):raise ValueError('unknown far-memory mode')
        doc=self._document_start(pos)
        if (self.index is None or (mode=='packed_index' and not isinstance(self.index,PackedFarIndex))
                or (mode=='streaming_index' and not isinstance(self.index,StreamingFarIndex))
                or self.index.document_start!=doc
                or pos < self.index.position or pos-self.index.position>self.max_step):
            self.index=(PackedFarIndex if mode=='packed_index' else StreamingFarIndex)(
                self.corpus.seq,pos,document_start=doc)
            self.stats['resets']+=1
        else:
            self.stats['incremental_steps']+=pos-self.index.position
            self.index.advance(pos)
        if self.index.position != pos:raise AssertionError('index time drift')
        return self.index.query(q,target)

    @torch.no_grad()
    def score_positions(self,positions,mode='streaming_index',batch_size=4,return_details=False):
        """Return original, near, and far NLL (nats/word) on causal positions.

        `target` is passed solely to compute a diagnostic next-word likelihood;
        the gate features, modal choice and prior are determined without it.
        Caller must supply ordered positions; split/document boundaries reset.
        """
        positions=np.asarray(positions,dtype=np.int64)
        if len(positions)==0 or np.any(positions<0) or np.any(positions>=len(self.corpus.seq)):
            raise ValueError('invalid/nonempty positions required')
        if np.any(np.diff(positions)<=0):raise ValueError('ordered unique positions only')
        requested_mode=mode
        if mode=='auto':
            # The packed index is most useful for long monotone, same-document
            # streams. Rescanning remains cheaper for short or sparse queries.
            starts=self.corpus.doc_start
            same_document=(starts is None or int(starts[positions[0]])==int(starts[positions[-1]]))
            dense=(len(positions)>=128 and bool(np.all(np.diff(positions)<=4)))
            mode='packed_index' if (same_document and dense) else 'exact_reference'
        times={k:0. for k in ('data','base','near','far','prior_and_gate')}
        original=[];nearloss=[];farll=[];details=[]
        for offset in range(0,len(positions),batch_size):
            inds=positions[offset:offset+batch_size]
            t=time.perf_counter()
            _,ctx,pieces,lens,y,_,_=batch_data(self.base,self.corpus,self.comp,inds)
            older=torch.from_numpy(self.corpus.get(inds,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
            near_hist=self.corpus.get(inds,32768)[0]
            far_hist=self.corpus.get(inds,HORIZON)[0] if mode=='exact_reference' else None
            times['data']+=time.perf_counter()-t
            t=time.perf_counter()
            state=self.stream.initial(older)
            ex=torch.from_numpy(self.corpus.data[inds].copy()).long()
            lg=torch.from_numpy(self.corpus.long[inds].copy()).long()
            now,base_loss,_=self.stream.forward(ex,lg,ctx,pieces,lens,state,older,y)
            times['base']+=time.perf_counter()-t
            all_records=[];near_rows=[];far_rows=[];near_prob=[];far_prob=[];near_count=[];far_count=[]
            for j,pos in enumerate(inds):
                q=ctx[j].numpy();target=int(y[j]);t=time.perf_counter()
                n=extract_paths(near_hist[j],q,target,8192)+extract_paths(near_hist[j],q,target,32768)
                times['near']+=time.perf_counter()-t
                t=time.perf_counter()
                f=self._far_records(int(pos),q,target,mode,
                    far_hist[j] if far_hist is not None else None)
                times['far']+=time.perf_counter()-t
                all_records.extend(n+f)
                near_rows.append(n);far_rows.append(f)
                near_prob.append([r['target_probability'] for r in n]);near_count.append([r['n'] for r in n])
                far_prob.append([r['target_probability'] for r in f]);far_count.append([r['n'] for r in f])
            t=time.perf_counter()
            modal=torch.tensor([r['modal'] for r in all_records],dtype=torch.long)
            prior=torch.exp(-self.model.trunk.nll(now.cpu().repeat_interleave(12,0),modal).reshape(len(inds),12)).numpy()
            for j in range(len(inds)):
                q=ctx[j].numpy()
                near_rows[j]=make_feature(near_rows[j],prior[j,:8],q)
                far_rows[j]=make_feature(far_rows[j],prior[j,8:],q,far=True)
            nd={'base':base_loss.cpu(),
                'features':torch.tensor(np.asarray(near_rows),dtype=torch.float32),
                'prob':torch.tensor(np.asarray(near_prob),dtype=torch.float32),
                'count':torch.tensor(np.asarray(near_count),dtype=torch.float32)}
            nnll,_=near_nll(self.near,nd)
            fd={'near':nnll,'far_features':torch.tensor(np.asarray(far_rows),dtype=torch.float32),
                'far_prob':torch.tensor(np.asarray(far_prob),dtype=torch.float32),
                'far_count':torch.tensor(np.asarray(far_count),dtype=torch.float32)}
            fnll,fg=loss_far(self.far,fd)
            times['prior_and_gate']+=time.perf_counter()-t
            original.extend(base_loss.cpu().tolist())
            nearloss.extend(nnll.tolist());farll.extend(fnll.tolist())
            if return_details:
                for j,pos in enumerate(inds):
                    details.append({'position':int(pos),'far_features':far_rows[j],
                        'far_count':far_count[j],'far_prob':far_prob[j],
                        'gate_mass':float(fg[j].sum())})
        return {'positions':positions.tolist(),'mode':mode,'requested_mode':requested_mode,'baseline_nll':original,
                'near_nll':nearloss,'far_nll':farll,'profile_s':times,
                'resets':self.stats['resets'],'steps':self.stats['incremental_steps'],
                **({'details':details} if return_details else {})}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--near',default='/mnt/data/fa1r719/adaptation/fa1_r716_225.pt')
    p.add_argument('--far',default='/mnt/data/fa1r719/adaptation/fa1_r718_full_candidate.pt')
    p.add_argument('--split',choices=['train','dev','test'],default='test')
    p.add_argument('--positions',type=int,default=4)
    p.add_argument('--start',type=int,default=134000)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--mode',choices=['auto','streaming_index','packed_index','exact_reference'],default='auto')
    p.add_argument('--output',default=None)
    a=p.parse_args();ts=time.monotonic()
    model,opts,base,comp,near,far=restore_frozen(a.root,a.near,a.far)
    rt=FA1R720Runtime(a.root,model,base,comp,near,far,split=a.split)
    ret=rt.score_positions(np.arange(a.start,a.start+a.positions),mode=a.mode,batch_size=a.batch)
    report={'round':'FA1 R7.20','promotion':False,'accepted_step':14416,
        'five_optimizer_histories_verified':len(opts),'elapsed_s':time.monotonic()-ts,
        'mean_baseline_nll':float(np.mean(ret['baseline_nll'])),
        'mean_near_nll':float(np.mean(ret['near_nll'])),
        'mean_far_nll':float(np.mean(ret['far_nll'])),'result':ret}
    if a.output:pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()