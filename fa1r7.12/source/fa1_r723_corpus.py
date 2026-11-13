"""Target-free context preparation for R7's full point/pair/triple Diamond."""
from __future__ import annotations
import numpy as np,torch
from r7_core import Corpus

def diamond_context(root,base,split,positions,hidden,corpus_dir=None):
    c=Corpus(root,split,corpus_dir=corpus_dir)
    positions=np.asarray(positions,dtype=np.int64)
    # Every retrieved token has index strictly before the current target.
    hist=c.get(positions,context=128)[0].astype(np.int64)
    word=hist[:,-32:].copy()
    vertices=[];bytes_in=[];sent_in=[];byte_target=[]
    # Only targets for auxiliary loss, never for the model's feature path.
    point_emb=base.H
    for ix,p in enumerate(positions):
        past=hist[ix]
        spelling=[base.vocab[int(k)] if 0<=int(k)<len(base.vocab) else '<unk>' for k in past]
        recent=' '.join(spelling[-48:]).encode('utf-8')[-128:]
        by=np.zeros(128,dtype=np.int64);by[-len(recent):]=np.frombuffer(recent,dtype=np.uint8) if len(recent) else []
        bytes_in.append(by)
        # A sentence head is the word following . ! ? (8 last such heads).
        sent=[1]
        for j in range(1,len(spelling)):
            if spelling[j-1] in ('.','!','?'):
                sent.append(int(past[j]))
        sent=( [1]*8 + sent[-8:] )[-8:]
        sent_in.append(sent)
        points=[]
        # Preserve the active R7 legacy pair selected by x2->head0, x1->head1.
        ex=c.data[p]
        anchors=(int(base.heads[int(ex[2]),0]),int(base.heads[int(ex[1]),1]))
        for q in anchors:
            if q not in points:points.append(q)
        for wi in reversed(past[-16:]):
            h0,h1=base.heads[int(wi)]
            for h in (int(h0),int(h1)):
                if h not in points:points.append(h)
                if len(points)==8:break
            if len(points)==8:break
        # If too many repetitions, choose target-free R7 point scores.
        if len(points)<8:
            preference=np.argsort(-(point_emb@hidden[ix].detach().numpy()))
            for h in preference:
                if int(h) not in points:points.append(int(h))
                if len(points)==8:break
        vertices.append(points)
        # The word vocabulary maps out-of-vocabulary words to ID zero.
        # The original UTF-8 spelling is retained in the corpus character
        # sidecar: use it for the *label*, or byte accuracy is falsified by
        # predicting the literal '<unk>' spelling for every unknown word.
        tw=c.spelling(int(p),base.vocab)
        first=tw.encode('utf-8')[:1]
        byte_target.append(int(first[0]) if first else 0)
    return {'word_ids':torch.from_numpy(word).long(),
            'byte_ids':torch.from_numpy(np.stack(bytes_in)).long(),
            'sentence_ids':torch.tensor(sent_in,dtype=torch.long),
            'vertex_ids':torch.tensor(vertices,dtype=torch.long),
            'first_byte_target':torch.tensor(byte_target,dtype=torch.long)}
