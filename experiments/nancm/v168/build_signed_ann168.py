#!/usr/bin/env python3
"""Pack TRAIN-only address neighbors and inherited v166 S+/S- geometry for native v168.
Never reads DEV/TEST or mutates inherited weights and optimizers.
"""
import hashlib,math,struct,sys,json,time
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'experiments/nancm/v166')]
from train_birth166 import fnv
T=time.monotonic();torch.set_num_threads(1)
model=torch.load(R/'checkpoints/v166_lexical_birth.pt',map_location='cpu',weights_only=False)
vocab={w:i+1 for i,w in enumerate(model['vocab'])};emb=model['model']['ctx.weight'].numpy()
z=model['model'];l=z['lower'].numpy();u=z['upper'].numpy();p=z['plus'].numpy();m=z['minus'].numpy()
f=(p*np.tanh(l[None,:,:]+u[:,None,:])-m*np.tanh(l[None,:,:]-u[:,None,:])+l[None,:,:]*u[:,None,:]/np.sqrt(8.)).sum(-1).T
U,S,VT=np.linalg.svd(f,full_matrices=False)
signed=(f@VT[:16,:].T).astype(np.float32)
signed/=np.linalg.norm(signed,axis=1,keepdims=True).clip(1e-9)
index=R/'checkpoints/v164_native_distribution.tsv'
with index.open() as ff:
    hdr=ff.readline().strip().split('\t')
    assert len(hdr)==4 and hdr[0]=='N164DIST1'
    words=set();nkeys=0
    for line in ff:
        k,ctx,w,n,total=line.rstrip('\n').split('\t');words.add(w)
        # TSV is grouped by key and might contain same context under different orders.
        if k+':'+ctx!=locals().get('prev'):nkeys+=1;prev=k+':'+ctx
lexicon=sorted(words);wit={w:i for i,w in enumerate(lexicon)}
print('SIDE WORDS',len(lexicon),'INDEX KEYS',nkeys,flush=True)
assert nkeys==199956
side=R/'checkpoints/v168_signed_ann.bin'
with side.open('wb') as out:
    out.write(b'N168ANN1')
    out.write(struct.pack('<3Q',*[int(x) for x in hdr[1:]]))
    out.write(struct.pack('<4I',nkeys,len(lexicon),32,16))
    for w in lexicon:
        assert 0<len(w)<=20 and w.isascii() and w.islower() and w.isalpha()
        out.write(struct.pack('<B',len(w))+w.encode('ascii'))
    out.write(signed.astype('<f4',copy=False).tobytes(order='C'))
    with index.open() as ff:
        ff.readline()
        old=None;group=[];k_prev=None;ctx_prev=None;num=0
        def packrow():
            global num
            if old is None:return
            assert len(group)<=8
            ctx=ctx_prev.split();weights=[.2,.3,.5][3-len(ctx):];den=sum(weights)
            q=np.zeros(32,np.float32);ss=np.zeros(16,np.float32)
            for w,r in zip(ctx,weights):q+=r*emb[vocab.get(w,0)];ss+=r*signed[fnv(w)]
            q/=den;ss/=den
            q/=max(1e-9,float(np.linalg.norm(q)));ss/=max(1e-9,float(np.linalg.norm(ss)))
            out.write(q.astype('<f4').tobytes()+ss.astype('<f4').tobytes())
            out.write(struct.pack('<BB',k_prev,len(group)))
            for i in range(8):
                w,prob=group[i] if i<len(group) else (lexicon[0],0.)
                out.write(struct.pack('<If',wit[w],prob))
            num+=1
        for line in ff:
            k,ctx,w,c,total=line.rstrip('\n').split('\t');key=k+':'+ctx
            if old!=key:
                packrow();old=key;k_prev=int(k);ctx_prev=ctx;group=[]
            group.append((w,int(c)/int(total)))
        packrow()
        assert num==nkeys,num
sha=hashlib.sha256(side.read_bytes()).hexdigest()
meta={'format':'N168ANN1','train_index_sha':hashlib.sha256(index.read_bytes()).hexdigest(),
      'inherited_native_fp':[int(x) for x in hdr[1:]],'source_v166_sidecar_sha':hashlib.sha256((R/'checkpoints/v166_lexical_birth.bin').read_bytes()).hexdigest(),
      'addresses':nkeys,'candidate_entries':733069,'unique_targets':len(lexicon),'key_components':'32D 3word causal learned lexical + 16D S+/S- signed SVD orientation','hash_sha256':sha,
      'bytes':side.stat().st_size,'seconds':time.monotonic()-T,'no_dev_or_test_in_pack':True}
(R/'results/v168_signed_ann_build.json').write_text(json.dumps(meta,indent=2))
print('FINAL',json.dumps(meta),flush=True)