#!/usr/bin/env python3
"""Strict causal held-out-book proposal evaluation for v166.
Ground truth is used only after producing rankings, never as model input.
"""
import argparse, hashlib, json, math, re, sys, time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_birth166 import Birth166,fnv
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import getwords

def collect(split='dev',count=24,perbook=20):
    rows=[]
    for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
        if bi>=count:break
        words=getwords(raw,skip)
        if perbook:
            words=[words[min(len(words)-1,int((j+.5)*len(words)/min(len(words),perbook)))] for j in range(min(len(words),perbook))] if words else []
        for pos,truth in words:
            prev=[m.group().lower().decode('ascii') for m in re.finditer(rb'[A-Za-z]+',raw[:pos]) if m.end()<pos]
            rows.append({'book':bi,'position':pos,'truth':truth,'past':prev[-3:]})
    return rows

def evaluate(p,split='dev',count=24,perbook=20,device='cpu'):
    pack=torch.load(p,map_location='cpu',weights_only=False)
    words=pack['vocab'];v={w:i+1 for i,w in enumerate(words)}
    model=Birth166(len(words),np.zeros(len(words)+1,dtype=np.float32),pack['model'])
    model.load_state_dict(pack['model']);model.eval()
    rows=collect(split,count,perbook)
    wc=np.zeros((len(rows),3),dtype=np.int64);hc=np.zeros_like(wc)
    for i,r in enumerate(rows):
        for j,w in enumerate(r['past']):
            k=3-len(r['past'])+j;wc[i,k]=v.get(w,0);hc[i,k]=fnv(w)+1
    m=torch.from_numpy(wc);h=torch.from_numpy(hc)
    truth=[r['truth'] for r in rows];n=len(rows)
    predictions={};elapsed=time.monotonic()
    def score(mode):
        out=[];loss=[]
        with torch.no_grad():
            for a in range(0,n,64):
                b=min(n,a+64)
                if mode=='normal':logits=model(m[a:b],h[a:b])
                elif mode=='prior':logits=model.bias[None,:].expand(b-a,-1).clone()
                elif mode=='no_concept':
                    logits=model(m[a:b],h[a:b])-model.concept(h[a:b]).index_select(1,model.hashes)*model.logcloud.exp()
                elif mode=='no_lex':
                    mm=torch.zeros_like(m[a:b]);logits=model(mm,h[a:b])
                elif mode=='shuffle_history':
                    ix=(torch.arange(a,b)+37)%n
                    logits=model(m[ix],h[ix])
                logits[:,0]=-1.e6
                top=logits.topk(min(96,len(words)),dim=-1).indices.tolist()
                out.extend([[words[j-1] for j in pred if j>0] for pred in top])
                logp=torch.log_softmax(logits,dim=-1)
                loss.extend([-logp[i,v.get(truth[a+i],0)].item() if v.get(truth[a+i],0) else float('nan') for i in range(b-a)])
        covered=[i for i,w in enumerate(truth) if w in v]
        good=np.array([loss[i] for i in covered]);metrics={'vocabulary_hit_rate':len(covered)/n,'nll_in_vocab_nats':float(good.mean()),'perplexity_in_vocab':float(np.exp(good.mean())),'n':n}
        for k in (1,5,16,32,64,96):
            metrics['top'+str(k)+'_hits']=sum(w in x[:k] for w,x in zip(truth,out))
            metrics['top'+str(k)+'_pct']=round(100*metrics['top'+str(k)+'_hits']/n,3)
        predictions[mode]=out
        return metrics
    report={'source':str(p),'split':split,'books':count,'sample':'deterministic same 20 positions per DEV book as v164','no_target_in_context':True,'rows':n,
            'index_provenance_sha256':pack['train_index_sha'],'ablations':{mode:score(mode) for mode in ('prior','no_lex','no_concept','normal','shuffle_history')},
            'elapsed_s':round(time.monotonic()-elapsed,2)}
    baseline=R/'results/v164_dev_rows.jsonl'
    if split=='dev' and baseline.exists() and count==24 and perbook==20:
        parent=[json.loads(line) for line in baseline.open()];assert len(parent)==n
        assert all(x['word']==w for x,w in zip(parent,truth)),'baseline samples misaligned'
        pwords=[x['predictions'].get('v164_combo_b0.6',[]) for x in parent]
        if not any(pwords):pwords=[x['predictions'].get('v164_b0.6',[]) for x in parent]
        report['paired_parent_v164_top32_hits']=sum(w in pp[:32] for w,pp in zip(truth,pwords))
        for k in (5,32,64):
            report[f'union_v166_{k}_plus_parent_v164_32_hits']=sum(w in set(pp[:32])|set(x[:k]) for w,pp,x in zip(truth,pwords,predictions['normal']))
            report[f'new_correct_word_not_in_parent_top32_{k}']=sum(w in x[:k] and w not in pp[:32] for w,pp,x in zip(truth,pwords,predictions['normal']))
    return report,rows,predictions

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['dev','test'],default='dev');ap.add_argument('--books',type=int,default=24);ap.add_argument('--per-book',type=int,default=20);a=ap.parse_args()
    report,_,_=evaluate(R/'checkpoints/v166_lexical_birth.pt',a.split,a.books,a.per_book)
    path=R/f'results/v166_{a.split}_proposal.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)