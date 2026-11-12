#!/usr/bin/env python3
"""Independent evaluation horizon: R22 frozen 1000-update adapter, 400 chunks.

No tuning at this horizon. Causal history can be 32,768 token positions from the
same text before this measured window; inherited decoder still runs locally.
"""
import argparse,json,pathlib,sys,time,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r13_corpus_cache import read_books_cached
from r11_raw import windows
from r21_relevance_novelty import RelevanceNoveltyGate
from r22_multiscale_memory import R22Residual
from r22_cpu_longmemory import build_cache,evaluate

def run(root,n):
 t=time.perf_counter();root=pathlib.Path(root);torch.set_num_threads(2)
 accepted=root/'checkpoint/r12_active.pal12';original=pal.sha(accepted)
 sp=root/'reports/fa2_2_r21_gate_candidate.pth';gatesha=pal.sha(sp)
 state21=torch.load(sp,weights_only=False,map_location='cpu')
 state22=torch.load(root/'reports/r22_multiscale_candidate.pth',weights_only=False,map_location='cpu')
 assert state21['parent_checkpoint_sha256']==state22['parent_checkpoint_sha256']==original
 assert state22['frozen_r21_gate_sha256']==gatesha and state22['steps']==1000
 gate21=RelevanceNoveltyGate();gate21.load_state_dict(state21['gate']);gate21.eval()
 gate22=R22Residual();gate22.load_state_dict(state22['model']);gate22.eval()
 saved,r6,comp,model,bridge,adapter,opt,_=pal.load_for_arm(root,accepted,True)
 net=R18LongMemory(model.memory.piece_emb.weight);net.load_state_dict(saved['r18_memory'])
 for x in (model,bridge,adapter,net,gate21):x.eval().requires_grad_(False)
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 choices=windows(docs,400,500,2000);out={}
 for split in ('dev','test'):
  chosen=choices[split][:n]
  data=build_cache(split,docs[split],chosen,400,model,bridge,adapter,net,gate21,
                   cache_dir=root/'.cache'/'r22_400'/split)
  out[split]=evaluate(gate22,data)
  print(json.dumps({'phase':'EVAL','split':split,'windows':len(chosen),
                    'scores':out[split]['scores'],
                    'maximum_retrieval_distance':out[split]['maximum_retrieval_distance']}),flush=True)
 assert pal.sha(accepted)==original and pal.sha(sp)==gatesha
 report={'round':'FA2.3 R22 frozen 1000-state 400-chunk horizon test','accepted_sha256':original,
         'r21_state_sha256':gatesha,'r22_step':1000,'promoted':False,'read_only':True,
         'samples_per_split':n,'results':out,'elapsed_seconds':time.perf_counter()-t,
         'limitations':['Historically reused book-disjoint splits; not untouched generalization',
                        '32k externally addressed prefix slots; not a 32k-token trained transformer',
                        'No semantic QA or multi-hop benchmark here']}
 (root/'reports/r22_horizon400.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'FINAL','elapsed_seconds':report['elapsed_seconds'],'promoted':False}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--windows',type=int,default=8)
 a=p.parse_args();run(a.root,a.windows)
