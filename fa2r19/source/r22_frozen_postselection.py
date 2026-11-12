#!/usr/bin/env python3
"""Read-only R22 1k-state evaluation; no retraining and NO promotion."""
import argparse, json, pathlib, sys, time, torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r13_corpus_cache import read_books_cached
from r11_raw import windows
from r21_relevance_novelty import RelevanceNoveltyGate
from r22_multiscale_memory import R22Residual
from r22_cpu_longmemory import build_cache, evaluate

def run(root,count):
 root=pathlib.Path(root);t=time.perf_counter();torch.set_num_threads(2)
 ck=root/'checkpoint/r12_active.pal12';sha=pal.sha(ck)
 r21p=root/'reports/fa2_2_r21_gate_candidate.pth';r21hash=pal.sha(r21p)
 old=torch.load(r21p,map_location='cpu',weights_only=False)
 new=torch.load(root/'reports/r22_multiscale_candidate.pth',map_location='cpu',weights_only=False)
 assert old['parent_checkpoint_sha256']==new['parent_checkpoint_sha256']==sha
 assert new['frozen_r21_gate_sha256']==r21hash and new['steps']==1000
 gate21=RelevanceNoveltyGate();gate21.load_state_dict(old['gate']);gate21.eval()
 gate22=R22Residual();gate22.load_state_dict(new['model']);gate22.eval()
 saved,r6,comp,model,bridge,adapter,opt,_=pal.load_for_arm(root,ck,True)
 net=R18LongMemory(model.memory.piece_emb.weight);net.load_state_dict(saved['r18_memory'])
 for x in (model,bridge,adapter,net,gate21):x.eval().requires_grad_(False)
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 picks=windows(docs,120,96,700);short=windows(docs,24,8000,180)
 cache_dir=root/'.cache'/'r22_s120_seed22022'
 out={}
 for split,kind,choices,span in [('dev','long',picks['dev'][:16],120),('dev','short',short['dev'][:8],24),
                                 ('test','long',picks['test'][:count],120),('test','short',short['test'][:8],24)]:
  data=build_cache(split,docs[split],choices,span,model,bridge,adapter,net,gate21,
                   cache_dir=cache_dir/('short_'+split if kind=='short' else split))
  out[split+'_'+kind]=evaluate(gate22,data)
  print(json.dumps({'phase':'EVAL','split':split,'kind':kind,'windows':len(data),'metrics':out[split+'_'+kind]['scores']}),flush=True)
 assert pal.sha(ck)==sha and pal.sha(r21p)==r21hash
 report={'round':'FA2.3 R22 1000-step frozen selected postselection','parent_sha256':sha,
         'r21_state_sha256':r21hash,'adapter_steps':1000,'checkpoint_promoted':False,
         'no_retraining':True,'context_source':'NaNROUTER source-verified distinct books',
         'results':out,'seconds':time.perf_counter()-t,
         'limits':['Previously accessed historical test books, not independent new-book holdout',
                   '32K sparse memory addresses, not 32K dense model context',
                   'Only small new residual trained; original decoder and inherited optimizers frozen',
                   'No paraphrase or multi-hop competence demonstrated']}
 (root/'reports/r22_frozen_1000_report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'FINAL','duration':report['seconds'],'promoted':False}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--test-windows',type=int,default=8)
 a=p.parse_args();run(a.root,a.test_windows)
