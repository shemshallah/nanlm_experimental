#!/usr/bin/env python3
"""FA2.2: read-only frozen R21 gate at R20's full 400-chunk context horizon."""
import argparse,hashlib,json,pathlib,sys,time,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r13_corpus_cache import read_books_cached
from r11_raw import windows
from r21_cpu_candidate import build_cache,evaluate
from r21_relevance_novelty import RelevanceNoveltyGate

def main(root,count):
 torch.set_num_threads(2);start=time.perf_counter();root=pathlib.Path(root)
 ck=root/'checkpoint/r12_active.pal12';original=pal.sha(ck)
 state=torch.load(root/'reports/fa2_2_r21_gate_candidate.pth',weights_only=False,map_location='cpu')
 assert state['parent_checkpoint_sha256']==original and state['step']==1000
 gate=RelevanceNoveltyGate();gate.load_state_dict(state['gate']);gate.eval()
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ck,True)
 net=R18LongMemory(model.memory.piece_emb.weight);net.load_state_dict(saved['r18_memory'])
 for n in (model,bridge,adapter,net):n.eval();n.requires_grad_(False)
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 choices=windows(docs,400,500,2000)
 outputs={}
 for split in ('dev','test'):
  samples=build_cache(docs[split],choices[split][:count],400,model,bridge,adapter,net)
  metric=evaluate(gate,samples)
  outputs[split]={'results':metric,'deltas_vs_r18':{key:metric['r21_adaptive'][key]-metric['r18'][key]
                             for key in ('bpb','nonspace_nll','nonspace_top1')},
                  'delta_r21_vs_r20_gain1':{key:metric['r21_adaptive'][key]-metric['r20_gain1'][key]
                             for key in ('bpb','nonspace_nll')},
                  'samples':len(samples)}
 assert pal.sha(ck)==original
 report={'round':'FA2.2 R21 full 400-chunk horizon gate-only postselected frozen probe',
         'accepted_sha256':original,'gate_steps':1000,'checkpoint_promoted':False,
         'eval_only':True,'horizon_chunks':400,'results':outputs,
         'seconds':time.perf_counter()-start,
         'limitations':['Test books previously used by PAL (not untouched fresh data)',
                        'Same 400-chunk R20 windows; no threshold or parameter selection on this probe',
                        'No evidence of semantic paraphrase or end-to-end long-document reasoning']}
 (root/'reports/r21_400_horizon_probe.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--windows',type=int,default=8)
 x=p.parse_args();main(x.root,x.windows)
