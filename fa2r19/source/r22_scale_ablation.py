#!/usr/bin/env python3
"""Read-only R22 400-chunk ablation across each 32k memory tier."""
import copy,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r22_multiscale_memory import R22Residual,BANDS
from r22_cpu_longmemory import evaluate


def main(root):
 root=pathlib.Path(root);torch.set_num_threads(2)
 original=pal.sha(root/'checkpoint/r12_active.pal12')
 state=torch.load(root/'reports/r22_multiscale_candidate.pth',map_location='cpu',weights_only=False)
 assert state['parent_checkpoint_sha256']==original and state['steps']==1000
 cache=root/'.cache'/'r22_400'
 scores={}
 for split in ('dev','test'):
  references=[{'cache_path':str(p)} for p in sorted((cache/split).glob(f'{split}_400_*.pth'))]
  assert len(references)==8
  scores[split]={}
  for configuration,active in [('R21_none',()),('R22_all',BANDS),('recent_only',('recent',)),
                               ('middle_only',('middle',)),('far_only',('far',)),
                               ('without_recent',('middle','far')),
                               ('without_middle',('recent','far')),
                               ('without_far',('recent','middle'))]:
   model=R22Residual();model.load_state_dict(state['model']);model.eval()
   with torch.no_grad():
    for name in BANDS:
     if name not in active:model.branches[name].amplitude.zero_()
   results=evaluate(model,references)['scores']['r22']
   scores[split][configuration]={k:results[k] for k in ('bpb','nonspace_nll','nonspace_top1')}
  baseline=scores[split]['R21_none']
  for k,v in scores[split].items():
   v['delta_bpb_vs_r21']=v['bpb']-baseline['bpb']
 print(json.dumps({'phase':'RESULTS','scores':scores}),flush=True)
 report={'round':'FA2.3 R22 frozen memory scale ablations','accepted_sha256':original,'step':1000,
         'read_only':True,'promoted':False,'window_span_chunks':400,
         'book_disjoint_development_and_historical_test':True,'windows_per_split':8,
         'scores':scores,'warnings':['Historical test books reused; cannot use to tune hyperparameters',
           'Independent scale amplitude zeroing measures marginal contribution only and is not a retrained control']}
 (root/'reports/r22_scale_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
 assert pal.sha(root/'checkpoint/r12_active.pal12')==original
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '.')
