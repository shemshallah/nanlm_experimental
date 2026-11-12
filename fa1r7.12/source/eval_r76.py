#!/usr/bin/env python3
"""Fixed-stride historical R7.6 ablations (no parameter updates)."""
import argparse,json,pathlib,torch
from r76_pair_training import restore,evaluate
from r7_core import Corpus

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
 ap.add_argument('--split',choices=('dev','test'),default='test')
 ap.add_argument('--count',type=int,default=4096);ap.add_argument('--batch',type=int,default=8)
 ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
 m,o,b,c,old,prior=restore(a.root,a.root/'checkpoint/pal_active.pt',torch.device('cpu'))
 results={'split':a.split,'checkpoint_joint_step':prior['joint_steps'],
          'historical_only':True,'independently_screened':False,
          'metrics':evaluate(m,b,c,Corpus(a.root,a.split),a.count,a.batch)}
 path=a.root/f'reports/R7p6_{a.split}_historical_{a.count}.json'
 path.write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
