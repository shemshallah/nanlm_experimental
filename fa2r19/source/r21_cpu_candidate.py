#!/usr/bin/env python3
"""Train only FA2.2's R21 memory gate on genuine inherited NaNROUTER book tokens.

Single accepted R18 model stays untouched. Frozen gate-proposal features are
cached once; candidate learned via NLL without resetting any inherited Adam
histories. Select on development ONLY; test once after selection. No gate is
promoted unless long+short BPB and non-space NLL all improve and new-book QA
and legacy guards are met. The latter are not part of this experiment, so the
candidate is *always* quarantined in reports/.
"""
import argparse,copy,hashlib,json,math,pathlib,random,sys,time
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_joint_continue import integrated_logits
from r18_long_memory import R18LongMemory
from r13_corpus_cache import read_books_cached
from r11_raw import batch,windows
from r21_relevance_novelty import RelevanceNoveltyGate,causal_proposal,FEATURE_NAMES

def build_cache(docs,choices,span,model,bridge,adapter,net):
 cache=[]
 with torch.no_grad():
  for pick in choices:
   ex,lc,targets,tags,bytes_=batch(docs,[pick],span)
   n=int((targets[0]!=-100).sum());y=targets[0,:n].clone()
   base=integrated_logits(model,bridge,adapter,net,ex,lc,tags,targets)[0,:n].detach().clone()
   d,f,meta=causal_proposal(net,base,y)
   cache.append({'logits':base,'proposal':d,'features':f,'targets':y,'bytes':int(bytes_),
                 'enabled':meta['eligible'],'book_index':pick[0]})
 return cache

@torch.no_grad()
def evaluate(gate,cache):
 totals={arm:{'loss':0.,'other_loss':0.,'bytes':0,'tokens':0,'nonspace':0,'correct':0,
               'nonspace_correct':0} for arm in ('r18','r20_gain1','r21_adaptive')}
 gate.eval()
 for w in cache:
  x=w['targets'];mask=x!=32
  new,selected_gates=gate(w['logits'],w['proposal'],w['features'])
  for k,z in (('r18',w['logits']),('r20_gain1',w['logits']+w['proposal']),('r21_adaptive',new)):
   ce=F.cross_entropy(z,x,reduction='none');rec=totals[k]
   rec['loss']+=float(ce.sum());rec['other_loss']+=float(ce[mask].sum());rec['bytes']+=w['bytes']
   rec['tokens']+=len(x);rec['nonspace']+=int(mask.sum())
   pred=z.argmax(-1);rec['correct']+=int((pred==x).sum());rec['nonspace_correct']+=int(((pred==x)&mask).sum())
 return {k:{'bpb':a['loss']/a['bytes']/math.log(2),
            'nonspace_nll':a['other_loss']/a['nonspace'],
            'nonspace_top1':a['nonspace_correct']/a['nonspace'],
            'tokens':a['tokens'],'bytes':a['bytes']} for k,a in totals.items()}

def run(args):
 start=time.perf_counter()
 torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
 root=pathlib.Path(args.root).resolve();path=root/'checkpoint/r12_active.pal12'
 original=pal.sha(path)
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,path,True)
 assert 'r18_memory' in saved and len(opt.param_groups)==4
 net=R18LongMemory(model.memory.piece_emb.weight)
 net.load_state_dict(saved['r18_memory'])
 for n in (model,bridge,adapter,net):n.eval();n.requires_grad_(False)
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 short=windows(docs,24,8000,180)
 long=windows(docs,args.long_span,max(64,args.train_windows,args.dev_windows,args.test_windows),args.max_tokens)
 # No test reads or selection prior to freezing learned gate.
 train=build_cache(docs['train'],long['train'][:args.train_windows],args.long_span,model,bridge,adapter,net)
 dev=build_cache(docs['dev'],long['dev'][:args.dev_windows],args.long_span,model,bridge,adapter,net)
 short_dev=build_cache(docs['dev'],short['dev'][:args.short_windows],24,model,bridge,adapter,net)
 gate=RelevanceNoveltyGate()
 optim=torch.optim.AdamW(gate.parameters(),lr=args.lr,weight_decay=1e-4)
 resumed=None
 if args.resume_state:
  resumed=torch.load(args.resume_state,map_location='cpu',weights_only=False)
  assert resumed['parent_checkpoint_sha256']==original, 'Refuse wrong inherited model'
  assert resumed['train_windows']==args.train_windows and resumed['long_span']==args.long_span
  gate.load_state_dict(resumed['gate']);optim.load_state_dict(resumed['optimizer'])
 initial={'long':evaluate(gate,dev),'short':evaluate(gate,short_dev)}
 best=copy.deepcopy(gate.state_dict()) if resumed else None
 best_opt=copy.deepcopy(optim.state_dict()) if resumed else None
 best_loss=resumed['best_loss'] if resumed else float('inf')
 best_step=resumed['step'] if resumed else 0
 trace=[]
 for step in range(best_step+1,args.steps+1):
  gate.train();ix=[(step*7+k*11+args.seed)%len(train) for k in range(args.batch)]
  loss=0.
  for j in ix:
   item=train[j];out,_=gate(item['logits'],item['proposal'],item['features'])
   loss=loss+F.cross_entropy(out,item['targets'])
  loss/=len(ix)
  optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),5.)
  optim.step()
  if step%args.eval_every==0 or step==args.steps:
   vals=evaluate(gate,dev);long=vals['r21_adaptive'];base=vals['r18'];shortvals=evaluate(gate,short_dev)
   # Select gate by dev long BPB, and tie-break via nonspace. No test consulted.
   key=long['bpb']+0.05*max(0.,long['nonspace_nll']-base['nonspace_nll'])
   if key<best_loss:
    best_loss=key;best=copy.deepcopy(gate.state_dict());best_opt=copy.deepcopy(optim.state_dict());best_step=step
   trace.append({'step':step,'train_ce':float(loss.detach()),'long_dev_bpb':long['bpb'],
                 'long_dev_other_nll':long['nonspace_nll'],
                 'short_dev_bpb':shortvals['r21_adaptive']['bpb'],
                 'amplitude':float(gate.amplitude.detach())})
   print(json.dumps({'phase':'DEV','step':step,'dev_long_bpb':long['bpb'],
                     'dev_short_bpb':shortvals['r21_adaptive']['bpb']}),flush=True)
 gate.load_state_dict(best);optim.load_state_dict(best_opt);gate.eval()
 candidate_file=root/'reports/fa2_2_r21_gate_candidate.pth'
 torch.save({'parent_checkpoint_sha256':original,'gate':best,'optimizer':best_opt,
             'step':best_step,'best_loss':best_loss,'train_windows':args.train_windows,
             'long_span':args.long_span,'accepted_model_promoted':False},candidate_file)
 final_dev={'long':evaluate(gate,dev),'short':evaluate(gate,short_dev)}
 # Book-disjoint historical test read exactly once, after dev model selection.
 testlong=build_cache(docs['test'],windows(docs,args.long_span,max(64,args.test_windows),args.max_tokens)['test'][:args.test_windows],
                      args.long_span,model,bridge,adapter,net)
 testshort=build_cache(docs['test'],short['test'][:args.short_windows],24,model,bridge,adapter,net)
 historical={'long':evaluate(gate,testlong),'short':evaluate(gate,testshort)}
 def gains(r):
  return {k: r['r21_adaptive'][k]-r['r18'][k] for k in ('bpb','nonspace_nll','nonspace_top1')}
 # This is not a clean, unseen-book semantic result or a promotion candidate.
 safe=pal.sha(path)==original
 assert safe,'Original accepted checkpoint changed'
 report={'round':'FA2.2 R21 adaptive relevance novelty gate','parent_checkpoint_sha256':original,
         'checkpoint_untouched':safe,'checkpoint_promoted':False,
         'status':'quarantined CPU candidate; no legacy Adam moments reset',
         'new_gate_parameters':sum(p.numel() for p in gate.parameters()),
         'gate_amplitude':float(gate.amplitude.detach()),
         'features':FEATURE_NAMES,'train_long_windows':len(train),
         'dev_long_windows':len(dev),'test_long_windows':len(testlong),
         'dev_short_windows':len(short_dev),'test_short_windows':len(testshort),
         'steps_attempted':args.steps,'best_step_dev':best_step,'resumed_from_step':resumed['step'] if resumed else 0,
         'gate_candidate_file':candidate_file.name,'baseline':initial,
         'dev_selected':final_dev,'historical_test':historical,
         'changes_vs_r18':{'dev_long':gains(final_dev['long']),'dev_short':gains(final_dev['short']),
                           'test_long':gains(historical['long']),'test_short':gains(historical['short'])},
         'dev_gates':{f'{s}_{m}_improved':final_dev[s]['r21_adaptive'][m]<final_dev[s]['r18'][m]
                     for s in ('short','long') for m in ('bpb','nonspace_nll')},
         'gate_state_dict':{k:v.detach().cpu().tolist() for k,v in gate.state_dict().items()},
         'trace':trace,'seconds':time.perf_counter()-start,
         'limits':['Historical test books were accessed in prior PAL research; not clean new-book evaluation',
                   'No independent paraphrase, semantic hard-negative or multi-hop gain established',
                   'Frozen decoder and inherited Adam states are unchanged; only new gate optimizer was trained',
                   'No model promotion without new-book improvement, legacy guards, and semantic transfer']}
 dest=root/'reports/r21_fa2_2_cpu_candidate.json';dest.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'FINAL','steps':args.steps,'best_step':best_step,
                   'changes':report['changes_vs_r18'],'checkpoint_untouched':safe,
                   'promoted':False,'seconds':report['seconds']}),flush=True)
 return report

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.')
 ap.add_argument('--steps',type=int,default=100);ap.add_argument('--batch',type=int,default=2)
 ap.add_argument('--lr',type=float,default=0.003);ap.add_argument('--eval-every',type=int,default=20)
 ap.add_argument('--long-span',type=int,default=120);ap.add_argument('--max-tokens',type=int,default=700)
 ap.add_argument('--train-windows',type=int,default=16);ap.add_argument('--dev-windows',type=int,default=6)
 ap.add_argument('--test-windows',type=int,default=6);ap.add_argument('--short-windows',type=int,default=10)
 ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=21022)
 ap.add_argument('--resume-state',default='')
 run(ap.parse_args())