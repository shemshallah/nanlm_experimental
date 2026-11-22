#!/usr/bin/env python3
"""TRAIN-calibrated candidate fusion of the v172 6k-step signed word teacher.
DEV is evaluated only after selecting on 20 TRAIN calibration books.
"""
import json,math,hashlib,collections,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
from train_conditional_cloud172 import load_parent, CloudNCE,dev_cohort,word_ids

def get_scores(model,contexts):
 model.eval()
 with torch.no_grad():
  W=model.out.weight.T;gain=model.gain.clamp(0,25);bias=model.bias
  out=[]
  for i in range(0,len(contexts),128):
   x=torch.tensor(contexts[i:i+128],dtype=torch.long)
   q=model.query(x);z=gain*q@W+bias
   out.append(torch.log_softmax(z,dim=1).cpu().numpy())
 return np.vstack(out)

def candidate_score(row,score,vocab,beta,augment=0,floor=1e-6):
 cands={c['word']:c['p'] for c in row['candidates']}
 if augment:
  for j in np.argpartition(score,-augment)[-augment:]:
   w=vocab[j];cands.setdefault(w,floor)
 ranking=sorted(cands,key=lambda w:-(math.log(max(1e-30,cands[w]))+beta*float(score[vocab.index(w)] if w in vocab else -40)))
 return ranking

def main():
 torch.set_num_threads(2)
 names,C,O,B,spatial,gain,fp=load_parent();mapping={w:i for i,w in enumerate(names)}
 ckpt=torch.load(ROOT/'checkpoints/v172_conditional_signed_cloud.pt',weights_only=False,map_location='cpu')
 model=CloudNCE(C,O,B,spatial,gain,names);model.load_state_dict(ckpt['state_dict'])
 train=[json.loads(x) for x in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
 pos=[json.loads(x) for x in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(train)]
 dev=[json.loads(x) for x in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
 contexts=[word_ids(' '.join([m.group().decode('ascii').lower() for m in __import__('re').finditer(rb'[A-Za-z]+',p['history'].encode('latin1'))][-3:]),mapping) for p in pos]
 D,Dy,_=dev_cohort(mapping)
 train_log=get_scores(model,np.asarray(contexts));dev_log=get_scores(model,D)
 calibration=set(sorted(set(r['book'] for r in train),key=lambda b:hashlib.sha256(('v172-calib-'+b).encode()).digest())[-20:])
 train_idx=[i for i,r in enumerate(train) if r['book'] not in calibration]
 val_idx=[i for i,r in enumerate(train) if r['book'] in calibration]
 print('TRAIN',len(train_idx),'VAL',len(val_idx),'DEV',len(dev),'NEURAL step',ckpt['step'],flush=True)
 def evaluate(rows,scores,indices,beta,augment,floor):
  cnt=collections.Counter()
  for i in indices:
   row=rows[i];sc=scores[i];base=row['baseline']
   cands={c['word']:c['p'] for c in row['candidates']}
   if augment:
    for j in np.argpartition(sc,-augment)[-augment:]:cands.setdefault(names[j],floor)
   def value(w):
    return math.log(max(1e-30,cands[w]))+beta*float(sc[mapping[w]] if w in mapping else -40)
   rank=sorted(cands,key=lambda w:(-value(w),w))
   cnt['n']+=1;cnt['baseline']+=base==row['truth'];cnt['top1']+=rank[0]==row['truth'];cnt['top5']+=row['truth'] in rank[:5];cnt['oracle']+=row['truth'] in cands
  return dict(cnt)
 options=[]
 for beta in (0,.02,.05,.1,.2,.4,.75,1.,1.5):
  for augment in (0,32,128):
   for floor in ((1e-7,1e-5,1e-3) if augment else (0.,)):
    ev=evaluate(train,train_log,val_idx,beta,augment,floor)
    options.append((ev['top1'],ev['top5'], -abs(beta),-augment,beta,augment,floor,ev))
 options.sort(reverse=True)
 print('TOPVAL',[(x[0],x[1],x[4],x[5],x[6]) for x in options[:12]],flush=True)
 selected=options[0];beta,aug,floor=selected[4:7]
 dev_report=evaluate(dev,dev_log,range(len(dev)),beta,aug,floor)
 print('DEV_FROZEN',{'selected':{'beta':beta,'augment':aug,'floor':floor},'dev':dev_report},flush=True)
 # Per-expert unconditioned accuracy and union diagnostic.
 exp={}
 for name,rows,scores in [('TRAIN-calibration',train,train_log),('DEV-reused',dev,dev_log)]:
  ids=val_idx if name=='TRAIN-calibration' else list(range(len(rows)))
  c=collections.Counter()
  for i in ids:
   row=rows[i];sc=scores[i];top=np.argpartition(sc,-128)[-128:];ordered=sorted(top,key=lambda j:-sc[j]);words=[names[j] for j in ordered]
   c['n']+=1;c['teacher_top1']+=row['truth'] in words[:1];c['teacher_top5']+=row['truth'] in words[:5];c['teacher_top32']+=row['truth'] in words[:32];c['teacher_top128']+=row['truth'] in words
   c['union128']+=row['truth'] in (set(words)|{x['word'] for x in row['candidates']})
  exp[name]=dict(c)
 print('EXPERT',exp,flush=True)
 report={'status':'TRAIN_CALIBRATED_RESEARCH_NOT_PROMOTED','teacher_step':ckpt['step'],'source':'500-book TRAIN-derived ngram counts, signed S+/S- geometry','train_calibration_books':len(calibration),'selected':{'beta':beta,'augment':aug,'floor':floor},'validation_best':selected[7],'dev_frozen':dev_report,'teacher_diagnostics':exp,'pristine_test':False,'training_selection':'strict TRAIN-book heldout only'}
 (ROOT/'results/v172_teacher_fusion_eval.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()