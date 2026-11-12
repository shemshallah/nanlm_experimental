#!/usr/bin/env python3
"""Finish matched CPU experiment after complete frozen control in previous log.

No weights/optimizer from control are used or saved; start original checksum,
RNG, train window pool and evaluation are asserted before training R12 active arm.
"""
import argparse,hashlib,json,os,pathlib,time,gc
import numpy as np
import torch
import r14_slim_base as r12
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
from r11_raw import windows as base_windows
r12.run_window=run_window_fast
r12.read_books=read_books_cached

def parse(log):
    rows=[]
    for line in pathlib.Path(log).read_text().splitlines():
        if line.startswith('{'):
            try:rows.append(json.loads(line))
            except:pass
    return {p['phase']:p for p in rows if p.get('phase')!='ARM'}|{'ARM':[p for p in rows if p.get('phase')=='ARM']}

def main(a):
    torch.set_num_threads(a.threads)
    root=pathlib.Path(a.root);ckpt=pathlib.Path(a.checkpoint)
    prior=parse(a.control_log)
    actual_sha=r12.sha(ckpt)
    assert actual_sha==prior['SOURCE']['sha256'],('wrong parent checkpoint',actual_sha,prior['SOURCE']['sha256'])
    control=next((x for x in prior['ARM'] if x['arm']=='frozen_residual_control'),None)
    assert control is not None,'prior frozen control did not finish'
    saved,r6,comp,model,bridge,adapter,opt,rg=r12.load_for_arm(root,ckpt,True,a.seed,a.capacity,a.topk)
    docs,manifest=read_books_cached(a.books,comp,r6.vocab,a.max_books)
    pools=base_windows(docs,a.span,max(a.eval,a.train_pool),a.max_tokens)
    for key in ('dev','test'):pools[key]=pools[key][:a.eval]
    pool_sha=hashlib.sha256(json.dumps(pools['train']).encode()).hexdigest()
    assert pool_sha==prior['BROADENED_SPLIT']['training_pool_sha256'],'mismatched train pool'
    assert len(pools['dev'])==prior['BROADENED_SPLIT']['development_windows'],'dev sample changed'
    baseline=r12.evaluate(model,bridge,adapter,docs['dev'],pools['dev'],a.span,a.eval_batch)
    for k in ('bits_per_byte','other_nll','other_top1','space_top1'):
        assert abs(baseline[k]-prior['INITIAL_IDENTITY']['baseline'][k])<1e-7,(k,baseline[k],prior['INITIAL_IDENTITY']['baseline'][k])
    old_before,future_before=r12.old_anchors(model,root,a.word_eval,a.future_eval)
    print(json.dumps({'phase':'RESUME_CONTROL','parent_sha256':actual_sha,'control_dev':control['dev'],
                      'baseline_dev':baseline,'train_pool':len(pools['train'])}),flush=True)
    del model,bridge,adapter,opt,rg,r6,comp,saved
    gc.collect()
    ev,extra,state=r12.train_arm(root,ckpt,docs,pools,a,True)
    accepted=(ev['bits_per_byte']+a.min_gain<control['dev']['bits_per_byte'] and
              ev['other_nll']+a.min_gain<control['dev']['other_nll'] and
              ev['bits_per_byte']+a.min_gain<baseline['bits_per_byte'] and
              ev['other_nll']+a.min_gain<baseline['other_nll'] and
              extra['old_word_nll']<=old_before+a.word_margin and
              all(extra['future'][str(h)]['nll']<=future_before[str(h)]['nll']+a.future_margin for h in (1,2,3)))
    report={'parent_sha256':actual_sha,'comparison':'resume exactly matched previously completed frozen control',
            'train_window_pool':len(pools['train']),'training_books':len(docs['train']),
            'steps':a.steps,'batch':a.batch,'control':control['dev'],'baseline':baseline,
            'learned':ev,'learned_diagnostics':extra,'anchor_before':{'word':old_before,'future':future_before},
            'accepted':accepted,'promoted':False,'test_evaluated_for_selection':False}
    if a.promote and accepted:
        saved_w,mod,br,ad,optimizer,rg,trng=state
        result=dict(saved_w)
        result.update({'model':mod.state_dict(),'stream_bridge':br.state_dict(),
            'stream_optimizer':optimizer.state_dict(),'stream_rng':rg.bit_generator.state,
            'torch_rng':trng,'parent_sha256':actual_sha,
            'r12_selected_arm':'causal_recall_plus_ffn','r12_config':vars(a),
            'r12_updates':int(saved_w['r12_updates'])+a.steps,
            'r12_residual':ad.state_dict(),'format':'PAL12_CAUSAL_RESIDUAL_V1'})
        tmp=ckpt.with_suffix(ckpt.suffix+'.tmp')
        torch.save(result,tmp);os.replace(tmp,ckpt)
        report['promoted']=True;report['checkpoint_sha256']=r12.sha(ckpt)
        report['r12_updates']=result['r12_updates']
    pathlib.Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'RESULT','accepted':accepted,'promoted':report['promoted'],
                      'learned_dev':ev,'control_dev':control['dev'],
                      'new_checkpoint_sha256':report.get('checkpoint_sha256')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.');p.add_argument('--books',default='nanrouter')
    p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
    p.add_argument('--control-log',required=True)
    p.add_argument('--report',default='reports/r14_matched_resumed.json')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--span',type=int,default=24);p.add_argument('--eval',type=int,default=96)
    p.add_argument('--train-pool',type=int,default=8000)
    p.add_argument('--max-tokens',type=int,default=160);p.add_argument('--max-books',type=int,default=0)
    p.add_argument('--eval-batch',type=int,default=2);p.add_argument('--capacity',type=int,default=64)
    p.add_argument('--topk',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--word-eval',type=int,default=64);p.add_argument('--future-eval',type=int,default=48)
    p.add_argument('--word-margin',type=float,default=.0008)
    p.add_argument('--future-margin',type=float,default=.00045)
    p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--promote',action='store_true');p.add_argument('--skip-ablations',action='store_true',default=True)
    main(p.parse_args())
