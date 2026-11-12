#!/usr/bin/env python3
"""R7.12-B frozen-PAL reliability-conditioned calibrated retrieval experiment.

A paired correct/wrong-memory contrast regularizes the gate. Selection uses
ONLY historical development; all older weights and moments stay unchanged.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pathlib,tempfile,time
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from r712_lm_bridge import CalibrationGate,collect,score,summary
from r711_delta import restore_overlay
from r710_checkpoint_integrity import sha256

class ReliabilityGate(CalibrationGate):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            torch.manual_seed(71222)
            nn.init.normal_(self.net[0].weight,std=.2)
            nn.init.normal_(self.net[2].weight,std=.04)
            self.net[2].bias.fill_(-3.75)

def comparison(gate,correct,wrong):
    c=score(gate,correct);w=score(gate,wrong)
    base=correct['baseline']
    assert torch.equal(base,wrong['baseline'])
    return {'correct_gain':float((base-c).mean().detach()),
       'wrong_gain':float((base-w).mean().detach()),
       'correct_advantage':float((w-c).mean().detach()),
       'correct_nll':float(c.mean().detach()),'n':len(base),
       'mean_gate_correct':float(gate(correct['features']).mean().detach()),
       'mean_gate_wrong':float(gate(wrong['features']).mean().detach())}

def main(a):
    t=time.monotonic();torch.set_num_threads(a.threads)
    root=a.root;before=sha256(root/'checkpoint/pal_active.pt')
    model,opts,base,comp,patch=restore_overlay(root,root/'adaptation/r711_experimental_delta.pt')
    torch.manual_seed(71222)
    gate=ReliabilityGate()
    tc=collect(root,model,base,comp,'train',a.train,7112,a.batch)
    tw=collect(root,model,base,comp,'train',a.train,7112,a.batch,wrong=True)
    dc=collect(root,model,base,comp,'dev',a.dev,7129,a.batch)
    dw=collect(root,model,base,comp,'dev',a.dev,7129,a.batch,wrong=True)
    # On strong evidence, set initially optimistic calibration; don't bias old weights.
    opts[4].add_param_group({'params':list(gate.parameters()),'lr':a.lr,'weight_decay':0.})
    opt=opts[4]
    initial=comparison(gate,dc,dw)
    print(json.dumps({'phase':'PRECOMPUTED','train':a.train,'dev':a.dev,
           'initial':initial,'seconds':round(time.monotonic()-t,2)}),flush=True)
    best=None;logs=[];rng=np.random.default_rng(71222)
    for step in range(1,a.steps+1):
        idx=rng.choice(a.train,size=min(a.minibatch,a.train),replace=False)
        c={k:v[idx] if isinstance(v,torch.Tensor) else len(idx) for k,v in tc.items()}
        w={k:v[idx] if isinstance(v,torch.Tensor) else len(idx) for k,v in tw.items()}
        lc=score(gate,c);lw=score(gate,w)
        # Penalty on wrong-source memory beating correct historical memory.
        loss=lc.mean()+a.contrast*F.softplus(8*(.01+lc-lw)).mean()/8
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(),1.)
        opt.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            result=comparison(gate,dc,dw)
            result['step']=step
            logs.append(result)
            print(json.dumps({'phase':'GATE','step':step,'dev_gain':result['correct_gain'],
              'dev_adv':result['correct_advantage'],'wrong_gain':result['wrong_gain']}),flush=True)
            # Gate only if both criteria improve the initial accepted no-gate
            # baseline; no cherry-picking on historical test.
            if result['correct_gain']>a.min_gain and result['correct_advantage']>a.min_adv:
                if best is None or result['correct_nll']<best['metrics']['correct_nll']:
                    best={'step':step,'metrics':result,
                        'state':copy.deepcopy(gate.state_dict()),
                        'optimizer':copy.deepcopy(opt.state_dict())}
    if best is not None:
        gate.load_state_dict(best['state']);opt.load_state_dict(best['optimizer'])
    tc_eval=comparison(gate,tc,tw)
    dc_eval=comparison(gate,dc,dw)
    # Historical test once, after selection. Not sealed books.
    ec=collect(root,model,base,comp,'test',a.test,7138,a.batch)
    ew=collect(root,model,base,comp,'test',a.test,7138,a.batch,wrong=True)
    er=comparison(gate,ec,ew)
    report={'candidate':'R7.12 B contrastive reliability conditioned gate','parent_sha256':before,
       'accepted_joint_steps':14416,'r711_auxiliary_updates':80,'candidate_is_promoted':False,
       'selection_criteria':{'dev_gain_gt':a.min_gain,'dev_correct_vs_wrong_gt':a.min_adv},
       'dev_initial':initial,'selected_step':best['step'] if best else 0,
       'dev_selected':dc_eval,'train_selected':tc_eval,'historical_test':er,
       'training_log':logs,'independent_book_test':False,
       'fifth_optimizer_new_group_only':True,'five_original_histories_unchanged':True,
       'accepted_hash_unchanged':sha256(root/'checkpoint/pal_active.pt')==before,
       'elapsed_s':round(time.monotonic()-t,2)}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    if best is not None:
        payload={'format':'PAL_R7_12_RELIABILITY_GATE_OVERLAY_V1','parent_sha256':before,
         'r711_delta_sha256':sha256(root/'adaptation/r711_experimental_delta.pt'),
         'parameters':gate.state_dict(),'fifth_group':opt.state_dict()['param_groups'][-1],
         'fifth_moments':{k:v for k,v in opt.state_dict()['state'].items()
             if k in opt.state_dict()['param_groups'][-1]['params']},
         'selected_step':best['step'],'metrics':report}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.r712_',suffix='.pending',dir=a.output.parent);os.close(fd)
        try:
            torch.save(payload,tmp)
            with open(tmp,'rb') as f:os.fsync(f.fileno())
            os.replace(tmp,a.output)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        a.output.with_suffix(a.output.suffix+'.sha256').write_text(sha256(a.output)+'  '+a.output.name+'\n')
        report['overlay_sha256']=sha256(a.output)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','selected_step':report['selected_step'],
      'dev':report['dev_selected'],'test':report['historical_test'],
      'source_hash_unchanged':report['accepted_hash_unchanged'],'elapsed':report['elapsed_s']}),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--train',type=int,default=1024);p.add_argument('--dev',type=int,default=384)
    p.add_argument('--test',type=int,default=512);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=128);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--eval-every',type=int,default=16);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=.006);p.add_argument('--contrast',type=float,default=1.0)
    p.add_argument('--min-gain',type=float,default=.001);p.add_argument('--min-adv',type=float,default=.001)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('adaptation/r712_reliable_delta.pt'))
    p.add_argument('--report',type=pathlib.Path,default=pathlib.Path('reports/R7p12_reliability.json'))
    main(p.parse_args())
