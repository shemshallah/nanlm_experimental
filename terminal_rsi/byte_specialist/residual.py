"""Host-owned recursive 256-byte residual specialist trained on v102 probabilities.

The donor's original source remains checksum-pinned. A deliberately tiny
instrumentation patch exposes *frozen* 256-way distributions; the standalone
Terminal-RSI host owns and checkpoints every residual weight and Adam moment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import time

import numpy as np

from . import nancm_v102 as donor
from .controller import (SOURCE_SHA256, assign_books, digest, select_windows,
                         write_windows)


def instrumented_source():
    """Add a read-only distribution tap and opt-in disk-backed context tables.

    The disk-backed path changes *only the storage allocation*, never the
    table layout, addressing, update equations, or prediction operations.
    The original v102 donor file remains checksum-pinned and unmodified.
    """
    code = donor.CSRC
    replacements = (
        ('static void eval_file(const char *wpath,',
         'static FILE *dist_out=NULL;\nstatic void eval_file(const char *wpath,'),
        ('#include <stdint.h>\n',
         '#include <stdint.h>\n#include <sys/mman.h>\n#include <unistd.h>\n'),
        ('static Slot *T[NCTX]; static Slot o0t[256];',
         '''static Slot *T[NCTX]; static Slot o0t[256];
static const char *ctxdir=NULL;
static FILE *ctxfile=NULL;
static Slot *alloc_ctx(size_t bytes){
  if(!ctxdir) return (Slot*)calloc(1,bytes);
  if(!ctxfile){
    char name[4096]; int n=snprintf(name,sizeof(name),"%s/.terminal_rsi_ctx_XXXXXX",ctxdir);
    if(n<0 || n>=(int)sizeof(name)){fprintf(stderr,"ctxdir too long\\n");return NULL;}
    int fd=mkstemp(name); if(fd<0){perror("mkstemp");return NULL;}
    unlink(name); ctxfile=fdopen(fd,"w+b");
    if(!ctxfile){perror("fdopen");close(fd);return NULL;}
  }
  int fd=fileno(ctxfile); off_t pos=lseek(fd,0,SEEK_END);
  if(pos<0 || ftruncate(fd,pos+(off_t)bytes)){perror("ftruncate"); return NULL;}
  void *p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,pos);
  if(p==MAP_FAILED){perror("mmap"); return NULL;}
  return (Slot*)p;
}'''),
        ('T[i]=(Slot*)calloc((size_t)1<<b,sizeof(Slot));',
         'T[i]=alloc_ctx(((size_t)1<<b)*sizeof(Slot));'),
        ('if(i>=skip){ dist256(pr256); float pt=pr256[c];',
         'if(i>=skip){ dist256(pr256); if(dist_out && !online) fwrite(pr256,sizeof(float),256,dist_out); float pt=pr256[c];'),
        ('const char *evalp=NULL,*bpep=NULL,*outp="res";',
         'const char *dist_path=NULL; const char *evalp=NULL,*bpep=NULL,*outp="res";'),
        ('else if(!strcmp(argv[i],"-noonline")) online_final=0;',
         'else if(!strcmp(argv[i],"-noonline")) online_final=0;\n    else if(!strcmp(argv[i],"-distout")) dist_path=argv[++i];\n    else if(!strcmp(argv[i],"-diskctx")) ctxdir=argv[++i];'),
        ('if(evalp){ char o[256]; snprintf(o,256,"%s_frozen.bin",outp);',
         'if(evalp){ if(dist_path){ dist_out=fopen(dist_path,"wb"); if(!dist_out) return 1; } char o[256]; snprintf(o,256,"%s_frozen.bin",outp);'),
        ('print_units("frozen");\n    if(online_final)',
         'print_units("frozen"); if(dist_out){fclose(dist_out);dist_out=NULL;}\n    if(online_final)'),
    )
    for before, after in replacements:
        if code.count(before) != 1:
            raise RuntimeError('Donor instrumentation signature not unique')
        code = code.replace(before, after)
    return code


def build_executable(work: Path):
    if digest(Path(donor.__file__).read_bytes()) != SOURCE_SHA256:
        raise ValueError('Donor source provenance mismatch')
    source = instrumented_source()
    c = work / 'v102_distribution_tap.c'
    exe = work / 'v102_distribution_tap'
    if not exe.exists() or not c.exists() or c.read_text() != source:
        c.write_text(source)
        subprocess.run(['gcc', '-O3', '-o', str(exe), str(c), '-lm'],
                       check=True, capture_output=True)
    return exe


def prepare_unseen(*, corpus: Path, work: Path, calibration=48, dev2=24, test2=24):
    """Train calibration uses books wholly beyond v102's 1M training prefix.

    Second dev/test samples exclude the *actual first-round* sampled books,
    not merely their sampled positions. The prior round's TEST is never read
    to choose or optimize this new policy.
    """
    manifest = json.loads((work / 'split_manifest.json').read_text())
    parts = assign_books(corpus)
    used_train = set()
    length = 0
    for name, data in sorted(parts['train'], key=lambda x: digest(x[0].encode())):
        if length >= manifest['max_train_bytes']:
            break
        used_train.add(name)
        length += len(data) + 2
    bpe = donor.BPE.load(work / 'bpe.txt')
    old_dev = {n for n, _ in select_windows(parts['dev'], bpe, count=48,
                                            width=manifest['width'], skip=manifest['skip'], seed=1001)}
    old_test = {n for n, _ in select_windows(parts['test'], bpe, count=48,
                                             width=manifest['width'], skip=manifest['skip'], seed=2002)}
    eligible = {
        'calibration': [(n,b) for n,b in parts['train'] if n not in used_train],
        'dev2': [(n,b) for n,b in parts['dev'] if n not in old_dev],
        'test2': [(n,b) for n,b in parts['test'] if n not in old_test],
    }
    requested = {'calibration':calibration, 'dev2':dev2, 'test2':test2}
    result = {'schema':'terminal-rsi-v102-residual-splits-v1',
              'parent_manifest_sha256':digest((work / 'split_manifest.json').read_bytes()),
              'used_train_book_count':len(used_train), 'prior_dev_book_count':len(old_dev),
              'prior_test_book_count':len(old_test), 'splits':{}}
    for label, count in requested.items():
        selection = select_windows(eligible[label], bpe, count=count,
                                   width=manifest['width'], skip=manifest['skip'],
                                   seed={'calibration':3003,'dev2':4004,'test2':5005}[label])
        result['splits'][label] = write_windows(work, label, selection, bpe,
                                                 skip=manifest['skip'])
    (work/'residual_manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def run_tap(work: Path, split: str, *, diskctx: bool = False):
    """Train the same original v102 model and dump strictly frozen distributions."""
    if split not in ('calibration','dev2','test2','dev3','test3','dev4'):
        raise ValueError('Non-whitelisted split')
    original = json.loads((work/'split_manifest.json').read_text())
    for suffix in ('bin','u'):
        shutil.copyfile(work/f'win_{split}.{suffix}',work/f'win.{suffix}')
    exe = build_executable(work)
    out = work/f'residual_{split}'
    cmd = [str(exe),'-bpe',str(work/'bpe.txt'),'-eval',str(work/'win.bin'),
           '-evalu',str(work/'win.u'),'-tshift','0','-o',str(out),
           '-maxb',str(original['max_train_bytes'])]
    cmd += donor.BASE + donor.lmflags(argparse.Namespace(work=str(work),beam=8), True,True,gc=0)
    cmd += ['-xc','2047','-noonline','-distout',str(out)+'.prob',str(work/'train.bin')]
    if diskctx:
        cmd[1:1] = ['-diskctx',str(work)]
    start=time.monotonic()
    proc=subprocess.run(cmd,text=True,capture_output=True)
    if proc.returncode or 'FINAL frozen' not in proc.stdout:
        raise RuntimeError(f'v102 distribution run failed (exit={proc.returncode}): '
                           + (proc.stderr or proc.stdout)[-600:])
    return {'split':split, 'elapsed_seconds':round(time.monotonic()-start,2),
            'source_sha256':digest(Path(donor.__file__).read_bytes()),
            'instrumented_sha256':digest(instrumented_source().encode()),
            'context_storage':'disk-backed' if diskctx else 'anonymous'}


def load_scored(work: Path, split: str):
    """All shapes, probabilities, targets, and donor NLL verified independently."""
    manifest_name=('guard_manifest.json' if split=='dev4' else
                   'context_manifest.json' if split in ('dev3','test3') else
                   'residual_manifest.json')
    meta=json.loads((work/manifest_name).read_text())['splits'][split]
    n=meta['scored_bytes']; per=n//meta['books']
    probabilities=np.fromfile(work/f'residual_{split}.prob', dtype='<f4')
    if probabilities.size != n*256:
        raise ValueError('Incorrect distribution file length')
    probabilities=probabilities.reshape(n,256).astype(np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid probabilities')
    totals=probabilities.sum(axis=1)
    if np.max(np.abs(totals-1.0)) > 3e-3:
        raise ValueError('Distribution is not normalized')
    payload=(work/f'win_{split}.bin').read_bytes()
    count,skip=struct.unpack_from('<II',payload)
    targets=[];pos=8
    for _ in range(count):
        width=struct.unpack_from('<I',payload,pos)[0]; pos+=4
        targets.extend(payload[pos+skip:pos+width]);pos+=width
    if pos!=len(payload) or len(targets)!=n:
        raise ValueError('Mismatch between windows and predictions')
    targets=np.asarray(targets,dtype=np.int64)
    rec=np.fromfile(work/f'residual_{split}_frozen.bin',
                    dtype=np.dtype([('loss','<f4'),('top1','u1'),('top5','u1'),('pad','u1',(2,))]))
    if len(rec)!=n or (rec['pad']!=0).any():
        raise ValueError('Missing/malformed donor records')
    reconstructed=-np.log(np.maximum(probabilities[np.arange(n),targets],1e-9))
    if np.max(np.abs(reconstructed-rec['loss'])) > .00001:
        raise ValueError('Donor probability/record mismatch')
    return probabilities,targets,per


class ResidualPolicy:
    """One unified 256-logit correction and persistent Adam/critic history."""
    def __init__(self):
        self.weights=np.zeros(256,np.float64)
        self.m=np.zeros(256,np.float64)
        self.v=np.zeros(256,np.float64)
        self.t=0
        self.critic=np.zeros(2,np.float64)  # space, non-space calibration errors

    def score(self, probs, targets, per):
        logits=np.log(np.maximum(probs,1e-20))+self.weights
        logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
        chosen=p[np.arange(len(targets)),targets]
        loss=-np.log(np.maximum(chosen,1e-30))
        top5=np.argpartition(p,-5,axis=1)[:,-5:]
        return {'bpc':float(loss.mean()/math.log(2)),
                'top1':float(np.mean(p.argmax(axis=1)==targets)),
                'top5':float(np.mean(np.any(top5==targets[:,None],axis=1))),
                'per_book_bpc':(loss.reshape(-1,per).mean(axis=1)/math.log(2)).tolist()}

    def train(self, probs, targets, *, steps, lr, phase, seed=2026):
        rng=np.random.default_rng(seed+self.t)
        report=[]
        for _ in range(steps):
            ix=rng.integers(0,len(targets),size=min(512,len(targets)))
            q=probs[ix];y=targets[ix];
            x=np.log(np.maximum(q,1e-20))+self.weights
            x-=x.max(axis=1,keepdims=True)
            z=np.exp(x);z/=z.sum(axis=1,keepdims=True)
            losses=-np.log(np.maximum(z[np.arange(len(ix)),y],1e-30))
            if phase=='RPR':
                group=(y!=32).astype(int)
                # Critic reweights the consistently harder family, capped to
                # avoid uncontrolled reward scaling / gaming the dev metric.
                factors=np.clip(1+0.15*(self.critic[group]-self.critic.mean()),.9,1.1)
            else:
                group=(y!=32).astype(int)
                factors=np.ones(len(ix))
            grad=z.copy();grad[np.arange(len(ix)),y]-=1
            g=(grad*factors[:,None]).mean(axis=0)+.002*self.weights
            self.t+=1
            self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g
            self.weights-=lr*(self.m/(1-.9**self.t))/(np.sqrt(self.v/(1-.999**self.t))+1e-8)
            for j in range(2):
                if np.any(group==j):
                    self.critic[j]=.95*self.critic[j]+.05*float(losses[group==j].mean())
            if self.t%32==0:report.append({'adam_step':self.t,'minibatch_nll':float(losses.mean())})
        return report

    def state(self):
        return {'weights':self.weights.tolist(),'adam_m':self.m.tolist(),
                'adam_v':self.v.tolist(),'adam_step':self.t,'critic':self.critic.tolist()}

    def save_checkpoint(self, destination: Path):
        np.savez_compressed(destination, weights=self.weights, m=self.m,
                            v=self.v,critic=self.critic,
                            adam_step=np.array(self.t,dtype=np.int64))

    @classmethod
    def load_checkpoint(cls, source: Path):
        model=cls()
        with np.load(source,allow_pickle=False) as checkpoint:
            if set(checkpoint.files)!={'weights','m','v','critic','adam_step'}:
                raise ValueError('Incomplete residual/optimizer checkpoint')
            for key in ('weights','m','v','critic'):
                expected=(2,) if key=='critic' else (256,)
                array=checkpoint[key]
                if array.shape!=expected or not np.isfinite(array).all():
                    raise ValueError('Invalid saved residual parameter')
                setattr(model,key,array.astype(np.float64))
            step=checkpoint['adam_step']
            if step.shape!=() or int(step)<0:
                raise ValueError('Invalid saved optimizer step')
            model.t=int(step)
        return model


def recursive_residual(work: Path, *, steps=96, lr=.01):
    """Verified RSR/RPR continuation; DEV2 decides, TEST2 read one time."""
    if steps < 2 or lr<=0:
        raise ValueError('Bad optimization budget')
    probs,targets,per=load_scored(work,'calibration')
    dev_p,dev_y,dev_per=load_scored(work,'dev2')
    policy=ResidualPolicy()
    calibration_base=policy.score(probs,targets,per)
    baseline=policy.score(dev_p,dev_y,dev_per)
    best=baseline['bpc']; best_state=policy.state(); accepted=False
    stages=[{'stage':'BASE','dev2':baseline,'adam_step':0}]
    for stage,rate in [('RSR-1',lr),('RPR-1',lr/4),('RSR-2',lr/8),('RPR-2',lr/16)]:
        receipts=policy.train(probs,targets,steps=steps,lr=rate,
                              phase='RPR' if 'RPR' in stage else 'RSR')
        score=policy.score(dev_p,dev_y,dev_per)
        better=score['bpc'] < best-.0002
        if better:
            best,accepted,best_state=score['bpc'],True,policy.state()
        stages.append({'stage':stage,'dev2':score,'adam_step':policy.t,
                       'candidate_better_than_best':better,'training_receipts':receipts})
    # Restore the BEST accepted model exactly, without throwing away moments
    # or mixing alternative independently trained models.
    chosen=ResidualPolicy()
    for key,arr in [('weights','weights'),('m','adam_m'),('v','adam_v'),('critic','critic')]:
        setattr(chosen,key,np.asarray(best_state[arr],dtype=np.float64))
    chosen.t=best_state['adam_step']
    calibration_selected=chosen.score(probs,targets,per)
    # Matched-budget negative control: randomize only calibration targets,
    # leave each genuine donor distribution and the DEV2 labels untouched.
    fake_targets=np.random.default_rng(9921).permutation(targets)
    null=ResidualPolicy()
    null.train(probs,fake_targets,steps=steps,lr=lr,phase='RSR')
    null.train(probs,fake_targets,steps=steps,lr=lr/4,phase='RPR')
    null.train(probs,fake_targets,steps=steps,lr=lr/8,phase='RSR')
    null.train(probs,fake_targets,steps=steps,lr=lr/16,phase='RPR')
    null_dev=null.score(dev_p,dev_y,dev_per)
    test_p,test_y,test_per=load_scored(work,'test2')
    test_baseline=ResidualPolicy().score(test_p,test_y,test_per)
    test_selected=chosen.score(test_p,test_y,test_per)
    test_shuffled=null.score(test_p,test_y,test_per)
    d=np.asarray(test_baseline['per_book_bpc'])-np.asarray(test_selected['per_book_bpc'])
    signal=np.asarray(test_shuffled['per_book_bpc'])-np.asarray(test_selected['per_book_bpc'])
    rng=random.Random(701)
    boot=sorted(float(np.mean(rng.choices(d.tolist(),k=len(d)))) for _ in range(2000))
    signal_ci=sorted(float(np.mean(rng.choices(signal.tolist(),k=len(signal)))) for _ in range(2000))
    meta=json.loads((work/'residual_manifest.json').read_text())
    result={'schema':'terminal-rsi-v102-host-owned-residual-v1',
            'source':'v102 frozen 256-byte output, host-residual Adam with persistent state',
            'corpus_manifest_sha256':digest((work/'residual_manifest.json').read_bytes()),
            'book_split':meta,'stages':stages,
            'selected_step':chosen.t,'candidate_dev_accepted':accepted,
            'calibration_baseline':calibration_base,
            'calibration_selected':calibration_selected,
            'shuffled_target_control_dev2':null_dev,
            'shuffled_target_control_updates':null.t,
            'shuffled_target_control_test2':test_shuffled,
            'sealed_test_baseline':test_baseline,'sealed_test_candidate':test_selected,
            'test_gain_ci95':[boot[50],boot[1949]],
            'correct_vs_shuffled_label_gain_ci95':[signal_ci[50],signal_ci[1949]],
            'test_better_books':int(np.sum(d>0)),
            'candidate_promoted':False,
            'promotion_gate':'REJECT until aligned top-5, held-out and terminal-task gates pass',
            'checkpoint_note':'Residual-only weights and Adam history, donor C model retrained, no FA2',
            'caveat':'The host correction is learned and retained; no persistent donor C weights yet'}
    out=work/'residual_search.json'
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    chosen.save_checkpoint(work/'residual_checkpoint.npz')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','tap','train'))
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--split',choices=('calibration','dev2','test2','dev3','test3','dev4'))
    parser.add_argument('--diskctx',action='store_true',
                        help='Back context tables with an unlinked temporary disk file')
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.corpus:parser.error('Missing --corpus')
        report=prepare_unseen(corpus=args.corpus,work=args.work)
    elif args.command=='tap':
        if not args.split:parser.error('Missing --split')
        report=run_tap(args.work,args.split,diskctx=args.diskctx)
    else:
        report=recursive_residual(args.work)
    if args.command=='train':
        print(json.dumps({'selected_step':report['selected_step'],
                          'dev_accepted':report['candidate_dev_accepted'],
                          'dev_stages':[(x['stage'],x['dev2']['bpc']) for x in report['stages']],
                          'shuffled_labels_dev2_bpc':report['shuffled_target_control_dev2']['bpc'],
                          'shuffled_labels_test2_bpc':report['shuffled_target_control_test2']['bpc'],
                          'vs_shuffled_labels_ci95':report['correct_vs_shuffled_label_gain_ci95'],
                          'test_baseline':{k:v for k,v in report['sealed_test_baseline'].items() if k!='per_book_bpc'},
                          'test_candidate':{k:v for k,v in report['sealed_test_candidate'].items() if k!='per_book_bpc'},
                          'test_gain_ci95':report['test_gain_ci95']},indent=2))
    else:print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
