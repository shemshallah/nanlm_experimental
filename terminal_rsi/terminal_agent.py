"""Small real-command Terminal-RSI loop: choose, run, verify, learn, persist.

Commands come from an explicitly supplied task manifest; this module never
generates shell strings. Evaluation is read-only with respect to the policy.
Unlike the synthetic digit harness, commands here really execute. This is an
initial terminal-agent runtime, not an unattended production deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import tempfile
import time

import numpy as np

from .core import Action, Calibration, Outcome, spr_reward, tb_gae


@dataclass(frozen=True)
class CommandTask:
    task_id: str
    family: str
    choices: tuple[tuple[str, ...], ...]
    expected_stdout: str
    expected_code: int = 0

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - {
                'task_id', 'family', 'choices', 'expected_stdout', 'expected_code'}:
            raise ValueError('Invalid command task fields')
        task_id, family = data.get('task_id'), data.get('family')
        choices = data.get('choices')
        expected = data.get('expected_stdout')
        code = data.get('expected_code', 0)
        if (not isinstance(task_id, str) or not task_id or len(task_id)>128 or
                not isinstance(family, str) or not family or len(family)>128 or
                not isinstance(expected, str) or len(expected)>4096 or
                type(code) is not int or not -255<=code<=255 or
                not isinstance(choices, list) or not 1<=len(choices)<=16):
            raise ValueError('Malformed command task')
        if any(not isinstance(c, list) or not 1<=len(c)<=24 or
               any(not isinstance(a,str) or not a or len(a)>8192 for a in c)
               for c in choices):
            raise ValueError('Commands must be nonempty argument arrays')
        return cls(task_id, family, tuple(tuple(c) for c in choices), expected, code)


def _sha(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),
                                      allow_nan=False).encode()).hexdigest()


class FeedbackPolicy:
    """One persistent categorical actor, critic, and Adam history per family."""
    def __init__(self, seed=2026):
        self.seed=seed
        self.rows={}
        self.total_updates=0
        self.receipt_digest='0'*64

    def _row(self,family,n):
        row=self.rows.get(family)
        if row is None:
            row={'logits':[0.]*n,'m':[0.]*n,'v':[0.]*n,
                 'step':0,'observations':0,'critic':.5}
            self.rows[family]=row
        if len(row['logits'])!=n:
            raise ValueError('Changing command count for an existing family')
        return row

    def probabilities(self,task):
        row=self._row(task.family,len(task.choices))
        values=np.asarray(row['logits'],np.float64)
        weights=np.exp(values-values.max())
        return weights/weights.sum()

    def choose(self,task,*,train=False):
        p=self.probabilities(task)
        row=self._row(task.family,len(task.choices))
        if train and row['observations']<len(task.choices):
            # Visit every explicitly supplied candidate before exploiting.
            return row['observations'], p
        return int(np.argmax(p)),p

    def observe(self,task,choice,success,*,train=False,lr=.08):
        if type(success) is not bool or not 0<=choice<len(task.choices):
            raise ValueError('Bad observed command outcome')
        row=self._row(task.family,len(task.choices))
        previous=float(row['critic'])
        reward=spr_reward(Outcome(task.family, success),Calibration({}))
        advantage=tb_gae((Action(0, previous, reward, True),)).advantages[0]
        # The critic learns from every observed execution, including frozen
        # evaluation. Only explicit training changes actor/Adam weights.
        row['critic']=.9*previous+.1*reward
        row['observations']+=1
        if train:
            if not 0<lr<=.2:raise ValueError('Bad optimizer rate')
            p=self.probabilities(task)
            gradient=p.copy();gradient[choice]-=1.
            gradient*=advantage
            m=np.asarray(row['m'])*.9+.1*gradient
            v=np.asarray(row['v'])*.999+.001*gradient*gradient
            row['step']+=1
            step=row['step']
            weights=np.asarray(row['logits'])-lr*(m/(1-.9**step))/(
                np.sqrt(v/(1-.999**step))+1e-8)
            row['logits']=weights.tolist()
            row['m']=m.tolist();row['v']=v.tolist()
            self.total_updates+=1
        return advantage

    def receipt(self,record):
        self.receipt_digest=hashlib.sha256(
            bytes.fromhex(self.receipt_digest)+
            json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def state(self):
        return {'schema':'terminal-rsi-live-actor-v1','seed':self.seed,
                'rows':self.rows,'total_updates':self.total_updates,
                'receipt_digest':self.receipt_digest}

    @classmethod
    def from_state(cls,state):
        if not isinstance(state,dict) or state.get('schema')!='terminal-rsi-live-actor-v1':
            raise ValueError('Wrong terminal-agent checkpoint')
        obj=cls(state['seed'])
        obj.total_updates=state['total_updates']
        obj.receipt_digest=state['receipt_digest']
        if (type(obj.total_updates) is not int or obj.total_updates<0 or
                not isinstance(obj.receipt_digest,str) or
                len(obj.receipt_digest)!=64):
            raise ValueError('Malformed terminal agent history')
        for family,row in state['rows'].items():
            n=len(row['logits'])
            if not isinstance(family,str) or not family or not 1<=n<=16:
                raise ValueError('Malformed terminal policy family')
            if (any(len(row[key])!=n or not np.isfinite(row[key]).all()
                    for key in ('logits','m','v')) or
                    type(row['step']) is not int or row['step']<0 or
                    type(row['observations']) is not int or row['observations']<0 or
                    not 0<=float(row['critic'])<=1):
                raise ValueError('Malformed terminal policy parameters')
        obj.rows=state['rows']
        return obj

    def save(self,path:Path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        state=self.state();payload={'state':state,'sha256':_sha(state)}
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',
                                         dir=path.parent,delete=False) as tmp:
            tmp.write(json.dumps(payload,sort_keys=True,indent=2)+'\n')
            tmp.flush();os.fsync(tmp.fileno())
            temporary=Path(tmp.name)
        os.replace(temporary,path)

    @classmethod
    def load(cls,path:Path):
        record=json.loads(Path(path).read_text())
        if set(record)!={'sha256','state'} or _sha(record['state'])!=record['sha256']:
            raise ValueError('Terminal checkpoint integrity failure')
        return cls.from_state(record['state'])


def execute(root:Path, argv,timeout=5.,output_limit=32768):
    """Execute argv without a shell; bound runtime, returned output, and cwd."""
    root=Path(root).resolve(strict=True)
    if not root.is_dir() or not 0<timeout<=60 or not 256<=output_limit<=262144:
        raise ValueError('Invalid execution bounds')
    begin=time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process=subprocess.Popen(list(argv),cwd=root,stdin=subprocess.DEVNULL,
                                     stdout=stdout,stderr=stderr,start_new_session=True)
        except OSError as exc:
            return {'returncode':None,'timed_out':False,
                    'stdout':'','stderr':str(exc)[:output_limit],
                    'elapsed_seconds':round(time.monotonic()-begin,4)}
        timed_out=False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        stdout.seek(0);stderr.seek(0)
        return {'returncode':process.returncode,'timed_out':timed_out,
                'stdout':stdout.read(output_limit).decode('utf-8','replace'),
                'stderr':stderr.read(output_limit).decode('utf-8','replace'),
                'elapsed_seconds':round(time.monotonic()-begin,4)}


def attempt(task:CommandTask,policy:FeedbackPolicy,root:Path,*,train=False,
            timeout=5.,lr=.08):
    """Predict -> execute -> independently verify -> critic -> actor update."""
    choice,probs=policy.choose(task,train=train)
    execution=execute(root,task.choices[choice],timeout=timeout)
    success=(not execution['timed_out'] and
             execution['returncode']==task.expected_code and
             execution['stdout'].rstrip('\r\n')==task.expected_stdout)
    advantage=policy.observe(task,choice,success,train=train,lr=lr)
    record={'task_id':task.task_id,'family':task.family,'choice':choice,
            'choice_probability':float(probs[choice]),
            'argv':list(task.choices[choice]),'verified_success':success,
            'reward':float(success),'advantage':advantage,
            'execution':execution,'policy_updated':bool(train)}
    policy.receipt(record)
    record['receipt_digest']=policy.receipt_digest
    return record


def main():
    p=argparse.ArgumentParser(description='Live Terminal-RSI task feedback loop')
    p.add_argument('--tasks',required=True,type=Path,
                   help='JSON list of explicitly chosen command tasks')
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--train',action='store_true',
                   help='Update policy/Adam after verified outcomes')
    p.add_argument('--timeout',type=float,default=5.)
    args=p.parse_args()
    tasks=[CommandTask.parse(item) for item in json.loads(args.tasks.read_text())]
    if not tasks:raise ValueError('At least one terminal task required')
    policy=(FeedbackPolicy.load(args.checkpoint) if args.checkpoint.exists()
            else FeedbackPolicy())
    results=[]
    for task in tasks:
        outcome=attempt(task,policy,args.root,train=args.train,timeout=args.timeout)
        results.append(outcome)
        policy.save(args.checkpoint)  # checkpoints after every observed action
    print(json.dumps({'schema':'terminal-rsi-live-run-v1',
                      'tasks':len(results),'verified_successes':sum(
                          x['verified_success'] for x in results),
                      'policy_updates':policy.total_updates,
                      'final_receipt_sha256':policy.receipt_digest,
                      'results':results},indent=2,sort_keys=True))


if __name__=='__main__':main()
