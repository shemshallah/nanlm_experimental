"""Multi-step Terminal-RSI episodes using the existing persistent policy.

Each step supplies candidate argv arrays and an independent output check.
After an unsuccessful execution the RSI loop observes the error, updates its
critic/actor, and selects an untried repair candidate. A successful step
unlocks the next; a single combined checkpoint stores both policy and cursor.

The boundary for a later low-BPC proposal model is `CommandProposer`: it may
create ranked candidate command arrays, while this planner owns execution,
outcome verification, state and learning. Such a model is NOT integrated yet.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from .core import Action, tb_gae
from .terminal_agent import CommandTask, FeedbackPolicy, _sha, attempt


class CommandProposer(Protocol):
    """Future low-BPC adapter: proposal only; never grades its own output."""
    def propose(self, goal: str, prior_results: tuple[dict,...]) -> tuple[tuple[str,...],...]: ...


@dataclass(frozen=True)
class PlanStep:
    command: CommandTask
    max_attempts: int
    max_generated: int


@dataclass(frozen=True)
class TerminalPlan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep,...]

    @classmethod
    def parse(cls,data):
        if (not isinstance(data,dict) or set(data)!={'plan_id','goal','steps'} or
                not isinstance(data['plan_id'],str) or not 0<len(data['plan_id'])<=128 or
                not isinstance(data['goal'],str) or not 0<len(data['goal'])<=2048 or
                not isinstance(data['steps'],list) or not 1<=len(data['steps'])<=32):
            raise ValueError('Invalid Terminal-RSI plan')
        steps=[]
        for index,source in enumerate(data['steps']):
            if not isinstance(source,dict):raise ValueError('Malformed plan step')
            options={k:v for k,v in source.items()
                     if k not in ('max_attempts','max_generated')}
            options['task_id']=f"{data['plan_id']}/step-{index:02d}"
            command=CommandTask.parse(options)
            tries=source.get('max_attempts',len(command.choices))
            generated=source.get('max_generated',0)
            if type(tries) is not int or not 1<=tries<=len(command.choices):
                raise ValueError('Invalid maximum repair attempts')
            if type(generated) is not int or not 0<=generated<=8:
                raise ValueError('Invalid generated command budget')
            steps.append(PlanStep(command,tries,generated))
        return cls(data['plan_id'],data['goal'],tuple(steps))

    def digest(self):
        return _sha({'plan_id':self.plan_id,'goal':self.goal,
                     'steps':[{'task_id':s.command.task_id,
                               'family':s.command.family,
                               'choices':s.command.choices,
                               'expected_stdout':s.command.expected_stdout,
                               'expected_code':s.command.expected_code,
                               'max_attempts':s.max_attempts,
                               'max_generated':s.max_generated} for s in self.steps]})


def run_plan(plan:TerminalPlan, policy:FeedbackPolicy, root:Path,
             checkpoint:Path, *, train=False,timeout=5.,max_commands=None,
             proposer:CommandProposer|None=None):
    """Execute/resume with a checkpoint after every observed command.

    `max_commands` lets a caller deliberately pause a plan after a known
    observation boundary, which is useful for deterministic resume testing.
    """
    root=Path(root).resolve(strict=True)
    if not root.is_dir():raise ValueError('Terminal plan root must be a directory')
    if max_commands is not None and (type(max_commands) is not int or max_commands<1):
        raise ValueError('Bad command budget')
    digest=plan.digest()
    state=policy.episode
    if (state is not None and state.get('status') in ('succeeded','failed') and
            (state.get('plan_digest')!=digest or state.get('root')!=str(root))):
        # One actor/optimizer history continues across independently verified
        # episodes. Only an already completed episode may be archived.
        policy.episode_history.append({
            'plan_id':state['plan_id'],'plan_digest':state['plan_digest'],
            'status':state['status'],'attempts':len(state['records']),
            'last_receipt_sha256':policy.receipt_digest})
        policy.episode=None
        state=None
    resumed=state is not None
    if state is None:
        state={'schema':'terminal-rsi-plan-episode-v1','plan_id':plan.plan_id,
               'plan_digest':digest,'root':str(root),'next_step':0,
               'tried':[],'generated':[],'generated_tried':[],
               'records':[],'status':'in_progress'}
        policy.episode=state
        policy.save(checkpoint)
    if (not isinstance(state,dict) or
            state.get('schema')!='terminal-rsi-plan-episode-v1' or
            state.get('plan_id')!=plan.plan_id or
            state.get('plan_digest')!=digest or state.get('root')!=str(root)):
        raise ValueError('Saved RSI episode belongs to a different plan/root')
    if (type(state.get('next_step')) is not int or
            not 0<=state['next_step']<=len(plan.steps) or
            not isinstance(state.get('records'),list) or
            state.get('status') not in ('in_progress','succeeded','failed')):
        raise ValueError('Invalid saved RSI episode cursor')
    if state['status']=='in_progress':
        performed=0
        while state['next_step']<len(plan.steps):
            index=state['next_step']
            step=plan.steps[index]
            tried=state['tried']
            if (not isinstance(tried,list) or len(tried)>step.max_attempts or
                    len(set(tried))!=len(tried) or
                    any(type(t) is not int or t<0 or t>=len(step.command.choices)
                        for t in tried)):
                raise ValueError('Invalid saved repair history')
            if len(tried)<step.max_attempts:
                candidate=step.command
                excluded=tried
                source='preplanned'
            else:
                if not step.max_generated or proposer is None:
                    state['status']='failed';policy.save(checkpoint)
                    break
                if not state['generated']:
                    proposed=proposer.propose(plan.goal,tuple(state['records']))
                    if (not isinstance(proposed,(list,tuple)) or
                            len(proposed)>step.max_generated):
                        raise ValueError('Proposer exceeded its command budget')
                    if proposed:
                        proposal_family=step.command.family[:110]+'-proposal'
                        checked=CommandTask.parse({
                            'task_id':step.command.task_id,
                            'family':proposal_family,
                            'choices':[list(cmd) for cmd in proposed],
                            'expected_stdout':step.command.expected_stdout,
                            'expected_code':step.command.expected_code})
                        state['generated']=[list(cmd) for cmd in checked.choices]
                        state['generated_tried']=[]
                        policy.save(checkpoint)  # proposed actions survive restart
                if not state['generated']:
                    state['status']='failed';policy.save(checkpoint)
                    break
                excluded=state['generated_tried']
                if (not isinstance(excluded,list) or
                        len(set(excluded))!=len(excluded) or
                        any(type(t) is not int or t<0 or t>=len(state['generated'])
                            for t in excluded)):
                    raise ValueError('Invalid generated repair history')
                if len(excluded)==len(state['generated']):
                    state['status']='failed';policy.save(checkpoint)
                    break
                candidate=CommandTask(
                    step.command.task_id,step.command.family[:110]+'-proposal',
                    tuple(tuple(x) for x in state['generated']),
                    step.command.expected_stdout,step.command.expected_code)
                source='proposed'
            result=attempt(candidate,policy,root,train=train,
                           timeout=timeout,excluded=tuple(excluded))
            result['step_index']=index
            result['source']=source
            state['records'].append(result)
            excluded.append(result['choice'])
            if result['verified_success']:
                state['next_step']+=1
                state['tried']=[]
                state['generated']=[]
                state['generated_tried']=[]
                if state['next_step']==len(plan.steps):state['status']='succeeded'
            elif source=='proposed' and len(excluded)==len(state['generated']):
                state['status']='failed'
            elif source=='preplanned' and len(tried)>=step.max_attempts and (
                    not step.max_generated or proposer is None):
                state['status']='failed'
            policy.save(checkpoint)
            performed+=1
            if state['status']!='in_progress' or (
                    max_commands is not None and performed>=max_commands):
                break
    records=state['records']
    # This is a retrospective credit diagnostic; online policy/Adam updates
    # have already occurred strictly after each verified command outcome.
    if records and state['status']!='in_progress':
        actions=tuple(Action(r['step_index'],r['reward']-r['advantage'],
                             r['reward'],i==len(records)-1)
                      for i,r in enumerate(records))
        credit=tb_gae(actions).advantages
    else:credit=()
    return {'schema':'terminal-rsi-plan-report-v1','plan_id':plan.plan_id,
            'goal':plan.goal,'status':state['status'],
            'completed_steps':state['next_step'],'total_steps':len(plan.steps),
            'attempts':len(records),'successful_attempts':sum(
                r['verified_success'] for r in records),
            'resumed':resumed,'online_learning':bool(train),
            'adam_updates':policy.total_updates,
            'final_receipt_sha256':policy.receipt_digest,
            'turn_aware_advantages':list(credit),
            'credit_pending':state['status']=='in_progress','records':records}


def main():
    p=argparse.ArgumentParser(description='Resumable, verified multi-step Terminal-RSI')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--train',action='store_true')
    p.add_argument('--timeout',type=float,default=5.)
    p.add_argument('--max-commands',type=int,
                   help='Pause after N command outcomes; resume using same checkpoint')
    args=p.parse_args()
    plan=TerminalPlan.parse(json.loads(args.plan.read_text()))
    actor=(FeedbackPolicy.load(args.checkpoint) if args.checkpoint.exists()
           else FeedbackPolicy())
    print(json.dumps(run_plan(plan,actor,args.root,args.checkpoint,
                              train=args.train,timeout=args.timeout,
                              max_commands=args.max_commands),
                     sort_keys=True,indent=2))


if __name__=='__main__':main()
