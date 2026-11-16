"""Low-BPC proposal ranking port for the live Terminal-RSI planner.

Each candidate is evaluated as an independent causal continuation of the
same *observed* goal/tool-output prefix. No candidate byte becomes context
until after the predictor assigns that byte a probability. These BPC scores
are command-text likelihoods, NOT probabilities of terminal task success.

The unchanged v102 C donor does not currently provide a recoverable, live
next-byte checkpoint. This module therefore accepts a donor stream factory;
it must not claim actual v102 candidate scores without such a provider.
"""
from __future__ import annotations

import json
import math
from typing import Protocol

import numpy as np

from .feedback_core import FeedbackRSI, StreamingRSICore


class NextByteStream(Protocol):
    def predict(self) -> np.ndarray: ...
    def observe(self, byte: int) -> None: ...


class DonorStreamFactory(Protocol):
    def __call__(self, observed_prefix: bytes) -> NextByteStream: ...


class CausalRSIPredictor:
    """Connect a next-byte donor stream to the unified RSI feedback core."""
    def __init__(self, model:FeedbackRSI, donor:NextByteStream,
                 observed_prefix:bytes):
        if not isinstance(observed_prefix,bytes) or not observed_prefix:
            raise ValueError('A nonempty observed prefix is required')
        self.donor=donor
        self.rsi=StreamingRSICore(model)
        self.rsi.reset_book('command-proposal-evaluation')
        for byte in observed_prefix:self.rsi.warmup(int(byte))
        self.pending=False

    def predict(self):
        if self.pending:raise ValueError('Previous command byte not observed')
        p=self.rsi.predict(self.donor.predict())
        self.pending=True
        return p

    def observe(self,byte):
        if not self.pending:raise ValueError('Command byte observation not pending')
        self.rsi.observe(byte)
        self.donor.observe(byte)
        self.pending=False


def command_bytes(argv):
    if (not isinstance(argv,(tuple,list)) or not argv or
            any(not isinstance(x,str) or not x for x in argv)):
        raise ValueError('Expected command argument array')
    return json.dumps(list(argv),ensure_ascii=False,
                      separators=(',',':')).encode('utf-8')


def observed_context(goal:str, results:tuple[dict,...], *, max_bytes=2048):
    """Include only earlier observed stdout/stderr/exit codes and the goal."""
    rows=[goal]
    for result in results:
        execution=result['execution']
        rows.append(json.dumps({'argv':result['argv'],
                                'returncode':execution['returncode'],
                                'stdout':execution['stdout'],
                                'stderr':execution['stderr']},
                               ensure_ascii=False,sort_keys=True))
    output='\n'.join(rows).encode('utf-8')[-max_bytes:]
    return output if output else b'\n'


def causal_bpc(predictor, candidate:bytes):
    if not candidate:raise ValueError('Empty candidate')
    bits=0.
    for byte in candidate:
        p=np.asarray(predictor.predict(),np.float64)
        if (p.shape!=(256,) or not np.isfinite(p).all() or
                np.any(p<0) or abs(float(p.sum())-1.)>1e-8):
            raise ValueError('Malformed command byte probability distribution')
        bits-=math.log2(max(float(p[byte]),1e-30))
        predictor.observe(int(byte))
    return bits/len(candidate)


class LowBPCRanker:
    """Reusable BPC adapter for the later alternate proposal-model experiment."""
    def __init__(self, model:FeedbackRSI, donor_factory:DonorStreamFactory):
        self.model=model
        self.donor_factory=donor_factory

    def score(self,goal:str,results:tuple[dict,...],commands):
        prefix=observed_context(goal,results)
        output=[]
        for argv in commands:
            predictor=CausalRSIPredictor(self.model,self.donor_factory(prefix),prefix)
            data=command_bytes(argv)
            output.append({'argv':tuple(argv),'bytes':len(data),
                           'bpc':causal_bpc(predictor,data)})
        return output


class BPCRankedProposer:
    """Reorder generated command suggestions by causal BPC, for ablations."""
    def __init__(self,base_proposer,ranker:LowBPCRanker):
        self.base=base_proposer
        self.ranker=ranker
        self.last_scores=[]

    def propose(self,goal:str,prior_results:tuple[dict,...]):
        suggestions=self.base.propose(goal,prior_results)
        self.last_scores=self.ranker.score(goal,prior_results,suggestions)
        return tuple(x['argv'] for x in sorted(self.last_scores,
                                               key=lambda x:x['bpc']))
