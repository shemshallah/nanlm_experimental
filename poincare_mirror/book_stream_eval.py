"""R63: true sequential book-aligned FA2 evaluation with causal R62 history.

Every prediction is scored ONCE. The same frozen inherited logits and
geometric addresses are used for neutral, exact and three isolated controls.
No optimizer changes, no donor book and no target-label lookup in memory.
"""
import math
from time import perf_counter

import torch
from torch.nn import functional as F

from book_successor_stream import CausalBookSuccessor,CausalLagBookSuccessor
from rank_safe_fusion import protect_parent_space_rank


ARMS=("neutral","exact","wrong_address","wrong_byte","wrong_successor")
METRICS=("ce_1","ce_2","ce_3","acc_1","acc_2","acc_3",
         "top5_1","top5_2","top5_3",
         "ce_nonspace_1","acc_nonspace_1","top5_nonspace_1",
         "ce_space_1","acc_space_1","top5_space_1")


def _append_stats(sums,counts,predictions,targets,horizon):
    if targets.numel()==0:
        return
    ce=F.cross_entropy(predictions,targets,reduction="none")
    top1=predictions.argmax(-1).eq(targets)
    top5=predictions.topk(5,-1).indices.eq(targets[:,None]).any(-1)
    for idx,values in ((horizon,ce),(3+horizon,top1),(6+horizon,top5)):
        sums[idx]+=values.sum().item()
        counts[idx]+=values.numel()
    if horizon==0:
        for start,mask in ((9,targets.ne(32)),(12,targets.eq(32))):
            if mask.any():
                for offset,values in enumerate((ce,top1,top5)):
                    sums[start+offset]+=values[mask].sum().item()
                    counts[start+offset]+=mask.sum().item()


@torch.no_grad()
def evaluate_causal_book(model,book_tokens,book_id,context=64,chunk=16,
                         history=32768,key_mode="hierarchical",
                         score_start=0,score_positions=None,
                         space_rank_guard=False,future_winner_guard=False,
                         future_horizon_bridge=0.0,counterfactual_readout=False,
                         recency_readout=False,recency_half_life=256.0):
    """Matched causal result per arm on one independent book's input prefix.

    Args:
      book_tokens: 1-D integer sequence including at least one future
      prediction target (EOS as 256 is permitted).
      book_id: caller's verified unique book identity.
    """
    if not isinstance(book_id,(str,int)) or not str(book_id):
        raise ValueError("Unique book identity is required")
    if (book_tokens.ndim!=1 or book_tokens.dtype!=torch.long or
            book_tokens.device.type!="cpu" or book_tokens.numel()<4):
        raise ValueError("Book must contain >=4 CPU long tokens")
    if torch.any((book_tokens<0)|(book_tokens>=257)):
        raise ValueError("Book contains out-of-vocabulary tokens")
    if (context<4 or chunk<1 or chunk>context or history<1 or
            context>model.cfg.context):
        raise ValueError("Invalid context/chunk/history, or context exceeds checkpoint")
    if model.training or model.pair_successor is None or (
            model.pair_successor.joint_weight is None):
        raise ValueError("Use eval() with inherited R59/R60 joint-memory model")
    if model.embed.weight.device.type!="cpu":
        raise ValueError("R63 causal book streaming currently requires CPU model")
    if key_mode not in ("pair_byte","pair_bigram","head_byte","hierarchical"):
        raise ValueError("Unsupported causal book key mode")
    total=len(book_tokens)-1
    if (not isinstance(score_start,int) or score_start<0 or score_start>=total or
            (score_positions is not None and
             (not isinstance(score_positions,int) or score_positions<1))):
        raise ValueError("Invalid causal scoring interval")
    score_end=(total if score_positions is None else
               min(total,score_start+score_positions))

    # All five histories start empty and never receive another book's data.
    wrong_pair=torch.arange(32640,dtype=torch.long).roll(137)
    wrong_byte=torch.arange(257,dtype=torch.long).roll(1)
    streams={}
    future_streams={}
    active_future=bool(model.cfg.pair_successor_future)
    if (future_winner_guard or future_horizon_bridge) and not active_future:
        raise ValueError('Future winner guard requires an R69 memory model')
    if (counterfactual_readout and
            model.pair_successor.counterfactual_weight is None):
        raise ValueError('Enable R74 counterfactual readout before scoring')
    if counterfactual_readout and recency_readout:
        raise ValueError('R74 and R75 candidate rails must be isolated')
    if recency_readout and model.pair_successor.recency_weight is None:
        raise ValueError('Enable R75 recency readout before scoring')
    recency_stream=None
    if recency_readout:
        recency_stream=CausalBookSuccessor(horizon=history,key_mode=key_mode,
                         recency_half_life=recency_half_life)
        recency_stream.begin_book(book_id)
    for arm in ARMS:
        kw=dict(horizon=history,key_mode=key_mode)
        if arm=="wrong_address":
            kw["query_map"]=wrong_pair
        if arm=="wrong_byte":
            kw["query_byte_map"]=wrong_byte
        if arm=="wrong_successor":
            kw["value_control"]="rotated"
        stream=CausalBookSuccessor(**kw)
        stream.begin_book(book_id)
        streams[arm]=stream
        if active_future:
            future_streams[arm]=[]
            for lag in (2,3):
                future=CausalLagBookSuccessor(lag=lag,**kw)
                future.begin_book(book_id)
                future_streams[arm].append(future)

    sums={arm:[0.0]*len(METRICS) for arm in ARMS}
    counts={arm:[0]*len(METRICS) for arm in ARMS}
    raw_targets=int(book_tokens[score_start+1:score_end+1].ne(256).sum().item())
    if raw_targets==0:
        raise ValueError("No literal bytes in requested scoring interval")
    start_time=perf_counter()
    for start in range(0,score_end,chunk):
        stop=min(score_end,start+chunk)
        left=max(0,stop-context)
        ids=book_tokens[left:stop].unsqueeze(0)
        _,hidden,_=model(ids)
        pairs=model._r54_geometry().r54_last_pairs.detach()
        if pairs.shape[:2]!=ids.shape:
            raise AssertionError("Stale or missing causal pair-address capture")
        # Warm-up is causal: input bytes and pair IDs update all histories,
        # but warm-up positions never contribute evaluation metrics.
        if stop<=score_start:
            for arm,stream in streams.items():
                for t in range(start,stop):
                    stream.step(book_id,int(book_tokens[t]),pairs[0,t-left].tolist())
                    if active_future:
                        for future in future_streams[arm]:
                            future.step(book_id,int(book_tokens[t]),
                                        pairs[0,t-left].tolist())
            if recency_stream is not None:
                for t in range(start,stop):
                    recency_stream.step(book_id,int(book_tokens[t]),
                                        pairs[0,t-left].tolist())
            continue
        raw_logits=model.head(hidden).detach()
        plus2,plus3=model.future_logits(hidden)
        query_start=max(start,score_start)
        lo=query_start-left
        hi=stop-left
        arm_values={}
        arm_histories={}
        future_values={}
        for arm,stream in streams.items():
            hist=torch.zeros_like(raw_logits)
            coverage=torch.zeros_like(ids,dtype=torch.float32)
            hits=torch.zeros_like(ids,dtype=torch.float32)
            for t in range(start,stop):
                evidence,cov,pos=stream.step(book_id,int(book_tokens[t]),
                                              pairs[0,t-left].tolist())
                if arm!="neutral":
                    hist[0,t-left]=evidence
                    coverage[0,t-left]=cov
                    hits[0,t-left]=pos
            if (torch.any(hist.sum(-1)>1.00001) or torch.any(coverage>1) or
                    torch.any(coverage<0) or torch.any(hits<0)):
                raise AssertionError(
                    f"Invalid causal arm={arm}, start={start}: "
                    f"hist_max={hist.sum(-1).max().item()}, "
                    f"coverage_max={coverage.max().item()}, "
                    f"coverage_min={coverage.min().item()}, "
                    f"hits_min={hits.min().item()}")
            arm_histories[arm]=(hist[:,lo:hi],coverage[:,lo:hi],hits[:,lo:hi])
            # Preserve inherited R55/R58 rails exactly in each matched arm.
            corrected,_=model.pair_successor(
                raw_logits,pairs,ids,
                joint_memory_override=(hist,coverage,hits))
            arm_values[arm]=corrected[0,lo:hi].detach()
            if active_future:
                evidence=[]
                for future in future_streams[arm]:
                    f_hist=torch.zeros_like(raw_logits)
                    f_cov=torch.zeros_like(ids,dtype=torch.float32)
                    f_hits=torch.zeros_like(ids,dtype=torch.float32)
                    for t in range(start,stop):
                        dist,cov,count=future.step(book_id,int(book_tokens[t]),
                                                   pairs[0,t-left].tolist())
                        if arm!='neutral':
                            f_hist[0,t-left]=dist
                            f_cov[0,t-left]=cov
                            f_hits[0,t-left]=count
                    evidence.extend((f_hist,f_cov,f_hits))
                future_values[arm]=model.pair_successor.future_forward(
                    (plus2,plus3),tuple(evidence),
                    protect_winner=future_winner_guard,
                    horizon_bridge=future_horizon_bridge)
        if counterfactual_readout:
            old=arm_values['exact']
            candidate=model.pair_successor.counterfactual_forward(
                old.unsqueeze(0),arm_histories['exact'],
                arm_histories['wrong_successor'])[0]
            arm_values['exact']=protect_parent_space_rank(old,candidate)
        if recency_stream is not None:
            recent_hist=torch.zeros_like(raw_logits)
            recent_cov=torch.zeros_like(ids,dtype=torch.float32)
            recent_hits=torch.zeros_like(recent_cov)
            for t in range(start,stop):
                dist,cov,count=recency_stream.step(book_id,int(book_tokens[t]),
                                                   pairs[0,t-left].tolist())
                recent_hist[0,t-left]=dist
                recent_cov[0,t-left]=cov
                recent_hits[0,t-left]=count
            old=arm_values['exact']
            candidate=model.pair_successor.recency_forward(old.unsqueeze(0),
                arm_histories['exact'],(recent_hist[:,lo:hi],
                        recent_cov[:,lo:hi],recent_hits[:,lo:hi]))[0]
            arm_values['exact']=protect_parent_space_rank(old,candidate)
        if space_rank_guard:
            for arm in ARMS:
                if arm!="neutral":
                    arm_values[arm]=protect_parent_space_rank(
                        arm_values["neutral"],arm_values[arm])
                if active_future:
                    future_values[arm]=tuple(protect_parent_space_rank(
                        old,new) for old,new in zip(future_values['neutral'],
                                                     future_values[arm]))
        for arm,logits in arm_values.items():
            pair=future_values[arm] if active_future else (plus2,plus3)
            for future,prediction in enumerate((
                    logits,pair[0][0,lo:hi],
                    pair[1][0,lo:hi])):
                horizon=future+1
                valid=min(stop-query_start,max(0,len(book_tokens)-query_start-horizon))
                if valid==0:
                    continue
                labels=book_tokens[query_start+horizon:query_start+horizon+valid]
                _append_stats(sums[arm],counts[arm],prediction[:valid],
                              labels,future)
    elapsed=perf_counter()-start_time
    output={}
    for arm in ARMS:
        if any(n==0 for n in counts[arm]):
            raise ValueError("Book lacks evaluated space/non-space or future targets")
        metrics={m:sums[arm][i]/counts[arm][i] for i,m in enumerate(METRICS)}
        output[arm]={
            **metrics,
            "bits_per_raw_byte":sums[arm][0]/(raw_targets*math.log(2)),
            "bits_per_byte_or_eos_target":metrics["ce_1"]/math.log(2),
            "positions_scored":counts[arm][0],
            "raw_target_bytes":raw_targets,
            "score_start":score_start,
            "score_end_exclusive":score_end,
            "evaluated_tokens_per_second":counts[arm][0]/max(elapsed,1e-9),
        }
    return output


@torch.no_grad()
def evaluate_disjoint_books(model,books,**kwargs):
    """One isolated R62 memory state per supplied book; no shared prefixes."""
    if not books:
        raise ValueError("No independent books supplied")
    out={}
    for book_id,tokens in books.items():
        identity=str(book_id)
        if identity in out:
            raise ValueError("Duplicate book identity")
        out[identity]=evaluate_causal_book(model,tokens,book_id,**kwargs)
    return out
