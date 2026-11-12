#!/usr/bin/env python3
"""PAL R18: causal, learned, long-distance token-addressable retrieval.

Extends the ONE inherited PAL R12/R15 recurrent model. It is an auxiliary
memory module, not another language model. Never reads the target at query t:
slot j holds key=(tokens[j-2],tokens[j-1]), value=tokens[j], j<=t-min_gap.
Old decoder logits are invariant when memory.logit_gain=0. Exact matches are
used as an optional safety *candidate filter*; Q/K similarity still ranks
hard negatives. The standalone retrieval task is explicitly a lexical copy
probe, not proof of semantic reasoning or frontier performance.
"""
from collections import defaultdict,deque
import math,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class R18LongMemory(nn.Module):
    def __init__(self,old_piece_embeddings:torch.Tensor,dim=64,key_dim=64,max_memory=2048,min_gap=65,topk=8):
        super().__init__()
        self.register_buffer('inherited_piece_embeddings',old_piece_embeddings.detach().clone(),persistent=False)
        width=2*old_piece_embeddings.shape[1]
        self.key_net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim),nn.GELU(),nn.Linear(dim,key_dim))
        self.query_net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim),nn.GELU(),nn.Linear(dim,key_dim))
        self.log_temperature=nn.Parameter(torch.tensor(0.))
        self.logit_gain=nn.Parameter(torch.tensor(0.))  # exact logit identity at migration
        self.recency=nn.Parameter(torch.tensor(0.))
        self.max_memory=max_memory;self.min_gap=min_gap;self.topk=topk

    def encode(self,pairs:torch.Tensor,query:bool):
        features=self.inherited_piece_embeddings[pairs.long()].flatten(-2)
        network=self.query_net if query else self.key_net
        return F.normalize(network(features),dim=-1,eps=1e-5)

    def scores(self,query_pairs,key_pairs):
        q=self.encode(query_pairs,True)
        k=self.encode(key_pairs,False)
        return (q.unsqueeze(-2)*k).sum(-1)*self.log_temperature.exp().clamp(.1,50)

    def contrastive_loss(self,query_pairs,candidate_pairs):
        scores=self.scores(query_pairs,candidate_pairs)
        return F.cross_entropy(scores,torch.zeros(len(scores),dtype=torch.long,device=scores.device)),scores

    def retrieve_logits(self,logits:torch.Tensor,tokens:torch.Tensor,learned=True,min_gap=None,max_memory=None):
        """Return updated logits and counts; all candidate values strictly precede t.

        Works on a single complete prediction stream. Limited to max_memory
        causal slots; online use can maintain the same slots incrementally.
        The exact two-token candidate is a conservative enablement guard to
        reduce harm on ordinary prose; approximate decoys share the last token.
        """
        assert logits.ndim==2 and tokens.ndim==1 and logits.shape[0]==tokens.shape[0]
        n=len(tokens); gap=self.min_gap if min_gap is None else min_gap
        mem=self.max_memory if max_memory is None else max_memory
        out=logits.clone();used=0;hits=0;distances=[];eligible=[]
        # For memory j, its key is prefix at j. The query t also uses prefix.
        last_map=defaultdict(deque); pair_map=defaultdict(deque)
        for t in range(n):
            j=t-gap
            if j>=2:
                pre=int(tokens[j-1]);pair=(int(tokens[j-2]),pre)
                last_map[pre].append(j);pair_map[pair].append(j)
            # Expire distant entries without growing unbounded histories.
            expired=t-mem-1
            if expired>=2:
                pre=int(tokens[expired-1]);pair=(int(tokens[expired-2]),pre)
                if last_map[pre] and last_map[pre][0]==expired:last_map[pre].popleft()
                if pair_map[pair] and pair_map[pair][0]==expired:pair_map[pair].popleft()
            if t<gap+2:continue
            a,b=int(tokens[t-2]),int(tokens[t-1])
            if b in (32,10,13):continue
            exact=pair_map.get((a,b))
            if not exact:continue
            # Include difficult distractors with the same most recent token.
            candidates=list(last_map[b])
            if not candidates:continue
            # Limit CPU cost, retaining the most recent slots plus the oldest
            # matching pair in the current memory horizon.
            if len(candidates)>64:
                tail=candidates[-63:];p=exact[0]
                candidates=([p] if p not in tail else [])+tail
            key_pairs=torch.tensor([(int(tokens[v-2]),int(tokens[v-1])) for v in candidates],device=logits.device)
            query=torch.tensor([a,b],device=logits.device)
            if learned:
                scores=self.scores(query,key_pairs)
            else:
                scores=torch.tensor([4.0 if (int(tokens[v-2]),int(tokens[v-1]))==(a,b) else 0.0 for v in candidates],device=logits.device)
            # Learned distance prior breaks identical-key ties.
            d=torch.tensor([t-v for v in candidates],device=logits.device,dtype=torch.float32)
            scores=scores-self.recency*torch.log1p(d)
            k=min(self.topk,len(candidates))
            idx=scores.topk(k).indices
            weights=F.softmax(scores[idx],dim=0)
            values=tokens[torch.tensor(candidates,device=logits.device)[idx]].long()
            boost=torch.zeros_like(logits[t]).scatter_add(0,values,weights)
            # No auxiliary logits head: additive copy evidence to inherited decoder.
            out[t]=out[t]+self.logit_gain*boost
            used+=1;eligible.append(t)
            if bool((values==tokens[t]).any()):hits+=1
            distances.append(int(d[idx[0]]))
        meta={'enabled_positions':used,'target_in_topk':hits,'average_top1_distance':float(np.mean(distances)) if distances else None,
              'positions':eligible}
        return out,meta


def collect_hard_negatives(books,max_examples=6000,seed=42,mem=2048,gap=65,negatives=15):
    """Whole-book disjoint split caller supplies; per-query positives BEFORE query.
    Positive shares a 2-token prefix; negatives share one token or random keys.
    Exclude trivial whitespace targets, and constrain positive gap to [gap,mem].
    """
    rng=random.Random(seed);examples=[]
    per_book=max(100,max_examples//max(1,len(books)))
    for b in books:
        tok=np.asarray(b.tokens,dtype=np.int32)
        seen=defaultdict(deque);last=defaultdict(deque);created=0
        # Randomized stride through entire book, deterministic by seed.
        stride=1
        for t in range(2,len(tok)):
            j=t-gap
            if j>=2:
                pair=(int(tok[j-2]),int(tok[j-1]))
                seen[pair].append(j);last[pair[1]].append(j)
            old=t-mem-1
            if old>=2:
                pair=(int(tok[old-2]),int(tok[old-1]))
                if seen[pair] and seen[pair][0]==old:seen[pair].popleft()
                if last[pair[1]] and last[pair[1]][0]==old:last[pair[1]].popleft()
            if t<gap+2 or rng.randrange(4):continue
            a,z=int(tok[t-2]),int(tok[t-1]);val=int(tok[t])
            if z in (32,10,13) or val in (32,10,13):continue
            positives=seen.get((a,z))
            if not positives:continue
            pos=rng.choice(positives)
            # Hard: same last token but different first token, different values
            dec=[x for x in last[z] if int(tok[x-2])!=a and int(tok[x])!=val]
            if len(dec)<max(4,negatives//2):continue
            neg=rng.sample(dec,min(len(dec),negatives))
            if len(neg)<negatives:
                pool=[x for x in range(max(2,t-mem),t-gap+1) if tok[x]!=val and x!=pos]
                if not pool:continue
                neg.extend(rng.choices(pool,k=negatives-len(neg)))
            rng.shuffle(neg)
            cand=[pos]+neg
            examples.append({'q':[a,z], 'keys':[[int(tok[x-2]),int(tok[x-1])] for x in cand],
                             'values':[int(tok[x]) for x in cand], 'target':val,'distance':t-pos,
                             'book':b.book_id,'query_position':t})
            created+=1
            if created>=per_book:break
    rng.shuffle(examples)
    return examples[:max_examples]


def batch_examples(examples,device='cpu'):
    return torch.tensor([x['q'] for x in examples],dtype=torch.long,device=device),torch.tensor([x['keys'] for x in examples],dtype=torch.long,device=device)

@torch.no_grad()
def test_retrieval(net,examples,batchsize=128):
    net.eval();n=0;correct=0;ce=0.;hard_correct=0;outcomes=[]
    for k in range(0,len(examples),batchsize):
        part=examples[k:k+batchsize]
        q,c=batch_examples(part)
        score=net.scores(q,c)
        pred=score.argmax(-1)
        n+=len(part);correct+=int((pred==0).sum())
        ce+=float(F.cross_entropy(score,torch.zeros(len(part),dtype=torch.long),reduction='sum'))
        hard_correct+=int(((score[:,0,None]>score[:,1:]).all(-1)).sum())
        outcomes.extend([{'book':x['book'],'distance':x['distance'],'correct':bool(pi==0)} for x,pi in zip(part,pred.tolist())])
    bins={}
    for low,high in ((65,128),(129,256),(257,512),(513,1024),(1025,2048)):
        xs=[r for r in outcomes if low<=r['distance']<=high]
        bins[f'{low}-{high}']={'n':len(xs),'top1':sum(z['correct'] for z in xs)/len(xs) if xs else None}
    return {'examples':n,'top1':correct/max(n,1),'nll':ce/max(n,1),
            'strict_hard_negative_accuracy':hard_correct/max(n,1),'distance_bins':bins}

def build_copy_queries(tokens,mem=2048,gap=65,max_candidates=64):
    """Prepare causal memory addresses without consulting current target tokens.
    `tokens[t]` is read only AFTER the query for t has been constructed.
    Stores index/value for slots j<=t-gap, so no future leakage.
    """
    x=[int(v) for v in tokens];N=len(x)
    # Bound per-token candidate lookup to O(max_candidates), even if a token
    # occurs tens of thousands of times within the memory window.
    seen_tail=defaultdict(lambda:deque(maxlen=max_candidates))
    seen_count=defaultdict(int);paired=defaultdict(deque)
    rows=[]
    for t in range(N):
        j=t-gap
        if j>=2:
            pair=(x[j-2],x[j-1]);seen_tail[pair[1]].append(j);seen_count[pair[1]]+=1;paired[pair].append(j)
        old=t-mem-1
        if old>=2:
            pair=(x[old-2],x[old-1])
            seen_count[pair[1]]-=1
            if seen_tail[pair[1]] and seen_tail[pair[1]][0]==old:seen_tail[pair[1]].popleft()
            if paired[pair] and paired[pair][0]==old:paired[pair].popleft()
        if t<gap+2:continue
        a,b=x[t-2],x[t-1]
        if b in (32,10,13) or not paired.get((a,b)):continue
        slots=list(seen_tail[b]);oldest=paired[(a,b)][0]
        if seen_count[b]>max_candidates:
            slots=slots[-(max_candidates-1):]
            if oldest not in slots:slots.insert(0,oldest)
        # Unlike the training label, candidate values are known history.
        # Include a checkable bound to avoid disguised future reads.
        assert all(j<=t-gap and j>=max(2,t-mem) for j in slots)
        rows.append((t,(a,b),[(x[j-2],x[j-1]) for j in slots],[x[j] for j in slots],[t-j for j in slots]))
    if not rows:return None
    cap=max(len(r[2]) for r in rows);E=len(rows)
    query=torch.tensor([r[1] for r in rows],dtype=torch.long)
    keys=torch.zeros((E,cap,2),dtype=torch.long)
    values=torch.zeros((E,cap),dtype=torch.long)
    distances=torch.zeros((E,cap),dtype=torch.float32)
    mask=torch.zeros((E,cap),dtype=torch.bool)
    for i,r in enumerate(rows):
        L=len(r[2]);keys[i,:L]=torch.tensor(r[2]);values[i,:L]=torch.tensor(r[3]);distances[i,:L]=torch.tensor(r[4]);mask[i,:L]=True
    return {'positions':torch.tensor([r[0] for r in rows],dtype=torch.long),
            'query':query,'keys':keys,'values':values,'distances':distances,'mask':mask,
            'max_source_position':max(x[0]-gap for x in rows)}


def apply_copy_boost(net,base_logits,copy_queries,disable=False,baseline_rank=False):
    if disable or copy_queries is None:
        return base_logits
    data=copy_queries
    scores=net.scores(data['query'],data['keys'])
    if baseline_rank:
        scores=4.*(data['query'][:,None,:]==data['keys']).all(-1).float()
    scores=scores-net.recency*torch.log1p(data['distances'].clamp_min(1))
    scores=scores.masked_fill(~data['mask'],-1e8)
    k=min(net.topk,scores.shape[1])
    sc,ind=scores.topk(k,dim=-1)
    val=data['values'].gather(1,ind)
    # Do not normalize masked entries when candidate count < k.
    valid=data['mask'].gather(1,ind)
    sc=sc.masked_fill(~valid,-1e8)
    weights=F.softmax(sc,dim=-1)*valid
    boost=torch.zeros((len(sc),base_logits.shape[-1]),dtype=base_logits.dtype,device=base_logits.device)
    boost=boost.scatter_add(1,val,weights)
    full=torch.zeros_like(base_logits).index_add(0,data['positions'],boost)
    return base_logits+net.logit_gain*full
