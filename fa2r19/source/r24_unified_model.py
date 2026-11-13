"""FA2 R24 unified model: PAL recurrent + mirror-MoE + R18/R21/R22/R23."""
import math,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from r18_long_memory import build_copy_queries,apply_copy_boost
from r21_relevance_novelty import causal_proposal
from r22_multiscale_memory import hierarchical_proposals
from r23_context64 import build_index

class SwiGLUExpert(nn.Module):
    def __init__(self,dim,width):
        super().__init__();self.norm=nn.LayerNorm(dim);self.up=nn.Linear(dim,width*2);self.down=nn.Linear(width,dim)
    def forward(self,x):
        a,b=self.up(self.norm(x)).chunk(2,-1)
        return self.down(F.silu(a)*b)
class CausalMirrorMoE(nn.Module):
    def __init__(self,dim,width=384,experts=4,topk=2):
        super().__init__()
        if not(1<=topk<=experts and width>=dim):raise ValueError('invalid experts')
        self.router=nn.Linear(dim*2,experts);self.experts=nn.ModuleList(SwiGLUExpert(dim,width) for _ in range(experts))
        self.mirror_gain=nn.Parameter(torch.tensor(0.));self.amplitude=nn.Parameter(torch.tensor(0.));self.topk=topk
    def forward(self,states):
        d=states.shape[-1];flat=states.reshape(-1,d);reflect=flat.flip(-1)
        probs=F.softmax(self.router(torch.cat((flat,reflect),-1)),-1)
        chosen,indices=probs.topk(self.topk,-1);chosen=chosen/chosen.sum(-1,keepdim=True).clamp_min(1e-8)
        combined=torch.zeros_like(flat);mirror=torch.tanh(self.mirror_gain)*reflect
        for j,exp in enumerate(self.experts):
            rows,slots=(indices==j).nonzero(as_tuple=True)
            if rows.numel():
                routed=exp(flat[rows]+(1 if j%2==0 else -1)*mirror[rows]);combined.index_add_(0,rows,routed*chosen[rows,slots,None])
        balance=probs.mean(0).square().sum()*len(self.experts)-1
        return combined.reshape_as(states),balance

def scramble_completed_values(memory):
    copied=dict(memory);vals=memory['values'].clone()
    for i in range(len(vals)):
        eligible=memory['mask'][i].nonzero(as_tuple=True)[0]
        if eligible.numel()>1:vals[i,eligible]=memory['values'][i,eligible.roll(1)]
    copied['values']=vals
    return copied

class UnifiedFA2(nn.Module):
    def __init__(self,model,bridge,adapter,r18,r21,r22,r23,*,local_history=512,expert_width=384,expert_count=4,expert_topk=2):
        super().__init__()
        if not 65<=local_history<=2048:raise ValueError('bad local history')
        self.model=model;self.bridge=bridge;self.adapter=adapter;self.r18=r18;self.r21=r21;self.r22=r22;self.r23=r23
        dim=model.memory.subword_out.in_features;self.moe=CausalMirrorMoE(dim,expert_width,expert_count,expert_topk);self.local_history=local_history
    def forward_chunk(self,book,segment,carry,*,counterfactual=False,donor_book=None,apply_r25=True,apply_r26=True,apply_r27=True):
        begin,end,t0,t1=segment
        examples=torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
        lexical=torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
        tags=torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None]-begin
        targets=torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
        if targets.numel()==0 or tags.min()<0 or tags.max()>=end-begin:raise ValueError('bad aligned chunk')
        hidden=self.model.core.hidden(examples.reshape(-1,7),lexical.reshape(-1,72)).reshape(1,end-begin,-1)
        logits,states=carry.forward_chunk(self.model,self.bridge,self.adapter,hidden,targets,tags,book_id=book.book_id,token_offset=t0,chunk_tag_offset=begin,return_states=True)
        # R30 is an opt-in, zero-output residual on the SAME learned PAL model.
        # Feedback reaches both the inherited MoE and the inherited decoder;
        # the latter path remains trainable even when the MoE gate is zero.
        r30_detail=None
        if hasattr(self,'r30'):
            r30_delta,r30_detail,carry.r30_apex=self.r30(
                states,prior_apex=getattr(carry,'r30_apex',None),
                return_detail=True,return_apex=True)
            deep,balance=self.moe(states+r30_delta)
            logits=logits[0]+torch.tanh(self.moe.amplitude)*self.model.memory.subword_out(deep)[0]
            logits=logits+F.linear(r30_delta,self.model.memory.subword_out.weight)[0]
        else:
            deep,balance=self.moe(states)
            logits=logits[0]+torch.tanh(self.moe.amplitude)*self.model.memory.subword_out(deep)[0]
        y=targets[0]
        from_pos=max(0,t0-self.local_history)
        prefix=torch.as_tensor(book.tokens[from_pos:t0].astype(np.int64));context=torch.cat((prefix,y))
        prior_logits=logits.new_zeros((len(prefix),logits.shape[-1]))
        q18=build_copy_queries(context,mem=self.r18.max_memory,gap=self.r18.min_gap)
        post18=apply_copy_boost(self.r18,torch.cat((prior_logits,logits),0),q18)[len(prefix):]
        r21_delta,f21,_=causal_proposal(self.r18,torch.cat((prior_logits,post18),0),context)
        post21,_=self.r21(post18,r21_delta[len(prefix):],f21[len(prefix):],query_position_offset=t0)
        proposals22,feats22,_=hierarchical_proposals(self.r18,post21,book.tokens,t0)
        post22,_=self.r22(post21,proposals22,feats22,query_position_offset=t0)
        actual=post22;permuted=post22;unrelated=None;detail=None;index=None
        if t0>=64:
            index=build_index(book.tokens,t0,t1-t0);index['query_position_offset']=t0
            actual,detail=self.r23(post22,index,piece_table=self.model.memory.piece_emb.weight)
            if counterfactual:
                permuted,_=self.r23(post22,scramble_completed_values(index),piece_table=self.model.memory.piece_emb.weight)
                if donor_book is not None and donor_book.book_id!=book.book_id:
                    donor_start=min(t0,len(donor_book.tokens)-(t1-t0))
                    if donor_start>=129:
                        alt=build_index(donor_book.tokens,donor_start,t1-t0,query_book_tokens=book.tokens,query_start=t0)
                        alt['query_position_offset']=t0
                        unrelated,_=self.r23(post22,alt,piece_table=self.model.memory.piece_emb.weight)
        uncalibrated=actual
        if apply_r25 and t0>=3 and hasattr(self,'r25'):
            from r25_episodic_cache import CausalNGramIndex
            if not hasattr(self,'_r25_book_indexes'):
                self._r25_book_indexes={}
            cache=self._r25_book_indexes.get(book.book_id)
            if cache is None:
                cache=CausalNGramIndex(book.tokens)
                self._r25_book_indexes[book.book_id]=cache
            samples,stat=cache.distributions(book.tokens,t0,len(y),actual.shape[-1])
            actual=self.r25(actual,torch.from_numpy(samples).to(actual.device),torch.from_numpy(stat).to(actual.device))
        before_r26=actual
        if apply_r25 and apply_r26 and t0>=6 and hasattr(self,'r26'):
            from r26_span_memory import CausalHighOrderIndex
            if not hasattr(self,'_r26_book_indexes'):
                self._r26_book_indexes={}
            high=self._r26_book_indexes.get(book.book_id)
            if high is None:
                high=CausalHighOrderIndex(book.tokens)
                # Bounded indexed-document cache on CPU; one active document.
                self._r26_book_indexes={book.book_id: high}
            # Gate features contain no target-dependent information; target is
            # supplied as zero only because the calibration API also emits p_y.
            base_prob=actual.detach().exp().cpu().numpy()
            dummy=np.zeros(len(y),dtype=np.int64)
            _,features=high.features(book.tokens,t0,len(y),base_prob,dummy)
            span=high.full_distributions(book.tokens,t0,len(y),actual.shape[-1])
            actual=self.r26(actual,torch.from_numpy(span).to(actual.device),
                torch.from_numpy(features).to(actual.device))
        before_r27=actual
        if apply_r25 and apply_r26 and apply_r27 and t0>=6 and hasattr(self,'r27'):
            from r27_trust import near_only_full
            near=near_only_full(self.r26,before_r26,
                torch.from_numpy(span).to(actual.device),
                torch.from_numpy(features).to(actual.device))
            actual=self.r27(before_r27,near,
                torch.from_numpy(features).to(actual.device),position=t0)
        byte_start=int(book.byte_starts[begin]);byte_end=int(book.byte_starts[end]) if end<len(book.byte_starts) else int(book.raw_len)
        if byte_end<=byte_start:raise ValueError('invalid byte span')
        return {'logits':actual,'without_r26':before_r26,'without_r27':before_r27,'without_r25':uncalibrated,'without_r23':post22,'scrambled_values':permuted,'unrelated_history':unrelated,'target':y,'bytes':byte_end-byte_start,'memory_detail':detail,'memory_index':index,'moe_balance':balance,'r30_detail':r30_detail,'start':t0}
