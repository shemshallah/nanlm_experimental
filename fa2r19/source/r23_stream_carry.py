import math,torch
from torch.nn import functional as F
class ContinuousPALState:
    def __init__(self,book_id=None):self.reset(book_id)
    def reset(self,book_id):
        self.book_id=book_id;self.cursor=0;self.state=None;self.previous=None;self.last_tag=None;self.key_history=[];self.value_history=[];self.r30_apex=None
    def detach(self):
        if self.state is not None:self.state=self.state.detach()
        if self.r30_apex is not None:self.r30_apex=self.r30_apex.detach()
        self.key_history=[x.detach() for x in self.key_history];self.value_history=[x.detach() for x in self.value_history]
    def forward_chunk(self,model,bridge,adapter,hidden,targets,tags,*,book_id,token_offset,chunk_tag_offset,detach_after=False,return_states=False):
        if self.book_id is None:self.book_id=book_id
        if book_id!=self.book_id:raise ValueError('cross-book state leakage')
        if int(token_offset)!=self.cursor:raise ValueError('noncontiguous document')
        if hidden.ndim!=3 or targets.ndim!=2 or tags.shape!=targets.shape:raise ValueError('bad shapes')
        B,T=targets.shape
        if T==0 or hidden.shape[0]!=B or bool((targets<0).any()):raise ValueError('bad stream')
        if int(tags.min())<0 or int(tags.max())>=hidden.shape[1]:raise ValueError('tag bounds')
        if self.previous is None:
            self.previous=torch.full((B,),model.memory.start_piece,dtype=torch.long,device=targets.device)
            self.state=hidden[:,0];self.last_tag=tags[:,0]+chunk_tag_offset
        retain=torch.sigmoid(bridge.retain_logit);logits=[];states=[]
        for t in range(T):
            local_tag=tags[:,t];global_tag=local_tag+chunk_tag_offset
            nxt=hidden.gather(1,local_tag[:,None,None].expand(-1,1,hidden.shape[-1]))[:,0]
            changed=global_tag!=self.last_tag
            self.state=torch.where(changed[:,None],retain*self.state+(1-retain)*bridge.context_gain*nxt,self.state)
            self.state=model.memory.subword_cell(model.memory.piece_emb(self.previous),self.state)
            refiner=getattr(self,"recurrent_refiner",None)
            if refiner is not None:self.state=refiner(self.state)
            out=self.state
            if adapter is not None:
                if self.key_history:
                    keys=torch.stack(self.key_history,1);values=torch.stack(self.value_history,1)
                    scores=(adapter.query(self.state)[:,None,:]*keys).sum(-1)/math.sqrt(self.state.shape[-1])
                    top=scores.topk(min(adapter.topk,scores.shape[1]),dim=-1)
                    chosen=values.gather(1,top.indices[:,:,None].expand(-1,-1,self.state.shape[-1]))
                    rec=(chosen*F.softmax(top.values,-1)[:,:,None]).sum(1)
                    out=out+adapter.attn_gain*adapter.attn_out(rec)
                out=out+adapter.ffn_gain*adapter.ffn(self.state)
                self.key_history.append(adapter.key(self.state));self.value_history.append(adapter.value(self.state))
                if len(self.key_history)>adapter.capacity:self.key_history.pop(0);self.value_history.pop(0)
            logits.append(model.memory.subword_out(out));states.append(out)
            self.previous=targets[:,t];self.last_tag=global_tag
        self.cursor+=T
        if detach_after:self.detach()
        out=torch.stack(logits,1)
        return (out,torch.stack(states,1)) if return_states else out
    def snapshot(self):return {'book_id':self.book_id,'token_cursor':self.cursor,'attention_cached':len(self.key_history),'has_recurrent_state':self.state is not None}

def contiguous_book_picks(book,chunk_span):
    if chunk_span<1:raise ValueError('bad chunk span')
    n=len(book.chunks)
    for st in range(0,n,chunk_span):
        en=min(n,st+chunk_span);t0=int(book.chunks[st]);t1=int(book.chunks[en]) if en<n else len(book.tokens)
        if t1>t0:yield st,en,t0,t1
