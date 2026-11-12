"""R7 trainable ordered 2048-WORD memory + reversible subword compositional channel.

One additive zero-initialized adapter on the authentic frozen R6 PAL5/PAL7/PAL9/PAL11 trunk.
Causal memory sees only strictly earlier training/dev/test target positions.
32 x 64 ordered word blocks: recurrent summary, soft block attention, and sparse
exact retrieval of two selected distant blocks, plus 64 exact recent words.
"""
import os,sys,pathlib,math,numpy as np,torch
from torch import nn
import torch.nn.functional as F
import sentencepiece as spm
from build_lossless_subwords import encode_bytes,decode_bytes
from r7_core import D,R6,Corpus
WORD_CONTEXT=2048;LOCAL=64;BLOCK=64;NCHUNKS=(WORD_CONTEXT-LOCAL)//BLOCK;MAXPIECE=24

class TokenCompositor:
    def __init__(self,root,r6):
        root=pathlib.Path(root)
        self.sp=spm.SentencePieceProcessor(model_file=str(root/'tokenizer/gutenberg_r7.model'))
        self.npieces=256+self.sp.get_piece_size();self.eos=self.npieces
        vocab=r6.vocab
        self.piece_mat=np.zeros((len(vocab),MAXPIECE),dtype=np.int64)
        self.piece_len=np.zeros(len(vocab),dtype=np.int64)
        self.piece_mat[1,0]=self.eos;self.piece_len[1]=1
        for i,word in enumerate(vocab):
            if i<2:continue
            ids=encode_bytes(self.sp,word.encode('utf8'))[:MAXPIECE]
            if not ids:ids=[self.eos]
            self.piece_mat[i,:len(ids)]=ids;self.piece_len[i]=len(ids)
        self.piece_mat_t=torch.from_numpy(self.piece_mat)
        self.piece_len_t=torch.from_numpy(self.piece_len)
    def strings(self,corp,indices,vocab):
        return [corp.spelling(int(i),vocab) for i in np.asarray(indices).reshape(-1)]
    def encode_words(self,words):
        ids=np.zeros((len(words),MAXPIECE),dtype=np.int64);lens=np.zeros(len(words),dtype=np.int64)
        for k,word in enumerate(words):
            parts=encode_bytes(self.sp,word.encode('utf8'))[:MAXPIECE]
            if not parts:parts=[self.eos]
            ids[k,:len(parts)]=parts;lens[k]=len(parts)
        return torch.from_numpy(ids),torch.from_numpy(lens)
    def local_pieces(self,corp,occur,ids,vocab):
        # Shape [B,64,MAXPIECE] for the exact most-recent 64 words.
        arr=self.piece_mat[ids].copy();lens=self.piece_len[ids].copy()
        match=np.argwhere((ids==0)&(occur>=0))
        for i,j in match:
            spelling=corp.spelling(int(occur[i,j]),vocab)
            enc=encode_bytes(self.sp,spelling.encode('utf8'))[:MAXPIECE]
            arr[i,j,:]=0;arr[i,j,:len(enc)]=enc;lens[i,j]=len(enc)
        return torch.from_numpy(arr),torch.from_numpy(lens)

class R7Memory(nn.Module):
    def __init__(self,r6,compositor,word_context=2048,model_dim=96):
        super().__init__();assert word_context==WORD_CONTEXT and model_dim==D
        self.vocab=r6.V if hasattr(r6,'V') else 32000
        self.register_buffer('word_embedding',torch.from_numpy(r6.E.copy()),persistent=False)
        self.piece_emb=nn.Embedding(compositor.npieces+1,64,padding_idx=0)
        self.piece_proj=nn.Linear(64,D,bias=False)
        nn.init.normal_(self.piece_emb.weight,std=.025)
        with torch.no_grad():self.piece_emb.weight[0].zero_()
        nn.init.zeros_(self.piece_proj.weight)
        self.query=nn.Linear(D,D,bias=False)
        self.local_key=nn.Linear(D,D,bias=False);self.local_val=nn.Linear(D,D,bias=False)
        self.chunk_key=nn.Linear(D,D,bias=False);self.chunk_val=nn.Linear(D,D,bias=False)
        self.loc_pos=nn.Parameter(torch.zeros(LOCAL,D));self.chunk_pos=nn.Parameter(torch.zeros(NCHUNKS,D))
        self.forget_gate=nn.Linear(D,D);self.mem_in=nn.Linear(D,D)
        self.final=nn.Linear(3*D,D,bias=False)
        # Gamma zero => exactly same shared NOW output as R6 before the first update.
        self.residual_scale=nn.Parameter(torch.zeros(()))
        self.subword_cell=nn.GRUCell(64,D)
        self.subword_out=nn.Linear(D,compositor.npieces+1)
        self.start_piece=compositor.eos
        self.npieces=compositor.npieces+1
        self.register_buffer('word_piece_mat',compositor.piece_mat_t,persistent=False)
        self.register_buffer('word_piece_len',compositor.piece_len_t,persistent=False)
    def compose(self,pieces,lens):
        # Reuses the SAME embedding matrix for composition and subword generation.
        vectors=self.piece_emb(pieces)
        mask=(torch.arange(pieces.shape[-1],device=pieces.device)<lens[...,None]).to(vectors.dtype)
        avg=(vectors*mask[...,None]).sum(-2)/lens.clamp(min=1)[...,None]
        return self.piece_proj(avg)
    def forward(self,h,ctx,local_pieces,local_lens,ablate=False,ablate_long=False,ablate_subwords=False,ablate_recurrent=False,ablate_exact=False,ablate_local=False,word_embedding_override=None):
        # ctx is the last 2048 observed WORD targets; target is not visible.
        B=ctx.shape[0];assert ctx.shape[1]==WORD_CONTEXT
        old=F.embedding(ctx,self.word_embedding if word_embedding_override is None else word_embedding_override)
        local=old[:,-LOCAL:,:]+(0. if ablate_subwords else self.compose(local_pieces,local_lens))
        local=local+self.loc_pos[None,:,:]
        old_blocks=old[:,:-LOCAL,:].reshape(B,NCHUNKS,BLOCK,D)
        chunks=old_blocks.mean(dim=2)+self.chunk_pos[None,:,:]
        q=self.query(h)
        local_score=(self.local_key(local)*q[:,None,:]).sum(-1)/(D**.5)
        local_weight=F.softmax(local_score,dim=-1)
        local_out=(local_weight[...,None]*self.local_val(local)).sum(1)
        cscore=(self.chunk_key(chunks)*q[:,None,:]).sum(-1)/(D**.5)
        cweight=F.softmax(cscore,dim=-1)
        chunk_out=(cweight[...,None]*self.chunk_val(chunks)).sum(1)
        # Exact token read from the two highest-affinity earlier memory chunks.
        selected=cscore.topk(2,dim=1).indices
        selected_blocks=old_blocks[torch.arange(B)[:,None],selected].reshape(B,2*BLOCK,D)
        exact_score=(self.local_key(selected_blocks)*q[:,None,:]).sum(-1)/(D**.5)
        exact_weight=F.softmax(exact_score,dim=-1)
        distant_out=(exact_weight[...,None]*self.local_val(selected_blocks)).sum(1)
        # Content-dependent gated recurrence over the complete 1984-word older history.
        state=torch.zeros_like(h)
        gate=self.forget_gate(chunks).sigmoid();update=self.mem_in(chunks).tanh()
        for t in range(NCHUNKS):state=gate[:,t]*state+(1-gate[:,t])*update[:,t]
        memory_out=local_out+chunk_out+distant_out+state
        if ablate_long:
            chunk_out=torch.zeros_like(chunk_out);distant_out=torch.zeros_like(distant_out);state=torch.zeros_like(state)
        if ablate_recurrent:state=torch.zeros_like(state)
        if ablate_exact:distant_out=torch.zeros_like(distant_out)
        if ablate_local:local_out=torch.zeros_like(local_out)
        mem=self.final(torch.cat((local_out,chunk_out+distant_out,state),dim=-1))
        z=torch.atanh(torch.clamp(h,-.999999,.999999))
        new_h=torch.tanh(z+(0. if ablate else self.residual_scale)*mem)
        return new_h,{'local':local_out,'chunks':chunk_out,'recurrent':state,'residual_scale':self.residual_scale,'mem_norm':memory_out.norm(dim=-1).mean()}
    def subword_loss(self,h,targets,lengths,teacher_prob=1.0):
        # True causal teacher forcing: token at time t predicts token at t+1.
        # The target word is NEVER fed into the R7 NOW/context memory.
        current=torch.full((len(h),),self.start_piece,dtype=torch.long,device=h.device)
        state=h;loss=0.;count=0
        for t in range(int(lengths.max())+1):
            state=self.subword_cell(self.piece_emb(current),state)
            logits=self.subword_out(state)
            desired=torch.where(t<lengths,targets[:,min(t,targets.shape[1]-1)],torch.full_like(current,self.start_piece))
            valid=t<=lengths
            loss=loss+F.cross_entropy(logits[valid],desired[valid],reduction='sum')
            count+=int(valid.sum())
            if t<targets.shape[1]:current=targets[:,t]
        return loss/max(1,count)
    @torch.no_grad()
    def spell(self,h,compositor,max_len=32,seed=0):
        gen=torch.Generator().manual_seed(seed);prev=torch.tensor([self.start_piece]);state=h.reshape(1,D)
        ids=[]
        for _ in range(max_len):
            state=self.subword_cell(self.piece_emb(prev),state)
            logits=self.subword_out(state)
            probs=F.softmax(logits,dim=-1)
            next_id=int(torch.multinomial(probs,1,generator=gen).item());
            if next_id==self.start_piece:break
            ids.append(next_id);prev=torch.tensor([next_id])
        try:return decode_bytes(compositor.sp,ids).decode('utf8')
        except UnicodeDecodeError:return '<invalid_utf8>'
