#!/usr/bin/env python3
"""FA1 R7.23 deep combinatorial diamond. One model, inherited 5 optimizers.

256 inherited R7 points -> 32640 original signed/mirror R7 pair tensors ->
2,763,520 individually addressable learned triple residues + 256 apex sectors ->
original pair topology -> inherited point topology -> SAME R7 tree decoder.

No targets or future context enter forward routing; target only enters loss.
Batch-wise: 8 active points -> 28 pairs -> 56 triples, exact combinadic IDs.
The large index is sparse ACTIVATION, not 2.7M dense transformer blocks.
"""
from __future__ import annotations
import copy, hashlib, itertools, pathlib, tempfile, os
import torch
from torch import nn
from torch.nn import functional as F
from fa1_r722_unified import load_unified, UnifiedR722
from fa1_r721_unified import blank
from r710_checkpoint_integrity import sha256, deep_equal

VERTICES=256
PAIRS=32640
TRIPLES=2763520
ACTIVE=8
WIDTH=32
PAIRS_LOCAL=list(itertools.combinations(range(ACTIVE),2))
TRIPLES_LOCAL=list(itertools.combinations(range(ACTIVE),3))
PAIR_LOOKUP={x:i for i,x in enumerate(PAIRS_LOCAL)}
TRI_EDGES=[[PAIR_LOOKUP[(i,j)],PAIR_LOOKUP[(i,k)],PAIR_LOOKUP[(j,k)]] for i,j,k in TRIPLES_LOCAL]


def pair_rank(a,b):
    """R7's original exact lexicographic pair address for a < b."""
    lo=torch.minimum(a,b); hi=torch.maximum(a,b)
    if bool((lo==hi).any()):raise ValueError('pair endpoints must be distinct')
    return lo*(511-lo)//2+hi-lo-1


def triple_rank(a,b,c):
    """Exact combinatorial number system, 0..C(256,3)-1 for sorted vertices."""
    x=torch.sort(torch.stack([a,b,c],dim=-1),dim=-1).values
    lo,mid,hi=x.unbind(-1)
    if bool(((lo==mid)|(mid==hi)).any()):raise ValueError('triple vertices must be distinct')
    return hi*(hi-1)*(hi-2)//6 + mid*(mid-1)//2 + lo


class MicroTower(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.norm2=nn.LayerNorm(dim)
    def forward(self,x):
        y,_=self.attn(x,x,x,need_weights=False)
        x=self.norm(x+y)
        return self.norm2(x+self.ffn(x))[:,-1]


class CombinatorialDiamond(nn.Module):
    """Exactly indexed triple apex, specializing 256-sector overmind.

    Legacy tensors H/P/M and E are read directly from the ORIGINAL R7 trunk,
    never cloned into another expert bank. The 256 top-level advisors are
    learned shared specialists; each of 2,763,520 triangles has an actual
    trainable 3-vector that forms its individual persistent specialization.
    """
    def __init__(self,width=WIDTH):
        super().__init__()
        d=width;self.d=d
        self.byte_embedding=nn.Embedding(256,d)
        self.byte_tower=MicroTower(d)
        self.word_projection=nn.Linear(96,d)
        self.word_tower=MicroTower(d)
        self.sent_projection=nn.Linear(96,d)
        self.sent_tower=MicroTower(d)
        self.hidden_projection=nn.Linear(96,d)
        self.bottom_up=nn.Linear(d*4,d)
        self.point_offset=nn.Embedding(VERTICES,d)
        self.point_in=nn.Linear(96,d)
        self.pair_in=nn.Linear(64,d)
        self.pair_fuse=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.LayerNorm(d))
        # Every triple has a distinct trainable address, not a hashed bucket.
        self.triple_code=nn.Embedding(TRIPLES,3)
        # Distinct low-magnitude codes give both the address table and its
        # shared projection gradients from the very first training update.
        # Gate stays at zero, so inherited predictions remain exactly equal.
        nn.init.normal_(self.triple_code.weight,std=0.01)
        self.triple_in=nn.Linear(3,d,bias=False)
        self.triangle_fuse=nn.Sequential(nn.Linear(d*4,d),nn.GELU(),nn.LayerNorm(d))
        self.triangle_query=nn.Linear(d,d,bias=False)
        self.triangle_score=nn.Linear(d,1)
        self.sector_keys=nn.Parameter(torch.randn(256,d)*0.02)
        self.sector_values=nn.Parameter(torch.randn(256,d)*0.02)
        self.sector_router=nn.Linear(d,d,bias=False)
        self.apex=nn.Sequential(nn.Linear(d*3,d*2),nn.GELU(),nn.Linear(d*2,d),nn.LayerNorm(d))
        self.triple_down=nn.Linear(d,d)
        self.pair_down=nn.Sequential(nn.Linear(d*2,d),nn.GELU())
        self.point_down=nn.Sequential(nn.Linear(d*2,d),nn.GELU())
        self.output=nn.Linear(d*3,96)
        # Learned apex meta-control: predicts when its own candidate provides
        # useful information beyond the entire inherited R7 distribution.
        # The target supervises this head in TRAINING only; forward routing
        # uses contextual representations, never the next-word target.
        self.trust=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.trust[-1].weight)
        nn.init.constant_(self.trust[-1].bias,-1.5)
        self.head_byte=nn.Linear(d,256)
        self.head_word=nn.Linear(d,96)
        self.gate=nn.Parameter(torch.zeros(()))
        # The full causal diamond is trained directly before the learned
        # mixture is opened; initialization remains bitwise R7.22-identical.
        self.register_buffer('edges',torch.tensor(PAIRS_LOCAL,dtype=torch.long),persistent=False)
        self.register_buffer('triangles',torch.tensor(TRIPLES_LOCAL,dtype=torch.long),persistent=False)
        self.register_buffer('tri_edges',torch.tensor(TRI_EDGES,dtype=torch.long),persistent=False)

    def forward(self,hidden,word_ids,byte_ids,sentence_ids,vertex_ids,trunk,
                two_way=True,without_apex=False,without_pair=False,return_trace=False):
        if word_ids.shape[1]!=32 or byte_ids.shape[1]!=128 or sentence_ids.shape[1]!=8 or vertex_ids.shape[1]!=ACTIVE:
            raise ValueError('expected 32 past words / 128 past bytes / 8 previous sentence starts / 8 distinct points')
        if bool((vertex_ids<0).any()) or bool((vertex_ids>=256).any()):raise ValueError('invalid points')
        if bool((torch.sort(vertex_ids,dim=1).values[:,1:]==torch.sort(vertex_ids,dim=1).values[:,:-1]).any()):
            raise ValueError('active points must be unique')
        batch=hidden.size(0);d=self.d
        # Three towers are the context that conditions EVERY diamond level.
        byte_input=self.byte_embedding(byte_ids.long()).reshape(batch,8,16,d).mean(2)
        byte=self.byte_tower(byte_input)
        words=F.embedding(word_ids.long(),trunk.E.detach()).reshape(batch,4,8,96).mean(2)
        word=self.word_tower(self.word_projection(words))
        sent=self.sent_tower(self.sent_projection(F.embedding(sentence_ids.long(),trunk.E.detach())))
        q=self.bottom_up(torch.cat([self.hidden_projection(hidden),byte,word,sent],-1))
        # 256 -> C(256,2): READ the exact original H and P/M, not new replacements.
        node=self.point_in(F.embedding(vertex_ids.long(),trunk.H.detach()))+self.point_offset(vertex_ids.long())
        e=self.edges
        ids_i=vertex_ids[:,e[:,0]]; ids_j=vertex_ids[:,e[:,1]]
        pair_ix=pair_rank(ids_i,ids_j)
        sign=torch.where(ids_i<ids_j,1.,-1.).unsqueeze(-1)
        old_pair=F.embedding(pair_ix,trunk.P.detach())-.2*F.embedding(pair_ix,trunk.M.detach())
        old_pair=sign*old_pair
        pair=self.pair_fuse(torch.cat([node[:,e[:,0]],node[:,e[:,1]],
                     self.pair_in(old_pair)],dim=-1))
        # C(256,2) -> C(256,3): every candidate triangle receives a unique
        # combinadic address and its own persistent trainable code.
        t=self.triangles
        tri_ids=triple_rank(vertex_ids[:,t[:,0]],vertex_ids[:,t[:,1]],vertex_ids[:,t[:,2]])
        tri_code=self.triple_in(self.triple_code(tri_ids))
        te=self.tri_edges
        tri=self.triangle_fuse(torch.cat([pair[:,te[:,0]],pair[:,te[:,1]],
                               pair[:,te[:,2]],tri_code],dim=-1))
        logits=self.triangle_score(torch.tanh(tri+self.triangle_query(q).unsqueeze(1))).squeeze(-1)
        triangle_weights=F.softmax(logits,dim=1)
        apex_in=(triangle_weights.unsqueeze(-1)*tri).sum(1)
        # Apex meta-mind: 256 trainable specializing operators inspect the
        # whole triangle summary, selecting four experts each token.
        route=F.normalize(self.sector_router(apex_in),dim=-1) @ F.normalize(self.sector_keys,dim=-1).T
        sectors,sector_ids=torch.topk(route,4,dim=-1)
        chosen=F.embedding(sector_ids,self.sector_values)
        sector_prob=F.softmax(sectors*4,dim=-1)
        meta=(sector_prob.unsqueeze(-1)*chosen).sum(1)
        apex=self.apex(torch.cat([apex_in,meta,q],dim=-1))
        if without_apex:apex=apex*0
        # C(256,3) -> C(256,2): send triangle judgments only to their 3 edges.
        down_tri=tri+self.triple_down(apex).unsqueeze(1) if two_way else tri
        pair_mail=torch.zeros_like(pair)
        for j in range(3):
            pair_mail.index_add_(1,te[:,j],down_tri*triangle_weights.unsqueeze(-1))
        pair_back=self.pair_down(torch.cat([pair,pair_mail],dim=-1))
        if without_pair:pair_back=pair_back*0
        # C(256,2) -> 256: edges reconstruct active point messages.
        point_mail=torch.zeros_like(node)
        point_mail.index_add_(1,e[:,0],pair_back)
        point_mail.index_add_(1,e[:,1],pair_back)
        point_back=self.point_down(torch.cat([node,point_mail/7],dim=-1))
        point_summary=point_back.mean(1)
        assembled=torch.cat([point_summary,apex,q],dim=-1)
        correction=0.125*self.output(assembled)
        trust_logit=self.trust(assembled).squeeze(-1)
        # Native inherited R7 word tree receives a context-modified state.
        # This distribution becomes active ONLY via a zero-init normalized
        # probability mixture in UnifiedR723Diamond.joint_nll.
        new_hidden=hidden+correction
        byte_logits=self.head_byte(point_summary+apex)
        target_vector=self.head_word(point_summary+apex)
        if return_trace:
            return new_hidden,byte_logits,target_vector,{
                'selected_sectors':sector_ids.detach(),
                'triangle_addresses':tri_ids.detach(),
                'triangle_attention':triangle_weights.detach(),
                'active_pair_addresses':pair_ix.detach(),
                'apex_std':apex.detach().std(),
                'gate':self.gate.detach().clone(),'trust_logit':trust_logit}
        return new_hidden,byte_logits,target_vector


class UnifiedR723Diamond(UnifiedR722):
    def __init__(self,core,calibration,near,far,semantic,long_semantic,diamond):
        # Keep SAME R7 inheritance, no duplicate frozen R7 tensors.
        super().__init__(core,calibration,near,far,semantic)
        self.long_semantic=long_semantic
        self.diamond=diamond

    def joint_nll(self,data,near_enabled=True,far_enabled=True,semantic_enabled=True,
                  diamond_enabled=True,two_way=True,return_extras=False,**ablations):
        # Retain the COMPLETE original R7.22 trunk + internal retrieval and
        # external near/far/semantic mixture. Never replace its NLL with the
        # tree-only likelihood (which would discard earlier R7 learned paths).
        old_nll,close=super().joint_nll(data,near_enabled,far_enabled,semantic_enabled)
        if not diamond_enabled:return old_nll,close
        h,b_logits,w_vec,trace=self.diamond(data['hidden'],data['word_ids'],
                 data['byte_ids'],data['sentence_ids'],data['vertex_ids'],
                 self.core.trunk,two_way=two_way,return_trace=True,**ablations)
        # Two normalized word distributions: inherited R7.22 distribution and
        # context-conditioned 32K-word hierarchical tree. Their probabilistic
        # mixture stays normalized for all vocabulary targets and is exactly
        # identical to old R7 when the bridge is closed.
        new_nll=self.core.trunk.nll(h,data['target'])
        # Context-dependent self-evaluation gates the *probability mixture*.
        # Global gate is zero at migration, preserving exact R7.22 parity.
        bridge=(0.25*torch.clamp(torch.tanh(self.diamond.gate),min=0.,max=1.)
                *torch.sigmoid(trace['trust_logit']))
        corrected=-torch.logaddexp(torch.log1p(-bridge)+(-old_nll),
                                  torch.log(bridge.clamp_min(1e-30))-new_nll)
        # Avoid 1e-30 numerical parity error when gate is exactly closed.
        corrected=torch.where(bridge==0,old_nll,corrected)
        if return_extras:return corrected,close,{'byte_logits':b_logits,'word_vector':w_vec,
                    'trace':trace,'bridge_weight':bridge,'branch_nll':new_nll}
        return corrected,close


def extend_r722(parent,root,threads=2):
    seed=sha256(parent)
    old,opts,base,comp,blob=load_unified(parent,root,threads)
    old_states=[copy.deepcopy(o.state_dict()) for o in opts]
    old_weights={k:v.clone() for k,v in old.state_dict().items()}
    diamond=CombinatorialDiamond()
    model=UnifiedR723Diamond(old.core,old.calibration,old.near,old.far,
                           old.semantic,old.long_semantic,diamond)
    opts[4].add_param_group({'params':list(diamond.parameters()),'lr':1e-3,'weight_decay':0.})
    for k,x in old_weights.items():assert torch.equal(model.state_dict()[k],x)
    for j in range(4):assert deep_equal(opts[j].state_dict(),old_states[j])
    after=opts[4].state_dict()
    assert after['param_groups'][:-1]==old_states[4]['param_groups']
    assert deep_equal(after['state'],old_states[4]['state'])
    return model,opts,base,comp,{'parent_sha256':seed,'accepted_step':blob['accepted_step']}


def save_diamond(path,model,opts,parent_sha,training_step=0,metrics=None):
    if len(opts)!=5 or len(opts[4].param_groups)!=8:raise ValueError('five optimizers and eight fifth groups required')
    obj={'format':'FA1_R7_23_DIAMOND_SINGLE_V1','accepted_step':14416,
         'r722_ancestor_sha256':parent_sha,'optimizer_count':5,
         'model':model.state_dict(),
         'optimizer_states':[o.state_dict() for o in opts],
         'new_diamond_updates':training_step,
         'selection':metrics or {'status':'identity-preserving migration'},
         'promoted':False,'topology':{'points':256,'pairs':32640,'triples':2763520,
                                    'specialists':256,'active_points':8}}
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.diamond_',suffix='.pending');os.close(fd)
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return {'sha256':sha256(path),'bytes':path.stat().st_size}


def load_diamond(path,root,threads=2):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('format')!='FA1_R7_23_DIAMOND_SINGLE_V1' or obj.get('accepted_step')!=14416 or obj.get('promoted') or obj.get('optimizer_count')!=5:
        raise ValueError('invalid R7.23 diamond lineage')
    old,opts,base,comp=blank(root,threads,embedded_state=obj['model'])
    from fa1_r722_semantic import SemanticGate
    old=UnifiedR722(old.core,old.calibration,old.near,old.far)
    opts[4].add_param_group({'params':list(old.semantic.parameters())+list(old.long_semantic.parameters()),'lr':5e-4,'weight_decay':0.})
    model=UnifiedR723Diamond(old.core,old.calibration,old.near,old.far,old.semantic,old.long_semantic,CombinatorialDiamond())
    opts[4].add_param_group({'params':list(model.diamond.parameters()),'lr':1e-3,'weight_decay':0.})
    model.load_state_dict(obj['model'],strict=True)
    for o,s in zip(opts,obj['optimizer_states']):o.load_state_dict(s)
    assert len(opts)==5 and len(opts[-1].param_groups)==8
    model.eval()
    return model,opts,base,comp,obj