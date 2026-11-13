#!/usr/bin/env python3
"""FA1 R7.24: exact oriented simplex geometry inside one inherited R7 Diamond.

0-simplex: 256 original point embeddings H.
1-simplex: C(256,2) original mirror pair codes P/M, S+ even, S- odd.
2-simplex: C(256,3) original triple addresses, mirror S+ even, S- odd;
           trainable connection holonomy and spherical angle curvature.
3-simplex: C(256,4) exact addresses but SPARSE active tetrahedra (70 for 8
           points), no 174,792,640-wide dense parameter table.

Discrete curvature is not an automatic zero: there is an independently learned
triangular curvature 2-form whose coboundary is measured on every active tetra.
All routing inputs are target-free and strictly inherited from causal context.
"""
from __future__ import annotations
import itertools,math
import torch
from torch import nn
from torch.nn import functional as F

VERTICES=256
PAIRS=32640
TRIPLES=2763520
TETRAHEDRA=174792640
ACTIVE=8
LOCAL_E=list(itertools.combinations(range(ACTIVE),2))
LOCAL_F=list(itertools.combinations(range(ACTIVE),3))
LOCAL_T=list(itertools.combinations(range(ACTIVE),4))
EI={t:i for i,t in enumerate(LOCAL_E)}
FI={t:i for i,t in enumerate(LOCAL_F)}
FACE_TO_EDGES=[[EI[(a,b)],EI[(a,c)],EI[(b,c)]] for a,b,c in LOCAL_F]
TET_TO_FACES=[[FI[(b,c,d)],FI[(a,c,d)],FI[(a,b,d)],FI[(a,b,c)]] for a,b,c,d in LOCAL_T]
TET_SIGNS=(1.,-1.,1.,-1.)

def rank4(a,b,c,d):
    z=torch.sort(torch.stack([a,b,c,d],dim=-1),dim=-1).values
    q,r,s,t=z.unbind(-1)
    if bool(((q==r)|(r==s)|(s==t)).any()):
        raise ValueError('tetrahedral vertices must be distinct')
    return t*(t-1)*(t-2)*(t-3)//24+s*(s-1)*(s-2)//6+r*(r-1)//2+q


def triangle_parity(a,b,c):
    # Alternating orientation under one swap of vertex labels.
    return torch.sign(((b-a)*(c-a)*(c-b)).float())


class CurvatureDiamond(nn.Module):
    """An *internal* point/edge/face/tetra exchange; not a replacement LM.

    Batches activate 8 vertices, 28 edges, 56 faces, 70 tetrahedra. All
    globally addressed edges/faces are exact combinadic identities; tetra
    operators share weights rather than creating 175M separate parameters.
    """
    def __init__(self, width=32):
        super().__init__()
        d=width
        self.width=d
        self.mirror_P=nn.Linear(64,d,bias=False)
        self.mirror_M=nn.Linear(64,d,bias=False)
        self.pair_even=nn.Sequential(nn.Linear(2*d,d),nn.SiLU(),nn.LayerNorm(d))
        self.pair_odd=nn.Sequential(nn.Linear(2*d,d),nn.SiLU(),nn.LayerNorm(d))
        self.edge_connection=nn.Sequential(nn.Linear(2*d,d),nn.Tanh(),nn.Linear(d,1))
        self.point_sphere=nn.Linear(d,3,bias=False)
        self.face_even=nn.Sequential(nn.Linear(2*d+5,d),nn.SiLU(),nn.LayerNorm(d))
        self.face_odd=nn.Sequential(nn.Linear(2*d+5,d),nn.SiLU(),nn.LayerNorm(d))
        self.independent_curvature=nn.Linear(2*d,1)
        self.tetra_message=nn.Sequential(nn.Linear(d+4,d),nn.SiLU(),nn.LayerNorm(d))
        self.face_tetra_fuse=nn.Sequential(nn.Linear(2*d,d),nn.Tanh(),nn.Linear(d,d))
        self.edge_out=nn.Sequential(nn.Linear(2*d,d),nn.Tanh(),nn.Linear(d,d))
        self.triangle_out=nn.Sequential(nn.Linear(2*d,d),nn.Tanh(),nn.Linear(d,d))
        # Trainable sparse 256-specialist meta-mind. Each sector has a distinct
        # low-rank operator, not merely a static value vector. Its query is
        # conditioned on the actual tetrahedral consistency summary.
        self.tetra_to_meta=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
        rank=4
        self.meta_up=nn.Parameter(torch.randn(256,d,rank)*0.02)
        self.meta_down=nn.Parameter(torch.randn(256,rank,d)*0.02)
        self.meta_bias=nn.Parameter(torch.zeros(256,d))
        # Exactly closed at migration: bitwise original R7.23 / R7.22 parity.
        self.gate=nn.Parameter(torch.zeros(()))
        self.register_buffer('edges',torch.tensor(LOCAL_E,dtype=torch.long),persistent=False)
        self.register_buffer('faces',torch.tensor(LOCAL_F,dtype=torch.long),persistent=False)
        self.register_buffer('face_edges',torch.tensor(FACE_TO_EDGES,dtype=torch.long),persistent=False)
        self.register_buffer('tetrahedra',torch.tensor(LOCAL_T,dtype=torch.long),persistent=False)
        self.register_buffer('tetra_faces',torch.tensor(TET_TO_FACES,dtype=torch.long),persistent=False)
        self.register_buffer('tetra_signs',torch.tensor(TET_SIGNS,dtype=torch.float32),persistent=False)

    def forward(self,node,vertex_ids,pair_ix,tri_ids,tri_code,trunk,trace=False):
        B,V,D=node.shape
        if V!=ACTIVE or D!=self.width or pair_ix.shape!=(B,28) or tri_ids.shape!=(B,56):
            raise ValueError('R7.24 expects 8 points, 28 pairs, 56 triangles')
        e=self.edges
        ni=node[:,e[:,0]];nj=node[:,e[:,1]]
        vi=vertex_ids[:,e[:,0]];vj=vertex_ids[:,e[:,1]]
        sign=torch.where(vi<vj,1.,-1.).unsqueeze(-1)
        old_P=self.mirror_P(F.embedding(pair_ix,trunk.P.detach()))
        old_M=self.mirror_M(F.embedding(pair_ix,trunk.M.detach()))
        # Canonical high-minus-low difference is invariant under local ordering.
        canonical_diff=sign*(nj-ni)
        even=self.pair_even(torch.cat([(ni+nj)*.5,old_P],dim=-1))
        odd_base=self.pair_odd(torch.cat([canonical_diff,old_M],dim=-1))
        odd=sign*odd_base
        # A truly oriented 1-cochain: swapping endpoints reverses theta.
        connection=math.pi*torch.tanh(self.edge_connection(torch.cat([even,odd_base],dim=-1)))
        theta=sign.squeeze(-1)*connection.squeeze(-1)
        fe=self.face_edges
        phi_ab=theta[:,fe[:,0]]
        phi_ac=theta[:,fe[:,1]]
        phi_bc=theta[:,fe[:,2]]
        # Connection holonomy: d(theta) on each oriented triangle.
        holonomy=phi_ab+phi_bc-phi_ac
        orient=triangle_parity(vertex_ids[:,self.faces[:,0]],
                               vertex_ids[:,self.faces[:,1]],
                               vertex_ids[:,self.faces[:,2]])
        face_plus=even[:,fe].mean(dim=2)
        # Orientation-corrected canonical mirror part.
        face_odd_basis=odd_base[:,fe].mean(dim=2)
        face_minus=face_odd_basis*orient.unsqueeze(-1)
        # Spherical solid-angle triangle curvature from *actual point geometry*.
        xyz=F.normalize(self.point_sphere(node),dim=-1,eps=1e-5)
        a=xyz[:,self.faces[:,0]]
        b=xyz[:,self.faces[:,1]]
        c=xyz[:,self.faces[:,2]]
        det=(a*torch.cross(b,c,dim=-1)).sum(dim=-1)
        denominator=1+(a*b).sum(-1)+(b*c).sum(-1)+(c*a).sum(-1)
        area=2*torch.atan2(det,denominator)
        # All 2-forms reverse sign under odd permutation of face vertices.
        learned_canonical=math.pi*torch.tanh(
            self.independent_curvature(torch.cat([face_plus,tri_code],-1))).squeeze(-1)
        learned_form=orient*learned_canonical
        residual=learned_form-holonomy
        feat=torch.stack([torch.sin(holonomy)*orient,torch.cos(holonomy),
                          torch.sin(area)*orient,torch.cos(area),
                          torch.sin(residual)*orient],dim=-1)
        up_even=self.face_even(torch.cat([face_plus,tri_code,feat],-1))
        # The parity-corrected basis is even; multiply by orientation for S-.
        odd_feat=torch.stack([torch.sin(holonomy)*orient,torch.cos(holonomy),
                              torch.sin(area)*orient,torch.cos(area),
                              torch.sin(residual)*orient],dim=-1)
        up_odd=orient.unsqueeze(-1)*self.face_odd(
            torch.cat([face_odd_basis,tri_code,odd_feat],dim=-1))
        # Tetrahedral closure: d(K) is nontrivial for an independent learned
        # 2-form. Exact connection part has d^2(theta)=0 by construction.
        tf=self.tetra_faces
        alternating=self.tetra_signs.view(1,1,4)
        closure=(learned_form[:,tf]*alternating).sum(-1)
        # Geometric spherical area of four faces (may sum to a multiple of 4pi).
        sphere_boundary=(area[:,tf]*alternating).sum(-1)
        tet_mean=up_even[:,tf].mean(2)
        tet_features=torch.stack([torch.sin(closure),torch.cos(closure),
                                  torch.sin(sphere_boundary),torch.cos(sphere_boundary)],-1)
        tetra_state=self.tetra_message(torch.cat([tet_mean,tet_features],-1))
        # Every face is covered by C(8-3,4-3)=5 active tetrahedra.
        face_mail=torch.zeros_like(up_even)
        face_mail.index_add_(1,tf.flatten(),tetra_state.unsqueeze(2).expand(-1,-1,4,-1).reshape(B,-1,D))
        face_mail=face_mail/5.
        face_delta=self.triangle_out(torch.cat([up_even+self.face_tetra_fuse(
            torch.cat([up_even,face_mail],dim=-1)),up_odd],-1))
        edge_delta=self.edge_out(torch.cat([even,odd],-1))
        # Reg 0 iff learned 2-form is approximately a closed discrete field.
        closure_loss=(1.-torch.cos(closure)).mean()
        stats={'closure_loss':closure_loss,
               'connection_holonomy_abs':holonomy.abs().mean(),
               'spherical_angle_abs':area.abs().mean(),
               'tetra_residual_abs':torch.atan2(torch.sin(closure),torch.cos(closure)).abs().mean(),
               'even_edge':even,'odd_edge':odd,
               'face_even':up_even,'face_odd':up_odd,
               'holonomy':holonomy,'solid_angle':area,
               'face_curvature':learned_form,
               'tetra_addresses':rank4(*[vertex_ids[:,self.tetrahedra[:,j]] for j in range(4)]),
               'tetra_boundary':closure,
               'tetra_summary':tetra_state.mean(1),
               'edge_connection':theta,
               'face_residual':residual,
               'geo_gate':torch.tanh(self.gate)}
        if not trace:
            # Avoid retaining very large tensors in downstream reports.
            stats={k:v for k,v in stats.items() if k in
                   ('closure_loss','connection_holonomy_abs','spherical_angle_abs',
                    'tetra_residual_abs','geo_gate','tetra_summary')}
        return edge_delta,face_delta,stats

    def specialize(self,apex,selected_ids,selected_weights,geo_trace):
        """Tetra-aware rank-4 processing by four of 256 apex specialists."""
        q=apex+self.tetra_to_meta(geo_trace['tetra_summary'])
        up=self.meta_up[selected_ids]      # B x 4 x D x r
        down=self.meta_down[selected_ids]  # B x 4 x r x D
        codes=F.silu(torch.einsum('bd,bkdr->bkr',q,up))
        output=torch.einsum('bkr,bkrd->bkd',codes,down)+self.meta_bias[selected_ids]
        return (selected_weights.unsqueeze(-1)*output).sum(1)


def geometry_identity_check(original,extended):
    """All original checkpoint tensors must be exactly the same."""
    a=original.state_dict();b=extended.state_dict()
    assert all(k in b and torch.equal(v,b[k]) for k,v in a.items())