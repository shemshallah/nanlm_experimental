#!/usr/bin/env python3
"""FA1 R7.24: oriented C(256,1/2/3/4) geometric operators inside one Diamond.

- S+ invariant to orientation; S- reverses with orientation.
- An oriented 1-cochain on the inherited pair mirror supplies face holonomy.
- Independent exact-address triangle S+/S- codes permit non-flat face residues.
- Tetrahedron closure uses the alternating four-face boundary. All C(256,4)
  addresses are exact; only C(8,4)=70 are activated per target.
- Every geometric route flows into the existing face->edge->point descent.
No claim is made that these latent coordinates describe physical-space geometry.
"""
from __future__ import annotations
import itertools, math
import torch
from torch import nn
from torch.nn import functional as F
from fa1_r723_diamond import CombinatorialDiamond, TRIPLES, ACTIVE, TRIPLES_LOCAL

TETRAHEDRA=math.comb(256,4)
TETRA_LOCAL=list(itertools.combinations(range(ACTIVE),4))
_TRI_LOOKUP={tri:i for i,tri in enumerate(TRIPLES_LOCAL)}
TETRA_FACES=torch.tensor([[ _TRI_LOOKUP[(j,k,l)], _TRI_LOOKUP[(i,k,l)],
                             _TRI_LOOKUP[(i,j,l)], _TRI_LOOKUP[(i,j,k)]]
                         for i,j,k,l in TETRA_LOCAL],dtype=torch.long)
FACE_SIGNS=torch.tensor([1.,-1.,1.,-1.])


def tetra_rank(a,b,c,d):
    """Unique colexicographic IDs over the full 174,792,640 tetrahedra."""
    x=torch.sort(torch.stack([a,b,c,d],dim=-1),dim=-1).values
    i,j,k,l=x.unbind(-1)
    if bool(((i==j)|(j==k)|(k==l)).any()):
        raise ValueError('tetra vertices must be distinct')
    return l*(l-1)*(l-2)*(l-3)//24+k*(k-1)*(k-2)//6+j*(j-1)//2+i


def orientation_sign(a,b,c):
    """Parity of oriented (a,b,c) relative to ascending vertex IDs."""
    inversions=(a>b).long()+(a>c).long()+(b>c).long()
    return (1-2*(inversions%2)).to(torch.float32)


class OrientedSimplicialGeometry(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        d=width
        self.point_coords=nn.Linear(96,3,bias=False)
        # Independent even/odd transforms preserve the original R7 P/M
        # tensors; learned geometry is added rather than replacing them.
        self.edge_plus=nn.Linear(64,d,bias=False)
        self.edge_minus=nn.Linear(64,d,bias=False)
        self.edge_phase=nn.Linear(64,1,bias=False)
        self.edge_net=nn.Sequential(nn.Linear(d*2+2,d),nn.GELU(),nn.Linear(d,d))
        self.face_plus=nn.Embedding(TRIPLES,1)
        self.face_minus=nn.Embedding(TRIPLES,1)
        nn.init.normal_(self.face_plus.weight,std=0.002)
        nn.init.normal_(self.face_minus.weight,std=0.002)
        self.face_net=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.Linear(d,d))
        self.tetra_net=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.Linear(d,d))
        self.edge_gain=nn.Parameter(torch.zeros(()))
        self.face_gain=nn.Parameter(torch.zeros(()))
        self.tetra_gain=nn.Parameter(torch.zeros(()))
        self.register_buffer('edge_local',torch.tensor(list(itertools.combinations(range(ACTIVE),2)),dtype=torch.long),persistent=False)
        self.register_buffer('tri_local',torch.tensor(TRIPLES_LOCAL,dtype=torch.long),persistent=False)
        self.register_buffer('tetra_local',torch.tensor(TETRA_LOCAL,dtype=torch.long),persistent=False)
        self.register_buffer('tetra_faces',TETRA_FACES,persistent=False)
        self.register_buffer('face_signs',FACE_SIGNS,persistent=False)
        self.register_buffer('triangle_edges',torch.tensor([[ self._edge_idx(i,j),self._edge_idx(i,k),self._edge_idx(j,k)] for i,j,k in TRIPLES_LOCAL],dtype=torch.long),persistent=False)

    @staticmethod
    def _edge_idx(i,j):
        return list(itertools.combinations(range(ACTIVE),2)).index((i,j))

    def signed_edge_channels(self,p,m,a,b):
        sign=torch.where(a<b,1.,-1.).unsqueeze(-1)
        return self.edge_plus(p),sign*self.edge_minus(m),sign*torch.tanh(self.edge_phase(m))

    def pairs(self,trunk,vertex_ids,pair_ix,ids_i,ids_j,pair,q):
        # Reusing original R7 latent coordinates makes this jointly learnable
        # with the existing 256-point geometry, without changing its weights.
        xyz=self.point_coords(F.embedding(vertex_ids.long(),trunk.H.detach()))
        xyz=F.normalize(xyz,dim=-1,eps=1e-6)
        p=F.embedding(pair_ix,trunk.P.detach())
        m=F.embedding(pair_ix,trunk.M.detach())
        plus,minus,phase=self.signed_edge_channels(p,m,ids_i,ids_j)
        ex=self.edge_local
        length=torch.linalg.vector_norm(xyz[:,ex[:,0]]-xyz[:,ex[:,1]],dim=-1,keepdim=True)
        feat=torch.cat([plus,minus,length,torch.cos(phase)],dim=-1)
        geo=self.edge_net(feat)
        pair=pair+torch.tanh(self.edge_gain)*geo
        return pair,{'xyz':xyz,'edge_phase':phase.squeeze(-1), 'edge_length':length.squeeze(-1),
                     'metrics':{'sminus_edge_abs':minus.detach().abs().mean()}}

    def signed_triangle_channels(self,triangle_ids,vertices,edge_curl):
        """Grade-3 S+ is orientation-even, S- is orientation-odd."""
        parity=orientation_sign(vertices[:,:,0],vertices[:,:,1],vertices[:,:,2])
        plus=self.face_plus(triangle_ids).squeeze(-1)+torch.cos(edge_curl)
        minus=parity*self.face_minus(triangle_ids).squeeze(-1)+torch.sin(edge_curl)
        return plus,minus

    def triangles_up(self,vertex_ids,tri_ids,tri,state):
        t=self.tri_local
        xyz=state['xyz']
        a,b,c=(xyz[:,t[:,i]] for i in (0,1,2))
        ab=b-a;ac=c-a;bc=c-b
        lab=torch.linalg.vector_norm(ab,dim=-1).clamp_min(1e-6)
        lac=torch.linalg.vector_norm(ac,dim=-1).clamp_min(1e-6)
        lbc=torch.linalg.vector_norm(bc,dim=-1).clamp_min(1e-6)
        area=0.5*torch.linalg.vector_norm(torch.cross(ab,ac,dim=-1),dim=-1)
        # Triangle S+ shape uses sorted side lengths, not one corner label.
        lengths=torch.stack([lab,lac,lbc],dim=-1).sort(dim=-1).values
        side0,side1,side2=lengths.unbind(-1)
        cosine=((side0.square()+side1.square()-side2.square())/
                (2*side0*side1).clamp_min(1e-6)).clamp(-1.,1.)
        te=self.triangle_edges
        # Exterior derivative dA: A_ij + A_jk - A_ik.
        curl=(state['edge_phase'][:,te[:,0]]+
              state['edge_phase'][:,te[:,2]]-
              state['edge_phase'][:,te[:,1]])
        vs=vertex_ids[:,t]
        even,curv=self.signed_triangle_channels(tri_ids,vs,curl)
        face_even=self.face_plus(tri_ids).squeeze(-1)
        face_odd=curv-torch.sin(curl)
        features=torch.stack([side0,side1,side2,area,cosine,
                              torch.sin(curl),torch.cos(curl),
                              face_even,face_odd],dim=-1)
        tri=tri+torch.tanh(self.face_gain)*self.face_net(features)
        ti=self.tetra_local
        id4=vertex_ids[:,ti]
        tetra_ids=tetra_rank(id4[:,:,0],id4[:,:,1],id4[:,:,2],id4[:,:,3])
        # A tetrahedron has four oriented faces: [bcd] - [acd] + [abd] - [abc].
        faces=self.tetra_faces
        signed_curv=curv[:,faces]
        closure=(signed_curv*self.face_signs.view(1,1,4)).sum(-1)
        x0,x1,x2,x3=(xyz[:,ti[:,i]] for i in range(4))
        signed_volume=(torch.cross(x1-x0,x2-x0,dim=-1)*(x3-x0)).sum(-1)/6.
        # Angle/volume and residual face closure give the sparse tetra critic
        # meaningful non-flat signals; d(dA)=0 is tested separately.
        tetra_feat=torch.stack([closure,torch.sin(closure),torch.cos(closure),
                                signed_volume,signed_volume.square(),
                                signed_curv.abs().mean(-1),
                                signed_curv.square().mean(-1)],dim=-1)
        tetra_vec=self.tetra_net(tetra_feat)
        triangle_mail=torch.zeros_like(tri)
        triangle_n=tri.new_zeros(tri.shape[:2])
        for j in range(4):
            triangle_mail.index_add_(1,faces[:,j],tetra_vec)
            triangle_n.index_add_(1,faces[:,j],torch.ones_like(closure))
        tri=tri+torch.tanh(self.tetra_gain)*triangle_mail/triangle_n.unsqueeze(-1).clamp_min(1)
        state['metrics'].update({'triangle_curl_abs':curl.detach().abs().mean(),
             'tetra_closure_abs':closure.detach().abs().mean(),
             'tetra_signed_volume_abs':signed_volume.detach().abs().mean(),
             'tetra_unique_ids':tetra_ids.detach().unique().numel()*torch.ones((),device=tri.device),
             'edge_gain':self.edge_gain.detach().clone(),
             'face_gain':self.face_gain.detach().clone(),
             'tetra_gain':self.tetra_gain.detach().clone()})
        state['tetra_ids']=tetra_ids
        state['face_curvature']=curv
        state['closure']=closure
        return tri,state


class GeometricDiamond(CombinatorialDiamond):
    """Same R7 Diamond, with genuine oriented rank-2/3 channels and sparse rank-4 critic."""
    def __init__(self,width=32):
        super().__init__(width=width)
        self.geometry=OrientedSimplicialGeometry(width=width)