#!/usr/bin/env python3
"""Fingerprint-checked flat float32 native serialization for v174 dual-rail NOW."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
ORDER=['embed','out','bias','geo','projection','gain','gpos','gneg',
 'gru_pos.weight_ih_l0','gru_pos.weight_hh_l0','gru_pos.bias_ih_l0','gru_pos.bias_hh_l0',
 'gru_neg.weight_ih_l0','gru_neg.weight_hh_l0','gru_neg.bias_ih_l0','gru_neg.bias_hh_l0',
 'signed_norm.weight','signed_norm.bias','to_lex.0.weight','to_lex.0.bias',
 'to_lex.2.weight','to_lex.2.bias','trust','signed_scale']
ck=torch.load(ROOT/'checkpoints/v174_temporal_signed_now.pt',map_location='cpu',weights_only=False)
state=ck['model_state_dict'];names=ck['word_order'];fp=ck['fingerprints'];assert len(names)==12000
assert set(state)==set(ORDER),(set(state)-set(ORDER),set(ORDER)-set(state))
payload=bytearray(b'N174NOW1')
payload.extend(struct.pack('<3Q4I',*fp,12000,12,32,64))
for w in names:
 b=w.encode('ascii');assert 0<len(b)<256
 payload.append(len(b));payload.extend(b)
payload.extend(struct.pack('<I',len(ORDER)))
manifest={}
for name in ORDER:
 t=state[name].detach().numpy().astype('<f4',copy=False);assert np.isfinite(t).all()
 k=name.encode('ascii');assert len(k)<256
 payload.append(len(k));payload.extend(k);payload.append(t.ndim)
 for d in t.shape:payload.extend(struct.pack('<I',d))
 payload.extend(t.tobytes(order='C'))
 manifest[name]=list(t.shape)
path=ROOT/'checkpoints/v174_temporal_signed_now.bin';path.write_bytes(payload)
out={'format':'N174NOW1','vocab':12000,'seq':12,'lexical_dim':32,'rails':'2x64','cloud': '128 S+ + 128 S- inherited','sha256':hashlib.sha256(payload).hexdigest(),'size':len(payload),'shapes':manifest,'training_step':ck['training_step'],'parent_fingerprints':list(fp),'trained_checkpoint':'checkpoints/v174_temporal_signed_now.pt','inherited_optimizer_preserved':True}
(ROOT/'results/v174_native_pack.json').write_text(json.dumps(out,indent=2));print('PACKED',out['size'],out['sha256'])