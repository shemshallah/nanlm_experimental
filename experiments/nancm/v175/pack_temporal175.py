#!/usr/bin/env python3
"""N175NOW1 tensor exact native codec, 48-word horizon, 12k whole-word softmax."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
PARENT_ORDER=['embed','out','bias','geo','projection','gain','gpos','gneg',
 'gru_pos.weight_ih_l0','gru_pos.weight_hh_l0','gru_pos.bias_ih_l0','gru_pos.bias_hh_l0',
 'gru_neg.weight_ih_l0','gru_neg.weight_hh_l0','gru_neg.bias_ih_l0','gru_neg.bias_hh_l0',
 'signed_norm.weight','signed_norm.bias','to_lex.0.weight','to_lex.0.bias',
 'to_lex.2.weight','to_lex.2.bias','trust','signed_scale']
ORDER=PARENT_ORDER+['copy_log_strength','copy_bigram','copy_trigram','copy_log_tau']
ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',map_location='cpu',weights_only=False)
state=ck['model_state_dict'];names=ck['word_order'];fp=ck['fingerprints'];assert len(names)==12000 and len(set(names))==12000
assert set(state)==set(ORDER),(set(state)-set(ORDER),set(ORDER)-set(state))
assert len(ck['optimizer_state_dict']['param_groups'])==4 and len(ck['optimizer_state_dict']['state'])>=19
payload=bytearray(b'N175NOW1')
payload.extend(struct.pack('<3Q4I',*fp,12000,48,32,64))
for w in names:
 b=w.encode('ascii');assert 0<len(b)<=40 and b.isalpha() and b.islower()
 payload.append(len(b));payload.extend(b)
payload.extend(struct.pack('<I',len(ORDER)))
shapes={}
for name in ORDER:
 t=state[name].detach().cpu().numpy().astype('<f4',copy=False);assert np.isfinite(t).all()
 k=name.encode('ascii');assert len(k)<256
 payload.append(len(k));payload.extend(k);payload.append(t.ndim)
 for d in t.shape:payload.extend(struct.pack('<I',d))
 payload.extend(t.tobytes(order='C'))
 shapes[name]=list(t.shape)
path=ROOT/'checkpoints/v175_native_episodic_48.bin';path.write_bytes(payload)
out={'format':'N175NOW1','vocab':12000,'history_words':48,'lexical_dim':32,'signed_rails':'2x64',
 'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'shapes':shapes,'training_step':ck['training_step'],
 'parents':'v174 dual GRU temporal + inherited v173 12k word and 128+128 S+/S- conceptual lattice',
 'optimizer_groups':len(ck['optimizer_state_dict']['param_groups']),'optimizer_states':len(ck['optimizer_state_dict']['state']),
 'all_inherited_optimizer_history_preserved':True}
(ROOT/'results/v175_native_pack.json').write_text(json.dumps(out,indent=2));print('PACK',out['bytes'],out['sha256'])