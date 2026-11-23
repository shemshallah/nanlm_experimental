#!/usr/bin/env python3
"""Parity: full 12k-word native causal expert vs exact PyTorch v174."""
import os,ctypes,json,sys,math
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from train_temporal_signed174 import ROOT,load_parent,CloudNCE,load_sign_geometry,DualPolarTemporal,tokenize,D
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
torch.set_num_threads(2)
names,C,O,B,spatial,gain,fp=load_parent();vocab={w:i for i,w in enumerate(names)}
ck=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',map_location='cpu',weights_only=False)
teacher=CloudNCE(C,O,B,spatial,gain,names);teacher.load_state_dict(ck['state_dict'])
gp,gn=load_sign_geometry(names);model=DualPolarTemporal(teacher,names,gp,gn)
trained=torch.load(ROOT/'checkpoints/v174_temporal_signed_now.pt',map_location='cpu',weights_only=False)
assert trained['word_order']==names and tuple(trained['fingerprints'])==tuple(fp)
model.load_state_dict(trained['model_state_dict']);model.eval()
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield174.so'))
lib.wf174_now_open.argtypes=[ctypes.c_char_p];lib.wf174_now_open.restype=ctypes.c_void_p
lib.wf174_now_close.argtypes=[ctypes.c_void_p]
lib.wf174_now_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];lib.wf174_now_predict.restype=ctypes.c_int
lib.wf174_now_word_name.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t];lib.wf174_now_word_name.restype=ctypes.c_int
assert not lib.wf174_now_open(b'/not/a/real/sidecar')
h=lib.wf174_now_open(str(ROOT/'checkpoints/v174_temporal_signed_now.bin').encode());assert h
for idx in (0,1,321,987,11999):
 buf=ctypes.create_string_buffer(64);assert lib.wf174_now_word_name(h,idx,buf,64)==0;assert buf.value.decode()==names[idx]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import winread
rows=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()]
rows_dev=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
inputs=[r['history'].encode('latin1') for r in rows[::max(1,len(rows)//(40 if args.full else 20))]][:40 if args.full else 20]
inputs += [wins[int(r['book'])][0][:r['pos']] for r in rows_dev[::max(1,len(rows_dev)//(30 if args.full else 16))]][:30 if args.full else 16]
inputs.extend([b'',b'Once upon a time ',b'abc xyz man man world ',b'NO?'] )
maxdiff=0.;max_nll_diff=0.;top1_matches=0;top5_matches=0
arr=(ctypes.c_float*12000)()
try:
 for i,past in enumerate(inputs):
  ws=tokenize(past.decode('latin1'))
  ids=[vocab.get(w,-1) for w in ws[-D:]];ids=([-1]*(D-len(ids)))+ids
  with torch.no_grad():probs=torch.softmax(model(torch.tensor([ids],dtype=torch.long)),dim=-1).numpy()[0]
  assert lib.wf174_now_predict(h,ctypes.c_char_p(past),len(past),arr,12000)==0
  native=np.ctypeslib.as_array(arr).copy()
  assert abs(float(native.sum(dtype=np.float64))-1)<3e-5 and np.isfinite(native).all() and (native>=0).all()
  diff=float(np.max(np.abs(native-probs)));maxdiff=max(maxdiff,diff)
  if int(np.argmax(native))==int(np.argmax(probs)):top1_matches+=1
  if set(np.argsort(-native)[:5])==set(np.argsort(-probs)[:5]):top5_matches+=1
  if i<3:print('PARITY',i,'maxdiff',diff,'top1',names[int(np.argmax(native))],flush=True)
 assert maxdiff<2e-5,(maxdiff,top1_matches,len(inputs))
 assert top1_matches==len(inputs) and top5_matches==len(inputs)
 print('PASS native/PyTorch whole 12,000-word normalization and ranking parity',len(inputs),'histories max_abs_err',maxdiff,flush=True)
finally:lib.wf174_now_close(h)
(ROOT/'results/v174_native_parity.json').write_text(json.dumps({'status':'PASS','histories':len(inputs),'word_vocab':12000,'max_abs_prob_error':maxdiff,'top1_exact_matches':top1_matches,'top5_set_exact_matches':top5_matches,'native_sidecar':str('checkpoints/v174_temporal_signed_now.bin'),'frozen_parent':True},indent=2))