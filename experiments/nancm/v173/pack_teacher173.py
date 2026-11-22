#!/usr/bin/env python3
"""Publish exact-normalized v173 signed cloud word head in inherited native format.
Includes fingerprint parity to 128+128 cloud and existing 12k lexical order.
"""
import json,hashlib,struct,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v172'))
from train_conditional_cloud172 import ROOT,load_parent
names,C,O,B,spatial,gain,fp=load_parent()
ck=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',weights_only=False,map_location='cpu')
assert ck['word_order']==names and tuple(ck['fingerprints'])==tuple(fp)
s=ck['state_dict'];ctx=s['ctx.weight'].numpy().astype('<f4');out=s['out.weight'].numpy().astype('<f4')
bias=s['bias'].numpy().astype('<f4');proj=s['cloud_projection.weight'].numpy().astype('<f4');geo=s['geo'].numpy().astype('<f4');gain=float(s['gain'])
assert ctx.shape==out.shape==(12000,32) and bias.shape==(12000,) and proj.shape==(32,8) and geo.shape==(12000,8)
assert all(np.isfinite(arr).all() for arr in (ctx,out,bias,proj,geo)) and 0<=gain<=25
payload=bytearray(b'N172TCH1'+struct.pack('<3Q3I f',*fp,12000,32,8,gain));payload.extend(proj.tobytes())
for i,w in enumerate(names):
 data=w.encode('ascii');assert 0<len(data)<=20
 payload.append(len(data));payload.extend(data);payload.extend(ctx[i].tobytes());payload.extend(out[i].tobytes());payload.extend(bias[i].tobytes());payload.extend(geo[i].tobytes())
p=ROOT/'checkpoints/v173_native_signed_teacher.bin';p.write_bytes(payload)
info={'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'codec':'N172TCH1','vocab':12000,'step':ck['step'],'retains_parent_fingerprints':list(fp),'optimizer':'checkpoints/v173_exact_ce_word.pt','generation':'exact CE continuation; native backward-compatible codec'}
(ROOT/'results/v173_teacher_pack.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))