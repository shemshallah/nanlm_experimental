#!/usr/bin/env python3
"""Export frozen v172 signed-cloud word teacher to native read-only sidecar.
Optimizer remains in the full PyTorch checkpoint; no old model is overwritten.
"""
import struct,hashlib,json
from pathlib import Path
import numpy as np,torch
from train_conditional_cloud172 import ROOT,load_parent,hash166,VOC,DIM

names,C,O,B,spatial,gain,fp=load_parent()
ck=torch.load(ROOT/'checkpoints/v172_conditional_signed_cloud.pt',weights_only=False,map_location='cpu')
assert tuple(ck['fingerprints'])==tuple(fp) and ck['word_order']==names
s=ck['state_dict'];ctx=s['ctx.weight'].cpu().numpy().astype('<f4');out=s['out.weight'].cpu().numpy().astype('<f4')
bias=s['bias'].cpu().numpy().astype('<f4');W=s['cloud_projection.weight'].cpu().numpy().astype('<f4');geo=s['geo'].cpu().numpy().astype('<f4');gain=float(s['gain'])
assert ctx.shape==out.shape==(12000,32) and bias.shape==(12000,) and W.shape==(32,8) and geo.shape==(12000,8)
assert np.isfinite(ctx).all() and np.isfinite(out).all() and np.isfinite(bias).all() and np.isfinite(W).all() and np.isfinite(geo).all() and 0<=gain<=25
payload=bytearray(b'N172TCH1'+struct.pack('<3Q3I f',*fp,12000,32,8,gain))
payload.extend(W.tobytes())
for i,w in enumerate(names):
 data=w.encode('ascii');assert 0<len(data)<=20
 payload.append(len(data));payload.extend(data)
 payload.extend(ctx[i].tobytes());payload.extend(out[i].tobytes());payload.extend(bias[i].tobytes());payload.extend(geo[i].tobytes())
path=ROOT/'checkpoints/v172_native_signed_teacher.bin';path.write_bytes(payload)
info={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest(),'size':len(payload),'magic':'N172TCH1','vocab':12000,'dim':32,'signed_geom_dim':8,'step':ck['step'],'parent_fingerprints':list(fp),'optimizer_checkpoint':'checkpoints/v172_conditional_signed_cloud.pt'}
(ROOT/'results/v172_teacher_pack.json').write_text(json.dumps(info,indent=2))
print(info)