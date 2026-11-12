#!/usr/bin/env python3
"""Verify R18 original model/optimizer history without redistributing parent checkpoint.

The reference digests were computed from exact R16 and R18 trees and checked
for equality at the R18 release. This validates preserved subtrees, not a
cryptographically signed chain of publication or an independent audit.
"""
import argparse,hashlib,json,pathlib,struct,sys
import numpy as np,torch
FIELDS=('model','stream_bridge','r12_residual','stream_optimizer','stream_rng',
        'torch_rng','numpy_rng','python_rng','r12_updates','r10_updates','r11_updates')

def tree_hash(v):
    h=hashlib.sha256()
    def inner(obj):
        if isinstance(obj,torch.Tensor):
            c=obj.detach().cpu().contiguous()
            h.update(b'TENSOR\0'+str(c.dtype).encode()+b'|'+str(tuple(c.shape)).encode()+b'|')
            h.update(c.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj,np.ndarray):
            h.update(b'NDARRAY\0'+str(obj.dtype).encode()+b'|'+str(obj.shape).encode()+b'|');h.update(obj.tobytes())
        elif isinstance(obj,dict):
            h.update(b'DICT\0'+str(len(obj)).encode())
            for k in sorted(obj,key=lambda x:(type(x).__name__,str(x))):inner(k);inner(obj[k])
        elif isinstance(obj,(list,tuple)):
            h.update(('TUPLE\0' if isinstance(obj,tuple) else 'LIST\0').encode()+str(len(obj)).encode())
            for item in obj:inner(item)
        elif isinstance(obj,(float,np.floating)):
            h.update(b'FLOAT\0'+struct.pack('!d',float(obj)))
        elif isinstance(obj,(int,np.integer)) and not isinstance(obj,(bool,np.bool_)):
            h.update(b'INT\0'+str(int(obj)).encode())
        elif isinstance(obj,(bool,np.bool_)):
            h.update(b'BOOL\0'+(b'1' if obj else b'0'))
        elif obj is None:h.update(b'NONE\0')
        elif isinstance(obj,str):h.update(b'STRING\0'+obj.encode('utf-8'))
        else:raise TypeError(f'Unsupported checkpoint field type {type(obj)!r}')
    inner(v);return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--capture',action='store_true');p.add_argument('--parent-zip',default=None)
    args=p.parse_args();root=pathlib.Path(args.root)
    active=root/'checkpoint/r12_active.pal12'
    saved=torch.load(active,map_location='cpu',weights_only=False)
    current={k:tree_hash(saved[k]) for k in FIELDS}
    report_path=root/'reports/r18_parent_equivalence.json'
    if args.capture:
        import zipfile,io
        assert args.parent_zip is not None
        with zipfile.ZipFile(args.parent_zip).open('checkpoint/r12_active.pal12') as f:
            parent=torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=False)
        predecessor={k:tree_hash(parent[k]) for k in FIELDS}
        assert current==predecessor,(current,predecessor)
        report={'original_parent_checkpoint_sha256':'dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f',
                'current_checkpoint_sha256':hashlib.sha256(active.read_bytes()).hexdigest(),
                'checkpoint_tree_digest_algorithm':'source/r18_verify_lineage.py tree_hash',
                'old_model_and_optimizer_equal_original':True,'digest_fields':current,
                'note':'Only inherited fields compared; R18 auxiliary weights and moments are additional in the same checkpoint'}
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    else:
        report=json.loads(report_path.read_text())
        assert current==report['digest_fields'],('lineage drift detected',current,report['digest_fields'])
    print(json.dumps({'passed':len(FIELDS),'fields':FIELDS,'checkpoint':str(active)},indent=2))
if __name__=='__main__':main()
