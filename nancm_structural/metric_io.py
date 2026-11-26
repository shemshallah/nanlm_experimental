"""Atomic JSON metric snapshots; NumPy values remain native JSON numbers."""
import json
from pathlib import Path
import numpy as np

def _native(obj):
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,np.ndarray):return obj.tolist()
    raise TypeError(f'Unsupported metric type: {type(obj).__name__}')

def write_metrics(path,metrics,stage='final'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(metrics,report_stage=stage)
    data=json.dumps(payload,default=_native,allow_nan=False,indent=2)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(data+'\n');tmp.replace(path)
    print('FINAL' if stage=='final' else 'PARTIAL',json.dumps(payload,default=_native,allow_nan=False),flush=True)
