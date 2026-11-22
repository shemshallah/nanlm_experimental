#!/usr/bin/env python3
"""NaNCM v169 experimental 256-way causal byte predictor.

Symbiotically composes:
  * inherited native v167/v168 128+128 S+/S- signed concept/ANN+lexical head,
  * a word-case/boundary/punctuation probability bridge fitted on TRAIN books,
  * a TRAIN-book causal 256-byte 0/1/2/3-gram distribution.

Every next256(past) returns a normalized 256-byte distribution.  The byte
mixture weights .5 within ASCII words and .1 outside were selected on the
previously explored DEV windows; these weights were frozen before reused TEST.
No target/future bytes are read, and no predecessor weights are changed.
"""
import ctypes,hashlib,json,math,os,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts'),str(Path(__file__).parent)]
from train_gate163 import setup
from test_native_speller156 import load,MODEL,F
from byte_bridge169 import Bridge,train_bridge
from byte_baseline169 import train


def create_ngram_sidecar(out):
 xs=train()
 data={'format':'N169BYTE3','trained_from':'NaNROUTER TRAIN-book external_v163 calibration windows only',
       'orders':[{key.hex():{str(b):int(c) for b,c in v.items()} for key,v in d.items()} for d in xs]}
 out.write_text(json.dumps(data,separators=(',',':')))
 return data

class ByteNgram:
 def __init__(self,pack):
  if pack.get('format')!='N169BYTE3' or len(pack['orders'])!=4:raise ValueError('invalid byte ngram sidecar')
  self.models=[{bytes.fromhex(k):{int(b):int(n) for b,n in d.items()} for k,d in order.items()} for order in pack['orders']]
  self.global_dist={i:(self.models[0][b''].get(i,0)+.1)/(sum(self.models[0][b''].values())+25.6) for i in range(256)}
 def probs(self,past):
  prior=self.global_dist.copy()
  for k in range(1,4):
   if len(past)<k:break
   row=self.models[k].get(past[-k:])
   if row:
    t=sum(row.values());strength=(10.,30.,80.)[k-1]
    prior={i:(row.get(i,0)+strength*prior[i])/(t+strength) for i in range(256)}
  return [prior[i] for i in range(256)]

class NaNCM169BytePredictor:
 """Context-only inference. Use as a streaming byte predictor; call close()."""
 def __init__(self,root=R,eta_inside=.5,eta_outside=.1):
  self.root=Path(root)
  self.eta_inside=float(eta_inside);self.eta_outside=float(eta_outside)
  for v in (self.eta_inside,self.eta_outside):
   if not math.isfinite(v) or not 0<=v<=1:raise ValueError('mixture strength must be in [0,1]')
  ngram_path=self.root/'checkpoints/v169_byte_ngram.json'
  bridge_path=self.root/'checkpoints/v169_byte_bridge.json'
  if not ngram_path.exists() or not bridge_path.exists():raise FileNotFoundError('TRAIN-only byte sidecars missing')
  # Release-specific immutability lock: byte sidecars cannot silently diverge
  # from the audited TRAIN-only statistics or inherited trained cloud.
  expected={
   'checkpoints/v166_lexical_birth.bin':'e6e2305c045680ade98eae32ea903b7bc43314da529ffe668525201219379bdb',
   'checkpoints/v168_signed_ann.bin':'c542c09c9e2d2ea846884f2b5f2a56f5970c8085509134378bfc47f5a6d806f5',
   'checkpoints/v169_byte_bridge.json':'b3816657affcc672b1a76e8a8591a8b7a86ae2b328336b16ce4fb46d86505d80',
   'checkpoints/v169_byte_ngram.json':'2d91307f8debd44476218dd9ffbb64c4dc9fef321b2ee97e28c28fcac76709c7'}
  for rel,digest in expected.items():
   if hashlib.sha256((self.root/rel).read_bytes()).hexdigest()!=digest:
    raise ValueError('v169 immutable checkpoint/byte-bridge fingerprint mismatch: '+rel)
  bridge_pack=json.loads(bridge_path.read_text())
  if bridge_pack.get('format')!='N169BYTE1':raise ValueError('unsupported byte bridge')
  self.ngram=ByteNgram(json.loads(ngram_path.read_text()))
  self.bridge=Bridge(bridge_pack)
  setup(core=False,domain=True,gate=None)
  os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(self.root/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(self.root/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(self.root/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
  self.lib=load(self.root/'bin/libwordfield169.so')
  self.handle=self.lib.wf148_open(*MODEL)
  if not self.handle:raise RuntimeError('v169 native weights/index invalid')
  self.anchor=None
 def close(self):
  if self.handle:self.lib.wf148_close(self.handle);self.handle=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def next256(self,past):
  """Return P(next_byte=0..255 | past); the caller provides only past bytes."""
  if not self.handle:raise RuntimeError('predictor closed')
  if not isinstance(past,bytes):raise TypeError('past must be bytes')
  m=re.search(rb'[A-Za-z]+\Z',past)
  prefix=m.group().lower().decode('ascii') if m else ''
  if len(prefix)>63:prefix=''  # fail toward boundary mode for extreme words
  anchor=past[:-len(prefix)] if prefix else past
  if anchor!=self.anchor:
   status=self.lib.wf148_begin(self.handle,ctypes.c_char_p(anchor),len(anchor),F(.25))
   if status<0:raise RuntimeError(f'native begin failed {status}; anchor={anchor[-100:]!r}, prefix={prefix!r}, pasttail={past[-100:]!r}, pastlen={len(past)}')
   self.anchor=anchor
  o=(F*27)();e=F()
  b=prefix.encode()
  if self.lib.wf148_next(self.handle,b,len(b),o,ctypes.byref(e))!=0:raise RuntimeError('native next failed')
  symbols=[float(v)+e.value/27. for v in o]
  from_words=self.bridge.dist(past,prefix,symbols)
  from_bytes=self.ngram.probs(past)
  eta=self.eta_inside if prefix else self.eta_outside
  p=[(1-eta)*a+eta*b for a,b in zip(from_bytes,from_words)]
  total=sum(p)
  if not math.isfinite(total) or abs(total-1)>0.0005:raise RuntimeError('256-way normalization failure')
  return [x/total for x in p]


def main():
 ng=R/'checkpoints/v169_byte_ngram.json';bg=R/'checkpoints/v169_byte_bridge.json'
 if not ng.exists():create_ngram_sidecar(ng)
 if not bg.exists():bg.write_text(json.dumps(train_bridge()))
 with NaNCM169BytePredictor() as predictor:
  for phrase in (b'The night was ',b'signed cloud ',b'There were '):
   probs=predictor.next256(phrase)
   ranked=sorted(range(256),key=lambda j:-probs[j])[:12]
   print('PRED',repr(phrase), [(chr(c) if 32<=c<127 else f'0x{c:02x}',round(probs[c],5)) for c in ranked])
 info={'architecture':'v169 native S+/S- ANN/lexical word generator + byte ngram via frozen 256-way mixture',
       'model_weight_sha256':hashlib.sha256((R/'checkpoints/v166_lexical_birth.bin').read_bytes()).hexdigest(),
       'signed_ann_sha256':hashlib.sha256((R/'checkpoints/v168_signed_ann.bin').read_bytes()).hexdigest(),
       'bridge_sha256':hashlib.sha256(bg.read_bytes()).hexdigest(),
       'ngram_sha256':hashlib.sha256(ng.read_bytes()).hexdigest(),
       'weights':{'within_word':.5,'outside_word':.1},'status':'EXPERIMENTAL_BOOK_DEV_SELECTED'}
 (R/'results/v169_streaming_model_manifest.json').write_text(json.dumps(info,indent=2))
 print('FINAL',json.dumps(info))
if __name__=='__main__':main()