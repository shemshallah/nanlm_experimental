#!/usr/bin/env python3
"""FA2.2 structural, causality, identity, gradient and checkpoint safety tests."""
import argparse,hashlib,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory
from r21_relevance_novelty import causal_proposal,RelevanceNoveltyGate

def main(root):
 torch.set_num_threads(2);torch.manual_seed(21022)
 ck=pathlib.Path(root)/'checkpoint/r12_active.pal12'
 original=hashlib.sha256(ck.read_bytes()).hexdigest()
 saved=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
 net=R18LongMemory(saved['model']['memory.piece_emb.weight'])
 net.load_state_dict(saved['r18_memory']);net.eval()
 v=int(net.inherited_piece_embeddings.shape[0]);n=320
 tok=torch.randint(60,min(550,v),(n,),dtype=torch.long)
 logits=torch.randn(n,v)
 delta,features,meta=causal_proposal(net,logits,tok,block=19)
 gate=RelevanceNoveltyGate();out,g=gate(logits,delta,features)
 checks={}
 checks['zero_initialized_preserves_all_logits_exact']=torch.equal(out,logits)
 checks['early_queries_disabled']=torch.equal(delta[:67],torch.zeros_like(delta[:67]))
 checks['short_context_logits_frozen']=torch.equal(out[:192],logits[:192])
 checks['nonexact_memory_operational']=meta['eligible']>=50 and bool(delta.abs().sum()>0)
 checks['source_positions_strictly_before_gap']=meta['max_source_position']<=n-1-net.min_gap
 checks['feature_finiteness']=bool(torch.isfinite(features).all())
 checks['feature_dimensions_match']=features.shape==(n,8)
 tok2=tok.clone();tok2[220:]=torch.randint(60,min(550,v),(n-220,))
 d2,f2,_=causal_proposal(net,logits,tok2,block=23)
 checks['future_targets_cannot_change_past_proposal']=torch.equal(delta[:220],d2[:220])
 checks['future_targets_cannot_change_past_features']=torch.equal(features[:220],f2[:220])
 # All nonlinear gate parameters are differentiable after amplitude becomes nonzero.
 with torch.no_grad():gate.amplitude.fill_(.2)
 pred,g=gate(logits,delta,features)
 loss=torch.nn.functional.cross_entropy(pred[210:],tok[210:]);loss.backward()
 checks['gain_gets_gradient']=gate.amplitude.grad is not None and bool(gate.amplitude.grad.abs()>0)
 checks['feature_head_gets_gradient']=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in gate.features.parameters())
 checks['nonempty_gate_modifies_logits']=bool((pred-logits).abs().sum()>0)
 checks['checkpoint_unchanged']=hashlib.sha256(ck.read_bytes()).hexdigest()==original
 result={'experiment':'FA2.2 R21 frozen-base adaptive relevance/novelty gate',
         'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'checks':checks,
         'checkpoint_sha256':original,'promotion':False,
         'note':'Causal identity, gradient and integrity only; not a semantic or held-out improvement result'}
 assert all(checks.values()),json.dumps(result,indent=2)
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');main(p.parse_args().root)
