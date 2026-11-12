#!/usr/bin/env python3
"""Structural anti-leakage and semantic benchmark quality tests; CPU."""
import argparse,hashlib,json,pathlib,sys,random
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory,build_copy_queries
from r19_semantic_frontier import (RELATIONS,TRAIN_REL,UNSEEN_REL,make_examples,Pack,
     SemanticPairAdapter,LateInteractionAdapter,multi_hop_probe,long_stream_sparse_probe)
import sentencepiece as spm

def main(root):
    torch.set_num_threads(2)
    root=pathlib.Path(root)
    cp=root/'checkpoint/r12_active.pal12';prev=hashlib.sha256(cp.read_bytes()).hexdigest()
    saved=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    emb=saved['model']['memory.piece_emb.weight']
    mem=R18LongMemory(emb);mem.load_state_dict(saved['r18_memory'])
    checks={}
    # Query+candidate memory is strictly causal; target at t absent from keys.
    z=torch.arange(270,dtype=torch.long)%250+700
    z[5:7]=torch.tensor([4201,4202]);z[7]=4150
    z[202:204]=torch.tensor([4201,4202]);z[204]=4150
    q=build_copy_queries(z,mem=2048,gap=65)
    ix=(q['positions']==204).nonzero()[0,0]
    checks['lexical_source_before_query']=bool((q['values'][ix][q['mask'][ix]]==4150).any())
    z2=z.clone();z2[202:204]=torch.tensor([4203,4204])
    q2=build_copy_queries(z2,mem=2048,gap=65)
    checks['modified_prefix_is_not_semantic_match']=bool(q2 is None or not (q2['positions']==204).any())
    z3=z.clone();z3[205:]=torch.flip(z3[205:],dims=[0])
    q3=build_copy_queries(z3,mem=2048,gap=65)
    # Future suffix can't affect query 204.
    jj=(q3['positions']==204).nonzero()[0,0]
    checks['causal_suffix_invariant']=bool(torch.equal(q['keys'][ix],q3['keys'][jj]) and torch.equal(q['values'][ix],q3['values'][jj]))
    sp=spm.SentencePieceProcessor(model_file=str(root/'tokenizer/gutenberg_r7.model'))
    checks['tokenizer_ids_fit_embedding']=sp.get_piece_size()<=len(emb)
    tr=make_examples(200,19019,'train',split=0)
    dv=make_examples(100,19020,'paraphrase',split=1)
    ts=make_examples(100,19320,'paraphrase',split=3)
    checks['disjoint_nonce_entities']=not ({x['subject'] for x in tr}&{x['subject'] for x in dv+ts})
    checks['all_candidates_shuffled_labeled']=all(len(x['facts'])==8 and x['facts'][x['label']].find(x['value'])>=0 for x in tr+dv+ts)
    checks['train_relations_do_not_include_test_only']=bool(not (set(TRAIN_REL)&set(UNSEEN_REL)))
    dev=Pack(dv[:8],sp)
    net=LateInteractionAdapter(emb,mem);scores=net.score(dev.q,dev.k)
    checks['attention_scores_finite']=bool(scores.shape==(8,8) and torch.isfinite(scores).all())
    l=F.cross_entropy(scores,dev.labels);l.backward()
    checks['trainable_token_alignment']=bool(net.qtok.weight.grad is not None and net.qtok.weight.grad.abs().sum()>0 and net.ktok.weight.grad.abs().sum()>0)
    net.eval();dist=long_stream_sparse_probe(net,sp,seed=999,distances=(8192,32768,65536))
    checks['distance_comparison_uses_same_queries']=bool(len({round(d['retrieval_top1'],9) for d in dist})==1 and all(d['same_queries_facts_all_distances'] for d in dist))
    hops=multi_hop_probe(net,sp,seed=20000,n=16)
    checks['two_hop_has_semantic_distractors']=bool(hops['semantic_distractors_each_hop'] and hops['second_hop_uses_first_prediction'] and hops['question_decomposition_oracle_provided'])
    checks['unmodified_accepted_checkpoint']=hashlib.sha256(cp.read_bytes()).hexdigest()==prev
    assert all(checks.values()),checks
    out={'round':'R19','tests':checks,'passed':sum(checks.values()),'total':len(checks),'checkpoint_sha256':prev}
    (root/'reports/r19_structure_tests.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();main(a.root)
