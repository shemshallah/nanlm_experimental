#!/usr/bin/env python3
"""PAL R19 native R18 adversarial audit: no weight updates or new task claims.

Artificial controlled streams isolate memory access, NOT long-text comprehension.
The random filler prevents trivial two-piece coincidences; positions and values
are created before running the matcher. Actual R18 retrieve queries are causal.
"""
import argparse,hashlib,json,pathlib,random,sys,time
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import build_copy_queries,R18LongMemory


def audit(root, lengths=(2048,8192,32768,65536), seed=1919):
    torch.set_num_threads(2)
    ck=root/'checkpoint/r12_active.pal12'
    before=hashlib.sha256(ck.read_bytes()).hexdigest()
    saved=torch.load(ck,weights_only=False,map_location='cpu',mmap=True)
    emb=saved['model']['memory.piece_emb.weight']
    net=R18LongMemory(emb)
    net.load_state_dict(saved['r18_memory']);net.eval()
    out={}; rng=random.Random(seed)
    for distance in lengths:
        n=distance+120;src=12;dest=src+distance
        # All background ids drawn from [100,1500); sentinel keys >4000,
        # guaranteeing that query variants are unseen in background.
        g=torch.Generator().manual_seed(seed+distance)
        seq=torch.randint(100,1500,(n,),generator=g)
        seq[src-2:src]=torch.tensor([4101,4102]);seq[src]=4210
        seq[dest-2:dest]=torch.tensor([4101,4102]);seq[dest]=4210
        assert int(seq[dest-1])==4102
        outcome={}
        for typ in ('exact','substitute_both_key_tokens','substitute_one_key_token'):
            s=seq.clone()
            if typ=='substitute_both_key_tokens':s[dest-2:dest]=torch.tensor([4103,4104])
            if typ=='substitute_one_key_token':s[dest-2:dest]=torch.tensor([4105,4102])
            memory=max(2048,distance+1)
            t=time.perf_counter();q=build_copy_queries(s,mem=memory,gap=65);dt=time.perf_counter()-t
            match=bool(q is not None and (q['positions']==dest).any())
            src_found=False;value_found=False;rank=None
            if match:
                ix=int((q['positions']==dest).nonzero()[0,0]);key=q['keys'][ix];values=q['values'][ix];valid=q['mask'][ix]
                src_found=bool(((key==torch.tensor([4101,4102])).all(-1)&valid).any())
                value_found=bool(((values==4210)&valid).any())
                if value_found:
                    with torch.no_grad():
                        scores=net.scores(q['query'][ix],key)
                        scores=scores.masked_fill(~valid,-1e8)
                        rank=int((scores>scores[((values==4210)&valid).nonzero()[0,0]]).sum())+1
            outcome[typ]={'query_enabled_at_target':match,'source_key_available':src_found,
                          'source_value_available':value_found,'target_rank_among_candidates':rank,
                          'seconds_index_build':dt}
        # Negative-control test: 2048-window cannot access earlier source
        if distance>2048:
            q_short=build_copy_queries(seq,mem=2048,gap=65)
            outcome['2048_window']={'source_visible_at_target':bool(q_short is not None and (q_short['positions']==dest).any())}
        out[str(distance)]=outcome
    after=hashlib.sha256(ck.read_bytes()).hexdigest()
    assert before==after
    report={'round':'PAL R19 native adversarial memory audit','checkpoint_sha256_unchanged':before,
            'distance_in_subword_tokens':out,'evaluation_kind':'controlled sparse index and learned-ranker stress, NOT language modeling',
            'expected_limitation':'R18 exact two-piece guard rejects changed prefix tokens before learned ranking; token substitutions are controlled structural analogues, not demonstrated natural-language synonyms',
            'no_test_examples_in_training':True}
    (root/'reports/r19_native_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args()
    r=audit(pathlib.Path(a.root));print(json.dumps(r,indent=2))
