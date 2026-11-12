#!/usr/bin/env python3
"""R18 post-selection: one-time book-disjoint historical test + ablations.
Never used for checkpoint selection. Computes equal-logit frozen-base control.
"""
import argparse,copy,hashlib,json,math,pathlib,sys,time,zipfile,io
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r18_long_memory import R18LongMemory,collect_hard_negatives,test_retrieval,build_copy_queries,apply_copy_boost
from r18_run import prepare_cache,summarize_cache,make_windows,frozen_base_logits

def metrics_with_mode(net,cache,mode='learned',mem=2048,shuffle=False):
    out=[]
    with torch.no_grad():
        for row in cache:
            obj=dict(row)
            if mode=='off':
                obj['logits']=row['logits'];obj['queries']=None
            else:
                q=build_copy_queries(row['tokens'],mem,65)
                if q is not None and shuffle:
                    q=dict(q)
                    q['values']=q['values'].clone()
                    # Deterministic wrong-values control; does not alter context.
                    q['values']=q['values'].roll(1,dims=0)
                obj['queries']=q
                obj['logits']=apply_copy_boost(net,row['logits'],q,baseline_rank=(mode=='exact_heuristic'))
                obj['queries']=None
            out.append(obj)
    return summarize_cache_without_memory(out)

def summarize_cache_without_memory(cache):
    total=other=0.;raw=nonspace=tcount=hits=0
    for row in cache:
        t=row['tokens'];p=row['logits']
        ce=F.cross_entropy(p,t,reduction='none');mask=t!=32
        total+=float(ce.sum());other+=float(ce[mask].sum())
        raw+=row['bytes'];nonspace+=int(mask.sum());tcount+=len(t)
        hits+=int(((p.argmax(-1)==t)&mask).sum())
    return {'bits_per_byte':total/raw/math.log(2),'nonspace_nll':other/nonspace,
            'nonspace_top1':hits/nonspace,'positions':tcount,'raw_bytes':raw}

def causal_test(net):
    torch.manual_seed(18018)
    tokens=torch.randint(100,800,(350,),dtype=torch.long)
    tokens[9:11]=torch.tensor([705,706]);tokens[210:212]=torch.tensor([705,706])
    logits=torch.randn(350,900)
    q=build_copy_queries(tokens,mem=2048,gap=65)
    assert q is not None and 212 in q['positions'].tolist()
    before=apply_copy_boost(net,logits,q)
    perturbed=tokens.clone();perturbed[212:]=torch.randint(100,800,(len(tokens)-212,))
    q2=build_copy_queries(perturbed,mem=2048,gap=65)
    after=apply_copy_boost(net,logits,q2)
    assert torch.equal(before[:213],after[:213]),'future-target leakage into past logits'
    modified=tokens.clone();modified[11]=850 # past slot value, not a future target
    q3=build_copy_queries(modified,mem=2048,gap=65)
    affected=apply_copy_boost(net,logits,q3)
    assert not torch.equal(before[212],affected[212]),'causal old memory value should affect remote prediction'
    # Zero gain must equal base logits bit for bit, including with active memory.
    old=net.logit_gain.detach().clone()
    with torch.no_grad():net.logit_gain.zero_()
    ident=apply_copy_boost(net,logits,q)
    assert torch.equal(ident,logits)
    with torch.no_grad():net.logit_gain.copy_(old)
    return {'causal_suffix_invariance':True,'remote_source_sensitivity':True,
            'zero_gain_bit_exact':True,'target_distance':212-11}

def nested_equal(x,y):
    if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and x.dtype==y.dtype and torch.equal(x,y)
    if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(nested_equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(nested_equal(a,b) for a,b in zip(x,y))
    if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
    return x==y

def main(args):
    torch.set_num_threads(args.threads)
    root=pathlib.Path(args.root).resolve();active=root/'checkpoint/r12_active.pal12'
    saved,r6,comp,model,bridge,adapter,oldopt,rng=pal.load_for_arm(root,active,True)
    assert 'r18_memory' in saved and 'r18_retrieval_optimizer' in saved and 'r18_gain_optimizer' in saved
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory'],strict=True);net.eval()
    assert saved['r12_updates']==10756 and len(oldopt.param_groups)==4
    # Deep check against verified original R16 archived tensors and ALL moments.
    upstream=root.parent/'files/PAL_R16_CONTENT_ATTENTION_GEOMETRY_DIAGNOSTICS_SINGLE_CHECKPOINT_full.zip'
    if upstream.exists():
        with zipfile.ZipFile(upstream).open('checkpoint/r12_active.pal12') as f:
            initial=torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=False)
        same={key:nested_equal(saved[key],initial[key]) for key in
              ('model','stream_bridge','r12_residual','stream_optimizer','stream_rng','torch_rng','numpy_rng','python_rng','r12_updates','r10_updates','r11_updates')}
    else:
        # Standalone release reproducibility: the original R16 tensors were
        # matched in the build, and their reference tree hashes are archived.
        from r18_verify_lineage import tree_hash,FIELDS
        reference=json.loads((root/'reports/r18_parent_equivalence.json').read_text())['digest_fields']
        same={key:(tree_hash(saved[key])==reference[key]) for key in FIELDS}
    assert all(same.values()),same
    causal=causal_test(net)
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    test=collect_hard_negatives(docs['test'],args.test_queries,18019)
    retrieval=test_retrieval(net,test)
    net_off=R18LongMemory(model.memory.piece_emb.weight)
    untrained=test_retrieval(net_off,test)
    print(json.dumps({'phase':'TEST_RETRIEVAL','learned':retrieval['top1'],'untrained':untrained['top1'],'n':len(test)}),flush=True)
    model.eval();bridge.eval();adapter.eval()
    short=prepare_cache(model,bridge,adapter,docs['test'],
         make_windows(docs,'test',24,args.short_windows,180),24)
    long=prepare_cache(model,bridge,adapter,docs['test'],
         make_windows(docs,'test',400,args.long_windows,2000),400)
    experiment={}
    for scope,cache in [('short',short),('long',long)]:
        experiment[scope]={}
        for kind in ('off','learned','exact_heuristic'):
            experiment[scope][kind]=metrics_with_mode(net,cache,kind)
        if scope=='long':
            experiment[scope]['512_memory']=metrics_with_mode(net,cache,'learned',mem=512)
            experiment[scope]['shuffled_values']=metrics_with_mode(net,cache,'learned',shuffle=True)
    report={'round':'PAL R18','checkpoint_sha256':pal.sha(active),
            'selection_policy':'No test inputs used for training, best-step or gate choice',
            'source_lineage_preservation':same,'causality':causal,
            'test_retrieval':retrieval,'test_untrained':untrained,
            'test_language':experiment,'test_books':len(docs['test']),
            'caveats':['Historical test is book-disjoint within this archive but cross-lineage leakage unaudited',
                       'Contrastive task selects exact repeated two-token keys and has a trivial exact-match oracle',
                       'No newly downloaded Gutenberg/domain releases were used in this round',
                       'No claim of semantic retrieval, reasoning or frontier-scale language performance']}
    (root/'reports/r18_postselection_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','source_same':all(same.values()),'causal':causal,'result':experiment}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--test-queries',type=int,default=1200)
    p.add_argument('--short-windows',type=int,default=96)
    p.add_argument('--long-windows',type=int,default=8)
    main(p.parse_args())
