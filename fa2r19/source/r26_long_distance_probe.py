#!/usr/bin/env python3
"""Long-prefix CPU probe comparing R25, R26 and unrelated-book R26.
The model's recurrent state processes every earlier word, not just index keys.
"""
import argparse,json,math,pathlib,time,torch
from torch.nn import functional as F
from r26_restore_unified import restore
from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState,contiguous_book_picks
from r25_unified_cpu import source_warmup
from r25_episodic_cache import CausalNGramIndex
from r26_span_memory import CausalHighOrderIndex

def run(args):
    root=pathlib.Path(args.root).resolve()
    u,opt,comp,r6,meta,ck=restore(root)
    u.eval()
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    b=next(z for z in books['dev'] if z.book_id==str(args.book))
    donors=[z for z in books['dev'] if z.book_id!=b.book_id]
    donor=next((z for z in donors if z.book_id==str(args.donor)),donors[0])
    targets=[int(x) for x in args.targets.split(',')]
    # An index built only through the longest evaluated prefix is EXACTLY
    # equivalent for all causal queries here, and avoids expensive future text.
    cap=max(targets)+4096
    u._r25_book_indexes={b.book_id:CausalNGramIndex(b.tokens[:cap])}
    u._r26_book_indexes={b.book_id:CausalHighOrderIndex(b.tokens[:cap])}
    r25donor=CausalNGramIndex(donor.tokens[:cap])
    r26donor=CausalHighOrderIndex(donor.tokens[:cap])
    carry=ContinuousPALState(b.book_id)
    result=[];next_target=0
    print('R26_PROBE_SETUP',json.dumps({'book':b.book_id,'donor':donor.book_id,'query_positions':targets}),flush=True)
    with torch.no_grad():
        for seg in contiguous_book_picks(b,args.warm_span):
            t0=seg[2]
            if next_target<len(targets) and t0>=targets[next_target]:
                row=u.forward_chunk(b,seg,carry,apply_r25=True,apply_r26=True)
                target=row['target'];base25=row['without_r26'];nt=len(target)
                cutoff=min(t0,len(donor.tokens)-nt)
                r24=row['without_r25']
                freq,stat=r25donor.distributions(b.tokens,t0,nt,base25.shape[-1],source_cutoff=cutoff)
                wrong25=u.r25(r24,torch.from_numpy(freq),torch.from_numpy(stat))
                m,feats=r26donor.features(b.tokens,t0,nt,wrong25.exp().numpy(),target.numpy(),source_cutoff=cutoff)
                wlog=u.r26.target_logprob(wrong25.gather(1,target[:,None]).squeeze(-1),
                                         torch.from_numpy(m),torch.from_numpy(feats))
                def bpb(logits):return float(F.nll_loss(logits,target,reduction='sum'))/(row['bytes']*math.log(2))
                record={'book_id':b.book_id,'donor':donor.book_id,'requested_token':targets[next_target],
                    'actual_token':int(t0),'state_cursor':int(carry.cursor),'tokens':int(nt),
                    'bytes':int(row['bytes']),'r25_bpb':bpb(base25),
                    'r26_bpb':bpb(row['logits']),
                    'wrong_book_bpb':float(-wlog.sum())/(row['bytes']*math.log(2))}
                far=[]
                high=u._r26_book_indexes[b.book_id]
                for q in range(t0,t0+nt):
                    for n in (4,5,6):
                        js=high.positions(b.tokens,q,n,131072,q)
                        if js:far.append(int(q-js[0]))
                record['furthest_high_order_selected_tokens']=max(far,default=0)
                result.append(record);print('R26_PROBE',json.dumps(record),flush=True)
                next_target+=1
                if next_target==len(targets):break
            else:source_warmup(u,b,seg,carry)
            carry.detach()
    (root/'reports'/f'r26_longdistance_dev_{b.book_id}.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--book',default='970');p.add_argument('--donor',default='1444')
    p.add_argument('--targets',default='33000,65000')
    p.add_argument('--warm-span',type=int,default=36)
    run(p.parse_args())
