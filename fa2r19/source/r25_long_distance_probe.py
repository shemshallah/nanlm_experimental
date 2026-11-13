#!/usr/bin/env python3
"""R25 genuine recurrent-state carry and causal ngram donor check to 32K."""
import argparse,json,math,pathlib,time,torch
from torch.nn import functional as F
from r25_unified_cpu import prepare
from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState,contiguous_book_picks
from r25_episodic_cache import CausalNGramIndex
from r24_longdistance_probe import warmup

def run(args):
    root=pathlib.Path(args.root).resolve();start=time.monotonic()
    model,opt,comp,r6,meta,parent,legacy=prepare(root,args.threads)
    candidate=torch.load(root/'reports/fa2_r25_unified_candidate.pth',weights_only=False,map_location='cpu')
    if candidate['format']!='FA2_R25_UNIFIED_CANDIDATE_V1':raise ValueError('wrong checkpoint')
    model.load_state_dict(candidate['selected']['model']);opt.load_state_dict(candidate['selected']['optimizer']);model.eval()
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    book=next(b for b in docs['dev'] if b.book_id==str(args.book_id))
    donors=[b for b in docs['dev'] if b.book_id!=book.book_id];donor=donors[0]
    donor_index=CausalNGramIndex(donor.tokens)
    carry=ContinuousPALState(book.book_id);targets=[int(x) for x in args.targets.split(',')];cur=0;probes=[]
    print('SETUP',json.dumps({'book':book.book_id,'donor':donor.book_id,'targets':targets}),flush=True)
    with torch.no_grad():
        for i,seg in enumerate(contiguous_book_picks(book,args.warm_span)):
            t0=seg[2]
            if cur<len(targets) and t0>=targets[cur]:
                row=model.forward_chunk(book,seg,carry,apply_r25=True)
                y=row['target'];base=row['without_r25'];nt=len(y);v=base.shape[-1]
                alt,st=donor_index.distributions(book.tokens,t0,nt,v,source_cutoff=min(t0,len(donor.tokens)-nt))
                wrong=model.r25(base,torch.from_numpy(alt),torch.from_numpy(st))
                def bpb(pred):return float(F.cross_entropy(pred,y,reduction='sum'))/(row['bytes']*math.log(2))
                rec={'book_id':book.book_id,'requested_position':targets[cur],'actual_position':t0,
                    'state_cursor':carry.cursor,'tokens':nt,'bytes':row['bytes'],
                    'base_r24_bpb':bpb(base),'r25_bpb':bpb(row['logits']),
                    'unrelated_r25_bpb':bpb(wrong),
                    'r25_delta_bpb':bpb(base)-bpb(row['logits']),
                    'correct_vs_donor_bpb':bpb(wrong)-bpb(row['logits'])}
                probes.append(rec);print('PROBE '+json.dumps(rec),flush=True);cur+=1
                if cur==len(targets):break
            else:warmup(model,book,carry,seg)
    report={'round':'FA2 R25 selected %s-update candidate' % candidate['selected']['r25_step'], 'source_book':book.book_id,
        'observations':probes,'all_targets_reached':len(probes)==len(targets),
        'elapsed_seconds':time.monotonic()-start,'not_blind':True,
        'backprop_span':'2 R24 chunks; 32K is forward-state and index horizon only'}
    (root/f'reports/r25_longdistance_dev_{book.book_id}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL '+json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--book-id',type=int,default=970);p.add_argument('--targets',default='2100,9000,32900')
    p.add_argument('--warm-span',type=int,default=128);p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
