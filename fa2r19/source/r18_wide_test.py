#!/usr/bin/env python3
"""Wider 32-window held-out test, frozen checkpoint and no test-driven tuning."""
import json,pathlib,sys,time,math,collections
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r18_long_memory import R18LongMemory,build_copy_queries,apply_copy_boost
from r18_run import make_windows,prepare_cache

def main():
 torch.set_num_threads(2);root=pathlib.Path(__file__).resolve().parents[1]
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,root/'checkpoint/r12_active.pal12',True)
 net=R18LongMemory(model.memory.piece_emb.weight);net.load_state_dict(saved['r18_memory']);net.eval()
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 choices=make_windows(docs,'test',400,32,2000)
 cache=prepare_cache(model,bridge,adapter,docs['test'],choices,400)
 rows=[];t0=time.monotonic()
 with torch.no_grad():
  for row in cache:
   q=build_copy_queries(row['tokens'],2048,65)
   adapted=apply_copy_boost(net,row['logits'],q)
   base_ce=F.cross_entropy(row['logits'],row['tokens'],reduction='none')
   new_ce=F.cross_entropy(adapted,row['tokens'],reduction='none')
   mask=(row['tokens']!=32)
   rows.append({'book':row['book'],'chunk':row['chunk'],'bytes':row['bytes'],
                'tokens':len(row['tokens']),'eligible':len(q['positions']) if q else 0,
                'base_nats':float(base_ce.sum()),'new_nats':float(new_ce.sum()),
                'base_other_nats':float(base_ce[mask].sum()),'new_other_nats':float(new_ce[mask].sum()),
                'other_tokens':int(mask.sum())})
 totals={k:sum(r[k] for r in rows) for k in ('bytes','tokens','eligible','base_nats','new_nats','base_other_nats','new_other_nats','other_tokens')}
 books=collections.defaultdict(list)
 for r in rows:books[r['book']].append(r)
 book_delta={b:(sum(r['new_nats']-r['base_nats'] for r in ls)/sum(r['bytes'] for r in ls)/math.log(2)) for b,ls in books.items()}
 delta=[r['new_nats']-r['base_nats'] for r in rows]
 # bootstrap over books only; already-selected historical test, descriptive CI
 rng=np.random.default_rng(18018)
 bookagg=[(sum(r['new_nats']-r['base_nats'] for r in ls),sum(r['bytes'] for r in ls)) for ls in books.values()]
 boots=[]
 for _ in range(2000):
  sample=rng.choice(len(bookagg),len(bookagg),replace=True)
  boots.append(sum(bookagg[i][0] for i in sample)/sum(bookagg[i][1] for i in sample)/math.log(2))
 report={'round':'PAL R18 wider historical test','test_windows':len(rows),'test_books':len(books),'tokens':totals['tokens'],
 'eligible_queries':totals['eligible'],'base_bpb':totals['base_nats']/totals['bytes']/math.log(2),
 'r18_bpb':totals['new_nats']/totals['bytes']/math.log(2),
 'difference_bpb':(totals['new_nats']-totals['base_nats'])/totals['bytes']/math.log(2),
 'baseline_nonspace_nll':totals['base_other_nats']/totals['other_tokens'],
 'r18_nonspace_nll':totals['new_other_nats']/totals['other_tokens'],
 'improved_windows':sum(d<0 for d in delta),'improved_books':sum(x<0 for x in book_delta.values()),
 'book_delta_bpb':dict(book_delta),'bootstrap_95pct_book_delta_bpb':list(np.quantile(boots,[.025,.975])),
 'performance_note':'One-time historical test follow-up, not sealed; dataset overlap across PAL lineages not audited',
 'timing_seconds_after_base_cache':time.monotonic()-t0}
 (root/'reports/r18_wider_32test.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
