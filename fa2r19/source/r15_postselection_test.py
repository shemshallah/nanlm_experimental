#!/usr/bin/env python3
"""Historical R14-matched test; evaluation only, never used for checkpoint selection."""
import hashlib, json, pathlib, torch
import r14_slim_base as pal
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
from r11_raw import windows
pal.run_window=run_window_fast
root=pathlib.Path('.');checkpoint=root/'checkpoint/r12_active.pal12'
torch.set_num_threads(2)
saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,checkpoint,True)
docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
picks=windows(docs,24,160,160)['test']
result={'dataset':'NaNROUTER historical book-split test','independent_sealed':False,
 'note':'Older PAL corpus overlap not completely ruled out; not used for model selection.',
 'test_books':len(docs['test']),'sample_book_ids':sorted({docs['test'][i].book_id for i,_ in picks}),
 'selection_hash':hashlib.sha256(json.dumps(picks).encode()).hexdigest(),
 'sample_windows':len(picks),'checkpoint_sha256':pal.sha(checkpoint),'r12_updates':saved['r12_updates'],
 'test_metrics':pal.evaluate(model,bridge,adapter,docs['test'],picks,24,2),
 'no_attention':pal.evaluate(model,bridge,adapter,docs['test'],picks,24,2,disable_attention=True),
 'no_ffn':pal.evaluate(model,bridge,adapter,docs['test'],picks,24,2,disable_ffn=True),
 'no_both':pal.evaluate(model,bridge,adapter,docs['test'],picks,24,2,disable_attention=True,disable_ffn=True)}
(root/'reports/r15_postselection_historical_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
