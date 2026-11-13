#!/usr/bin/env python3
"""Frozen R27 external-to-selection long-book probe. No weight updates."""
import argparse,hashlib,json,pathlib,torch
from r26_restore_unified import restore
from r26_unified_cpu import extract_split
from r13_corpus_cache import read_books_cached
from r27_trust import CounterfactualTrustGate
from r27_local_pilot import metrics

def main():
 root=pathlib.Path('.').resolve();torch.set_num_threads(2)
 ck=root/'reports/r27_local_overlay.pt'
 if not ck.exists():
  print('R27_RESERVED_SKIPPED: No selected candidate overlay exists');return
 data=torch.load(ck,map_location='cpu',weights_only=False)
 parent=root/'reports/fa2_r26_unified_candidate.pth'
 if hashlib.sha256(parent.read_bytes()).hexdigest()!=data['parent_r26_sha256']:
  raise RuntimeError('R26 parent digest mismatch')
 model,optimizer,comp,r6,meta,ancestor=restore(root)
 model.add_module('r27',CounterfactualTrustGate())
 model.r27.load_state_dict(data['r27_state'],strict=True)
 for p in model.parameters():p.requires_grad_(False)
 books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 ids=('8571','13505','15985')
 group=[b for b in books['dev'] if str(b.book_id) in ids]
 if len(group)!=len(ids):raise ValueError('missing locked reserve books')
 rows=extract_split(model,group,[132000],20,1,'r27_reserved_no_selection')
 report={'format':'R27_HELD_OUT_LONGBOOK_AUDIT_V1','checkpoint_r27_step':data['selected_step'],
         'source_r26_sha256':data['parent_r26_sha256'],
         'selection_books_excluded':['866','970','1444'],
         'reserved_books':[b.book_id for b in group],
         'scope':'Historical PAL development authors, not globally unseen authors',
         'results':metrics(model,rows)}
 (root/'reports/r27_reserve_long_132k.json').write_text(json.dumps(report,indent=2)+'\n')
 print('R27_RESERVED_FINAL',json.dumps(report),flush=True)

if __name__=='__main__':main()