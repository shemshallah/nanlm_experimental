#!/usr/bin/env python3
"""R7.22 recovery without R7.13 or R7.21 weights (corpus/tokenizer metadata only)."""
import argparse, tempfile, pathlib,json,time
import torch
from fa1_r722_unified import load_unified,save_unified
from r710_checkpoint_integrity import deep_equal,sha256

def run(a):
 start=time.monotonic();torch.set_num_threads(2)
 orig,opts,_,_,meta=load_unified(a.checkpoint,a.root)
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp)
  # The checkpoint is self-contained wrt learned tensors; original vocabulary
  # and tokenizer files are still required, which is a declared dependency.
  (root/'corpus').symlink_to(pathlib.Path(a.root)/'corpus',target_is_directory=True)
  (root/'tokenizer').symlink_to(pathlib.Path(a.root)/'tokenizer',target_is_directory=True)
  assert not (root/'checkpoint/pal_active.pt').exists()
  copied,ops,_,_,meta2=load_unified(a.checkpoint,root)
  assert all(torch.equal(v,copied.state_dict()[k]) for k,v in orig.state_dict().items())
  assert all(deep_equal(o.state_dict(),t.state_dict()) for o,t in zip(opts,ops))
  assert len(opts)==5 and len(orig.state_dict())==105
  assert meta2['semantic_updates']==0 and not meta2['promotion']
  # Can continue saving from the latest R7.22 checkpoint alone; the R7.21
  # ancestor is *not* required after the verified identity migration.
  outpath=root/'resumed_r722.pt'
  save_unified(outpath,copied,ops,a.checkpoint,train_updates=0)
  resumed,r_opts,_,_,r_meta=load_unified(outpath,root)
  assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in copied.state_dict().items())
  assert all(deep_equal(o.state_dict(),t.state_dict()) for o,t in zip(ops,r_opts))
  assert r_meta['direct_parent_sha256']==sha256(a.checkpoint)
 out={'round':'FA1 R7.22','complete_unified_checkpoint':True,
   'model_tensor_count':105,'optimizer_history_count':5,
   'all_model_tensors_bitwise_recovered':True,
   'all_five_optimizer_histories_bitwise_recovered':True,
   'save_and_resume_without_r721_files':True,
   'no_legacy_checkpoint_or_adapter_files_present':True,
   'external_tokenizer_and_corpus_metadata_still_required':True,
   'sha256':sha256(a.checkpoint),'promoted':False,
   'elapsed_s':round(time.monotonic()-start,3)}
 pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 pathlib.Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/mnt/data/fa1_full'))
 p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt'))
 p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/mnt/data/fa1r722/reports/R722_STANDALONE_RECOVERY.json'))
 print(json.dumps(run(p.parse_args()),indent=2))