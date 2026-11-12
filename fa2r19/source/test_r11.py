#!/usr/bin/env python3
"""R11 raw-source invariants. Does not change model weights/checkpoint."""
import hashlib,json,pathlib,sys
import numpy as np,torch
from r11_raw import read_books,batch,eval_raw
from r10_stream import load_model,StreamingBridge
from build_lossless_subwords import decode_bytes

def run(root,books,checkpoint):
    torch.set_num_threads(2)
    saved,r6,comp,model=load_model(root,checkpoint)
    br=StreamingBridge();br.load_state_dict(saved['stream_bridge'])
    collection,meta=read_books(books,comp,r6.vocab,max_books=1)
    a=collection['train'][0];b=collection['dev'][0];c=collection['test'][0]
    checks={}
    checks['one_checkpoint']=saved['format']=='PAL11_RAW_SINGLE_V1' and saved['r10_updates']==2020 and saved['r11_updates']>0
    checks['optimizer_continuity']=len(saved['stream_optimizer']['state'])>=10 and 'optimizer' not in saved
    checks['unique_documents']=len({a.book_id,b.book_id,c.book_id})==3 and len({a.raw_sha,b.raw_sha,c.raw_sha})==3
    checks['manifest_sha']=all(hashlib.sha256((books/z.split/(z.book_id+'.txt')).read_bytes()).hexdigest()==z.raw_sha for z in (a,b,c))
    checks['exact_raw_bytes']=all(decode_bytes(comp.sp,z.tokens)==(books/z.split/(z.book_id+'.txt')).read_bytes() for z in (a,b,c))
    checks['original_whitespace']=bool((a.tokens==10).any() and (a.tokens==32).any())
    checks['causal_history']=bool(np.array_equal(a.examples[0,:3],np.ones(3,dtype=np.int32)))
    (e,lc,t,tag,bytes_)=batch([a],[(0,5)],12)
    h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72))
    q=e.clone();q[:,:,3:]=32000-1
    checks['target_future_not_in_now']=torch.equal(h,model.core.hidden(q.reshape(-1,7),lc.reshape(-1,72)))
    checks['ordered_chunk_tag']=int(tag[0,0])==0 and bool(torch.all(tag[0,1:]>=tag[0,:-1]))
    checks['nonspace_targets']=bool(torch.any((t!=-100)&(t!=32)))
    checks['shared_mirror_shapes']=tuple(model.core.P.shape)==(32640,64) and tuple(model.core.M.shape)==(32640,64)
    with torch.no_grad():
        loss,n=br(model,h.reshape(e.shape[0],e.shape[1],-1),t,tag)
        checks['finite_loss']=bool(torch.isfinite(loss)) and n==int((t!=-100).sum())
    checks['sampling_book_boundary']=bytes_>0 and bytes_<a.raw_len
    out={'passed':sum(checks.values()),'total':len(checks),'checks':checks,'r11_updates':saved['r11_updates']}
    assert all(checks.values()),out
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '.')
    books=pathlib.Path(sys.argv[2] if len(sys.argv)>2 else 'nanrouter')
    checkpoint=pathlib.Path(sys.argv[3] if len(sys.argv)>3 else 'checkpoint/r11_active.pal11')
    run(root,books,checkpoint)
