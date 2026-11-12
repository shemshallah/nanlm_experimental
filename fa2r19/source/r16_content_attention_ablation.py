#!/usr/bin/env python3
"""R16 observational diagnostic. No optimizer steps or checkpoint mutation.

Loss is on the byte-reversible subword stream. For category BPB, each token's
log-loss is distributed uniformly across the raw bytes it represents. For
SentencePiece byte-fallback tokens, codepoint bytes are assigned fractionally
across all overlapping piece spans. Category contributions sum to total BPB.
This is a token-log-loss *attribution* by byte category, NOT a new byte decoder.
"""
import argparse, collections, hashlib, json, math, os, pathlib, sys, time
import numpy as np, torch
from torch.nn import functional as F
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r11_raw import windows, batch
from build_lossless_subwords import TOKEN_PATTERN, decode_bytes

CATEGORIES=['space','lowercase','uppercase','punctuation','newline','digits','other_whitespace','nonascii','control']


def category(byte):
 if byte == 32:return 'space'
 if byte==10:return 'newline'
 if byte in (9,11,12,13):return 'other_whitespace'
 if 97<=byte<=122:return 'lowercase'
 if 65<=byte<=90:return 'uppercase'
 if 48<=byte<=57:return 'digits'
 if 33<=byte<=47 or 58<=byte<=64 or 91<=byte<=96 or 123<=byte<=126:return 'punctuation'
 if byte>=128:return 'nonascii'
 return 'control'


def slice_metadata(book,pick,span,sp,root):
 _,start=pick;end=min(start+span,len(book.chunks));assert end<len(book.chunks)
 t0=int(book.chunks[start]);t1=int(book.chunks[end]);
 b0=int(book.byte_starts[start]);b1=int(book.byte_starts[end]);
 raw=(root/book.split/(book.book_id+'.txt')).read_bytes()[b0:b1]
 ids=book.tokens[t0:t1].tolist();q=0;weights=[]
 for match in TOKEN_PATTERN.finditer(raw):
  part=match.group()
  if all(b in (9,10,11,12,13,32) for b in part):
   assert ids[q:q+len(part)]==list(part),('whitespace token mismatch',book.book_id,start)
   for b in part:weights.append({category(b):1.})
   q+=len(part);continue
  word=part.decode('utf8','strict');proto=sp.encode(word,out_type='immutable_proto')
  pieces=list(proto.pieces);part_ids=[256+p.id for p in pieces]
  assert ids[q:q+len(part_ids)]==part_ids,('piece ids mismatch',book.book_id,start)
  # Codepoint offsets in SentencePiece proto, not byte offsets.
  per_char=[[] for _ in word]
  for j,p in enumerate(pieces):
   lo=max(0,min(int(p.begin),len(word)-1))
   hi=max(lo+1,min(int(p.end),len(word)))
   for ci in range(lo,hi):per_char[ci].append(j)
  assert all(per_char),('missing character attribution',book.book_id,start)
  pw=[collections.defaultdict(float) for _ in pieces]
  for ci,ch in enumerate(word):
   holders=per_char[ci]
   for byte in ch.encode('utf8'):
    for j in holders:pw[j][category(byte)]+=1./len(holders)
  weights.extend(pw);q+=len(part_ids)
 assert q==len(ids),(q,len(ids),book.book_id)
 totals=collections.Counter()
 for row in weights:totals.update(row)
 assert abs(sum(totals.values())-len(raw))<1e-5,(totals,len(raw))
 true_counts=collections.Counter(category(byte) for byte in raw)
 for key in CATEGORIES:assert abs(totals[key]-true_counts[key])<1e-5,(key,totals[key],true_counts[key])
 return weights,len(raw)


def forward_token_losses(model,bridge,adapter,h,targets,tags,disable_attention=False,disable_ffn=False,capacity=64,route='legacy',router=None,return_tensors=False):
 B,T=targets.shape;previous=torch.full((B,),model.memory.start_piece,device=targets.device,dtype=torch.long)
 state=h[:,0];last=tags[:,0];retain=bridge.retain_logit.sigmoid();khist=[];vhist=[];logits_seq=[]
 for t in range(T):
  valid=targets[:,t]!=-100;ix=tags[:,t].clamp(0,h.shape[1]-1)
  next_h=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
  changed=(ix!=last)&valid
  state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*next_h,state)
  state=model.memory.subword_cell(model.memory.piece_emb(previous),state)
  out=state
  if not disable_attention and khist:
   keys=torch.stack(khist[-capacity:],dim=1);values=torch.stack(vhist[-capacity:],dim=1)
   scores=(adapter.query(state).unsqueeze(1)*keys).sum(-1)/math.sqrt(state.shape[-1])
   k=min(adapter.topk,scores.shape[1]);inds=scores.topk(k,dim=1).indices
   selected=values.gather(1,inds[:,:,None].expand(-1,-1,state.shape[-1]))
   recalled=(selected*F.softmax(scores.gather(1,inds),dim=-1)[:,:,None]).sum(1)
   out=out+adapter.attn_gain*adapter.attn_out(recalled)
  if not disable_ffn:
   if router is not None:
    ff=adapter.ffn[2](router(adapter.ffn[0](state)))
   elif route=='legacy':ff=adapter.ffn(state)
   else:
    x=adapter.ffn[0](state)
    if route=='linear':
     # Exact Euclidean reference. Linear projections and SiLU activation.
     y=F.silu(x)
    elif route=='geometry':
     # Geometric routing REPLACES the FFN nonlinearity, not added to its output.
     # Poincare radial factor via conformal norm, bounded to avoid instability.
     # This is an *untrained activation substitution*, not a matched trained test.
     z=torch.tanh(x/4.)
     r2=z.square().mean(dim=-1,keepdim=True).clamp(max=.999)
     conf=2/(1-r2)
     y=z*conf/2
    else:raise ValueError(route)
    ff=adapter.ffn[2](y)
   out=out+adapter.ffn_gain*ff
  if not disable_attention:
   khist.append(adapter.key(state));vhist.append(adapter.value(state))
  logits_seq.append(model.memory.subword_out(out))
  previous=torch.where(valid,targets[:,t],torch.zeros_like(previous));last=torch.where(valid,ix,last)
 logits=torch.stack(logits_seq,dim=1)
 nll=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='none').reshape(B,T)
 if return_tensors:return nll,logits.argmax(-1)
 return nll.detach().cpu().numpy(),logits.argmax(-1).detach().cpu().numpy()


def evaluate(model,bridge,adapter,books,picks,span,root,sp,caps=(64,),ablations=True,routes=False,report_prefix=''):
 variants={}
 for cap in caps:
  variants[f'cap{cap}']={'capacity':cap}
  if ablations:variants[f'cap{cap}_no_attention']={'capacity':cap,'disable_attention':True}
  if ablations and cap==caps[0]:
   variants[f'cap{cap}_no_ffn']={'capacity':cap,'disable_ffn':True}
   variants[f'cap{cap}_no_both']={'capacity':cap,'disable_ffn':True,'disable_attention':True}
 if routes:variants.update({'ffn_linear':{'capacity':caps[0],'route':'linear'},'ffn_geometric_untrained':{'capacity':caps[0],'route':'geometry'}})
 scores={k:{'total_nats':0.,'raw_bytes':0,'category_nats':collections.defaultdict(float),'category_bytes':collections.defaultdict(float),'category_correct_weight':collections.defaultdict(float),'token_total':0} for k in variants}
 total_windows=len(picks);tic=time.time()
 with torch.no_grad():
  for idx in range(0,len(picks),2):
   selected=picks[idx:idx+2]
   e,lc,t,tag,bcount=batch(books,selected,span)
   B,W=e.shape[:2]
   h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
   meta=[slice_metadata(books[bi],(bi,start),span,sp,root) for bi,start in selected]
   assert sum(z[1] for z in meta)==bcount
   for name,kw in variants.items():
    nll,pred=forward_token_losses(model,bridge,adapter,h,t,tag,**kw)
    entry=scores[name]
    for i,(weights,bytes_) in enumerate(meta):
     assert len(weights)==int((t[i]!=-100).sum())
     entry['total_nats']+=float(nll[i,:len(weights)].sum())
     entry['token_total']+=len(weights);entry['raw_bytes']+=bytes_
     for j,d in enumerate(weights):
      size=sum(d.values());assert size>0
      correct=float(pred[i,j]==int(t[i,j]))
      for cat,weight in d.items():
       entry['category_bytes'][cat]+=weight
       entry['category_nats'][cat]+=float(nll[i,j])*weight/size
       entry['category_correct_weight'][cat]+=correct*weight
   if idx%16==0:print(json.dumps({'phase':'EVAL','stage':report_prefix,'windows':idx+len(selected),'total':total_windows,'elapsed_s':round(time.time()-tic,1)}),flush=True)
 results={}
 for name,v in scores.items():
  by={cat:{'bytes':round(v['category_bytes'][cat],5),'attributed_bpb':v['category_nats'][cat]/v['category_bytes'][cat]/math.log(2) if v['category_bytes'][cat] else None,'piece_hit_weighted':v['category_correct_weight'][cat]/v['category_bytes'][cat] if v['category_bytes'][cat] else None,'bits':v['category_nats'][cat]/math.log(2)} for cat in CATEGORIES}
  nats_sum=sum(v['category_nats'].values())
  assert abs(nats_sum-v['total_nats'])<.02,(name,nats_sum,v['total_nats'])
  results[name]={'bpb':v['total_nats']/v['raw_bytes']/math.log(2),'raw_bytes':v['raw_bytes'],'n_subwords':v['token_total'],'by_category':by}
 return results


def main(args):
 torch.set_num_threads(args.threads)
 root=pathlib.Path(args.root).resolve();ckpt=root/'checkpoint/r12_active.pal12'
 assert pal.sha(ckpt)=='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f','wrong parent R15 checkpoint'
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ckpt,True)
 model.eval();bridge.eval();adapter.eval()
 books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 # Identical R15 post-selection historical picks; NO model tuning on these.
 test_picks=windows(books,24,160,160)['test']
 report={'R15_checkpoint_SHA256':pal.sha(ckpt),'R12_updates':saved['r12_updates'],
    'method':'subword NLL allocated over byte categories via SentencePiece codepoint spans; values sum to true total BPB',
    'category_note':'Not separately normalized per-byte probabilities. Category metrics are attributed cross-entropies.',
    'historical_test_selection':'R15 historical fixed picks; overlap with previous PAL corpora not fully excluded',
    'historical_test':evaluate(model,bridge,adapter,books['test'],test_picks,24,root/'nanrouter',comp.sp,caps=(64,),ablations=True,report_prefix='test'),
    'expansion_note':'Untrained zero-update retrospective context-capacity evaluation on development books, not an architectural training comparison'}
 # Fixed new dev windows, independently select longer spans before evaluating,
 # no tuning from historical test data. 96 chunks spans > 64 subword history.
 dev_picks=windows(books,args.long_span,args.long_windows,args.max_long_tokens)['dev']
 report['long_dev']={'span_chunks':args.long_span,'windows':len(dev_picks),'picks_hash':hashlib.sha256(json.dumps(dev_picks).encode()).hexdigest(),
   'results':evaluate(model,bridge,adapter,books['dev'],dev_picks,args.long_span,root/'nanrouter',comp.sp,caps=(64,256,1024),ablations=False,routes=True,report_prefix='long_dev')}
 report['scope_limits']=['Frozen R15 weights: geometry activation replacement is NOT matched-trained and cannot establish geometric utility.',
   'Long-span model was trained mostly on 24-chunk windows (64-token attention), so extrapolation may fail.',
   'Historical test is not independent; long-span development is not a sealed test.',
   'No change to optimizer, checkpoint, training update counter, or RNG state.']
 out=root/'reports/r16_content_attention_ablation.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'DONE','report':str(out),'test_bpb':{k:v['bpb'] for k,v in report['historical_test'].items()},'long_dev_bpb':{k:v['bpb'] for k,v in report['long_dev']['results'].items()}},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--threads',type=int,default=2);p.add_argument('--long-span',type=int,default=96);p.add_argument('--long-windows',type=int,default=6);p.add_argument('--max-long-tokens',type=int,default=650)
 main(p.parse_args())
