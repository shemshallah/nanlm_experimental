#!/usr/bin/env python3
"""Frozen, preselected R20 content-only gate vs one accepted R18 checkpoint.

The positive selection gain=1.0 and content_only=True was made on development
windows before this script was run on the HISTORICAL book-disjoint test split.
Nothing here modifies or promotes the checkpoint. Historical test has been used
by earlier PAL rounds, so it is not a pristine independent new-book test.
"""
import argparse,json,math,pathlib,sys,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r18_joint_continue import integrated_logits
from r13_corpus_cache import read_books_cached
from r11_raw import batch,windows
from r20_soft_admission import soft_candidate_copy

@torch.no_grad()
def evaluate(root,split='test',short_windows=32,long_windows=8):
    root=pathlib.Path(root);ck=root/'checkpoint/r12_active.pal12';original=pal.sha(ck)
    saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ck,True)
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory']);net.eval();model.eval();bridge.eval();adapter.eval()
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    experiments={}
    for scope,span,nwindows,max_tokens in (('short',24,short_windows,180),('long',400,long_windows,2000)):
        w=windows(docs,span,max(8000,nwindows),max_tokens)[split][:nwindows]
        metrics={key:{'loss':0.,'nonspace_loss':0.,'bytes':0,'nonspace_count':0,'correct':0,'nonspace_correct':0,'n':0} for key in ('r18','r20_content_only_gain1')}
        for pick in w:
            ex,lc,t,tags,bytes_=batch(docs[split],[pick],span)
            n=int((t[0]!=-100).sum());x=t[0,:n];mask=x!=32
            original_logits=integrated_logits(model,bridge,adapter,net,ex,lc,tags,t)[0,:n]
            candidate,_=soft_candidate_copy(net,original_logits,x,torch.tensor(1.),content_only=True)
            for label,logits in (('r18',original_logits),('r20_content_only_gain1',candidate)):
                ce=F.cross_entropy(logits,x,reduction='none');m=metrics[label]
                m['loss']+=float(ce.sum());m['nonspace_loss']+=float(ce[mask].sum())
                m['bytes']+=bytes_;m['n']+=n;m['nonspace_count']+=int(mask.sum())
                m['correct']+=int((logits.argmax(-1)==x).sum())
                m['nonspace_correct']+=int(((logits.argmax(-1)==x)&mask).sum())
        experiments[scope]={'windows':len(w),'arms':{name:{'bpb':z['loss']/z['bytes']/math.log(2),
           'nonspace_nll':z['nonspace_loss']/z['nonspace_count'],'nonspace_top1':z['nonspace_correct']/z['nonspace_count'],
           'raw_bytes':z['bytes'],'targets':z['n']} for name,z in metrics.items()}}
    assert original==pal.sha(ck)
    out={'round':'FA2.1 R20 content-only frozen-gain historical postselection test','split':split,
         'preselected_on_dev':{'gain':1.0,'content_only':True},'checkpoint_sha256':original,
         'results':experiments,'checkpoint_promoted':False,
         'limitations':['Historical test book-disjoint but used in prior PAL experiments; not untouched new books',
             'No R20 training, semantic paraphrase benchmark or multi-hop downstream evaluation']}
    dst=root/f'reports/r20_content_{split}_postselection.json';dst.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--split',choices=('dev','test'),default='test');args=p.parse_args()
    torch.set_num_threads(2);evaluate(args.root,args.split)
