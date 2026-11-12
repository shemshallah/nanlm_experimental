#!/usr/bin/env python3
"""Count how much genuinely older lexical information exists in PAL's samples.

This is an *opportunity analysis*, not evidence the trained recurrent memory uses
it. Historical Gutenberg splits lack independent document provenance.
"""
import argparse,json,pathlib,numpy as np
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER

def run(root,count=4096):
    root=pathlib.Path(root);out={}
    for split in ('dev','test'):
        cor=Corpus(root,split);p=np.arange(count,dtype=np.int64)*(len(cor.seq)-1)//count
        vocab=cor.seq;cnt={'positions':len(p),'known_target':0,'in_exact':0,
                           'in_older':0,'older_only':0,'bigram_match':0,
                           'older_bigram_match':0,'older_only_bigram_match':0}
        for i in range(0,len(p),128):
            ix=p[i:i+128];past,_=cor.get(ix,OUTER_CONTEXT)
            earlier=past[:,:OLDER];exact=past[:,OLDER:]
            targets=vocab[ix].astype(np.int64)
            seen_old=np.any(earlier==targets[:,None],axis=1)
            seen_exact=np.any(exact==targets[:,None],axis=1)
            key=exact[:,-1]
            # Keys are historical words, values are their following historical word.
            old_bi=np.any((earlier[:,:-1]==key[:,None]) & (earlier[:,1:]==targets[:,None]),axis=1)
            new_bi=np.any((exact[:,:-1]==key[:,None]) & (exact[:,1:]==targets[:,None]),axis=1)
            cnt['known_target']+=int(np.sum(targets>1))
            cnt['in_exact']+=int(seen_exact.sum())
            cnt['in_older']+=int(seen_old.sum())
            cnt['older_only']+=int(np.sum(seen_old & ~seen_exact))
            cnt['bigram_match']+=int(new_bi.sum())
            cnt['older_bigram_match']+=int(old_bi.sum())
            cnt['older_only_bigram_match']+=int(np.sum(old_bi & ~new_bi))
        out[split]=cnt
    result={'past_older_words':OLDER,'exact_words':OUTER_CONTEXT-OLDER,
            'note':'Opportunity only: historical records may lack explicit document boundaries; counts are not proven learned retrieval.',
            'splits':out}
    (root/'reports/R7p4_retrieval_opportunity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--count',type=int,default=4096)
    a=p.parse_args();print(json.dumps(run(a.root,a.count),indent=2))
