#!/usr/bin/env python3
"""Recalculate fixed-position audit summaries from independently logged shards.

Correctly combines *sums*, not averages of rounded results, and checks offsets.
"""
import argparse,json,pathlib,math
FIELDS=('active_nll','semantic_off_nll','pair_off_nll','one_word_only_nll','retrieval_off_nll','outer_off_nll')
COUNTS=('semantic_coverage','semantic_target_hits','exact_key_coverage','pair_coverage')

def audit(files):
    shards=[json.loads(pathlib.Path(f).read_text()) for f in files]
    shards.sort(key=lambda d:d['shard_start'])
    assert len({(r['split'],r['step'],r['total_fixed_count']) for r in shards})==1
    last=0;total=0;totals={k:0.0 for k in FIELDS};totals.update({k:0 for k in COUNTS})
    for s in shards:
        assert s['shard_start']==last,(s['shard_start'],last)
        assert s['shard_count']==s['results']['n']
        last+=s['shard_count'];total+=s['results']['n'];m=s['results']
        for k in FIELDS:totals[k]+=s['results']['n']*m[k]
        for k in COUNTS:totals[k]+=m[k]
    assert last==shards[0]['total_fixed_count']
    for k in FIELDS:totals[k]/=total
    totals['semantic_gain']=totals['semantic_off_nll']-totals['active_nll']
    totals['pair_gain']=totals['one_word_only_nll']-totals['semantic_off_nll']
    totals['exact_gain']=totals['retrieval_off_nll']-totals['one_word_only_nll']
    totals['total_gain']=totals['retrieval_off_nll']-totals['active_nll']
    totals['outer_gain']=totals['outer_off_nll']-totals['active_nll']
    return {'split':shards[0]['split'],'joint_step':shards[0]['step'],'n':total,'historical_only':True,'metrics':totals}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('shards',nargs='+');p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();res=audit(a.shards)
    if a.output:a.output.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res,indent=2))
