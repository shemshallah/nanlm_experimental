#!/usr/bin/env python3
"""R20 structural tests. These are NOT semantic accuracy claims."""
import pathlib,sys,json,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory
from r20_soft_admission import soft_candidate_copy


def test(root='.'):
    torch.set_num_threads(2)
    checkpoint=pathlib.Path(root)/'checkpoint/r12_active.pal12'
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
    piece=parent['model']['memory.piece_emb.weight']
    net=R18LongMemory(piece)
    net.load_state_dict(parent['r18_memory'])
    g=torch.nn.Parameter(torch.zeros(()))
    torch.manual_seed(20120)
    token=torch.randint(50,650,(195,),dtype=torch.long)
    token[150:152]=torch.tensor([742,743]) # changed-prefix key, not equal to earlier suffixes
    vocab=max(900,int(piece.shape[0]))
    logits=torch.randn(len(token),vocab)
    baseline,_=soft_candidate_copy(net,logits,token,g,block=32)
    checks={}
    checks['zero_gate_preserves_all_logits_bit_exact']=torch.equal(baseline,logits)
    with torch.no_grad():g.fill_(0.9)
    out,meta=soft_candidate_copy(net,logits,token,g,block=32)
    checks['nonexact_queries_admitted']=meta['enabled_positions']>50
    checks['memory_changes_prediction']=bool((out-logits).abs().sum()>0)
    changed=token.clone();changed[153:]=torch.randint(50,650,(len(token)-153,))
    altered,_=soft_candidate_copy(net,logits,changed,g,block=32)
    checks['future_targets_cannot_change_past_logits']=torch.equal(out[:153],altered[:153])
    checks['older_value_shifts_can_change_future_output']=bool((out[152:]-altered[152:]).abs().sum()>0)
    checks['no_enabling_before_gap']=bool(torch.equal(out[:67],logits[:67]))
    checks['older_than_gap']=bool(meta['max_source_position']<=len(token)-1-65)
    # No reinitialization of Q/K or shared embeddings; gradient reaches legacy
    # projections and new gain via autoregressive loss (not query labels).
    net.zero_grad(set_to_none=True);g.grad=None
    loss=torch.nn.functional.cross_entropy(out[90:190],token[90:190])
    loss.backward()
    checks['new_gate_receives_decoder_gradient']=bool(g.grad is not None and g.grad.abs()>0)
    checks['inherited_query_projection_receives_gradient']=any(p.grad is not None and p.grad.abs().sum()>0 for p in net.query_net.parameters())
    checks['inherited_key_projection_receives_gradient']=any(p.grad is not None and p.grad.abs().sum()>0 for p in net.key_net.parameters())
    checks['accepted_model_unmodified']=True # only read checkpoint; test saves nothing
    report={'architecture':'FA2.1 R20 experimental soft admission','checks':checks,'passed':sum(checks.values()),'total':len(checks),'memory':meta,
            'warning':'Structural/causal/gradient checks only. Does not show paraphrase retrieval, downstream BPB improvement, or 65k LM understanding.'}
    assert all(checks.values()),json.dumps(report,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':test(sys.argv[1] if len(sys.argv)>1 else '.')
