#!/usr/bin/env python3
"""Fast structural, non-oracle tests of FA2.3's R22 causal multiscale memory."""
import hashlib,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory
from r22_multiscale_memory import R22Residual, hierarchical_proposals, BANDS


def run(root):
    torch.set_num_threads(2)
    ck=pathlib.Path(root)/'checkpoint/r12_active.pal12'
    digest=hashlib.sha256(ck.read_bytes()).hexdigest()
    saved=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
    net=R18LongMemory(saved['model']['memory.piece_emb.weight'])
    net.load_state_dict(saved['r18_memory']);net.eval()
    torch.manual_seed(22001)
    # Recurrent bytes in both 8k and 32k regimes; a controlled repetition
    # ensures that exact-prefix rescue gets candidates even after striding.
    tok=torch.randint(33,115,(33400,),dtype=torch.long)
    tok[400:403]=torch.tensor([70,71,72])
    tok[25200:25203]=torch.tensor([70,71,73])
    tok[33100:33103]=torch.tensor([70,71,74])
    t0=33200;n=200;v=256
    logits=torch.randn(n,v)
    d,f,m=hierarchical_proposals(net,logits,tok,t0)
    reader=R22Residual(minimum_query_position=0)
    out,g=reader(logits,d,f)
    check={}
    check['zero_residual_preserves_original_logits_exactly']=torch.equal(out,logits)
    check['real_recent_memory_retrieval']=m['recent']['eligible']>0
    check['real_middle_memory_retrieval']=m['middle']['eligible']>0
    check['real_32k_far_memory_retrieval']=m['far']['eligible']>0
    check['at_least_one_far_read_beyond_8192']=m['far']['maximum_read_distance']>=8192
    check['extra_prefix_rescue_candidates']=m['recent']['rescue_positions']>0
    check['finite_memory_proposals']=all(torch.isfinite(d[z]).all().item() for z in BANDS)
    with torch.no_grad(): reader.branches['far'].amplitude.fill_(0.3)
    altered,g=reader(logits,d,f)
    check['far_gate_changes_predictions']=bool((altered-logits).abs().sum()>0)
    # Source and query prefixes for positions <80 are invariant to edited
    # later tokens, including future *targets* within window.
    changed=tok.clone();changed[t0+80:]=torch.randint(33,115,(len(tok)-t0-80,))
    d2,f2,m2=hierarchical_proposals(net,logits,changed,t0)
    check['future_text_cannot_change_past_predictions']=all(
        torch.equal(d[z][:80],d2[z][:80]) and torch.equal(f[z][:80],f2[z][:80]) for z in BANDS)
    # The old memory is frozen, but gradients must flow into all new gates.
    with torch.no_grad():
        for b in BANDS:reader.branches[b].amplitude.fill_(0.4)
    for b in BANDS:reader.branches[b].amplitude.grad=None
    combined,_=reader(logits,d,f)
    loss=torch.nn.functional.cross_entropy(combined[80:],tok[t0+80:t0+n])
    loss.backward()
    check['all_three_new_gates_receive_lm_gradients']=all(
        reader.branches[b].amplitude.grad is not None and
        reader.branches[b].amplitude.grad.abs().item()>0 for b in BANDS)
    check['checkpoint_not_overwritten']=hashlib.sha256(ck.read_bytes()).hexdigest()==digest
    assert all(check.values()),json.dumps(check,indent=2)
    print(json.dumps({'checks':check,'passed':sum(check.values()),'total':len(check),'metadata':m},indent=2))

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else '.')
