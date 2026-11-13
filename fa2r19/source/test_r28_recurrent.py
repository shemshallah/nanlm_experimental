"""FA2 R28 causal mirror recurrence: identity, BPTT and isolation tests.

Synthetic only: intentionally runs in GitHub CI without the 124MB checkpoint.
"""
import torch
from torch import nn
from r23_stream_carry import ContinuousPALState
from r28_recurrent import MirrorRecurrentResidual,bind_recurrent

class SmoothCell(nn.Module):
    def forward(self,x,h):return .96*h+.04*x
class ToyMemory(nn.Module):
    def __init__(self,dim=96,vocab=24):
        super().__init__();self.start_piece=0
        self.piece_emb=nn.Embedding(vocab,dim)
        self.subword_cell=SmoothCell();self.subword_out=nn.Linear(dim,vocab)
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.memory=ToyMemory()
        self.r28=MirrorRecurrentResidual()
class ToyBridge(nn.Module):
    def __init__(self):
        super().__init__();self.retain_logit=nn.Parameter(torch.tensor(2.))
        self.context_gain=nn.Parameter(torch.tensor(1.))

def chunks(model,bridge,targets,*,refined,first_state_hook=False):
    state=ContinuousPALState('toy')
    if refined:bind_recurrent(state,model)
    output=[];saved=None
    for k in range(4):
        # 48 tokens in four 12-token BPTT fragments, with no graph detaches.
        y=targets[:,k*12:(k+1)*12]
        hidden=torch.full((1,1,96),.01*(k+1))
        tags=torch.zeros_like(y)
        out=state.forward_chunk(model,bridge,None,hidden,y,tags,
            book_id='toy',token_offset=k*12,chunk_tag_offset=k)
        output.append(out)
        if k==0 and first_state_hook:
            saved=state.state
            saved.retain_grad()
    return torch.cat(output,dim=1),saved

def run():
    torch.manual_seed(28028)
    model=ToyModel();bridge=ToyBridge()
    labels=torch.randint(0,24,(1,48))
    without,_=chunks(model,bridge,labels,refined=False)
    identity,_=chunks(model,bridge,labels,refined=True)
    assert torch.equal(without,identity),'R27 source logits changed at zero R28 gain'
    # The logit at time t cannot depend on target[t] or any later target.
    changed=labels.clone();changed[:,36:]=torch.randint(0,24,(1,12))
    future,_=chunks(model,bridge,changed,refined=True)
    assert torch.equal(identity[:,:37],future[:,:37]),'future leaked into prefix'
    # Make the residual active and verify the fourth chunk has a gradient
    # path all the way into the first chunk's recurrent state.
    with torch.no_grad():model.r28.amplitude.fill_(.5)
    out,first_state=chunks(model,bridge,labels,refined=True,first_state_hook=True)
    loss=out[:,36:].square().mean();loss.backward()
    assert first_state.grad is not None and first_state.grad.norm()>1e-8,'truncated at 12/24 tokens'
    assert model.r28.up.weight.grad is not None and model.r28.up.weight.grad.norm()>0
    # Strict stream isolation must remain intact even after R28 binding.
    carry=bind_recurrent(ContinuousPALState('book-a'),model)
    try:
        carry.forward_chunk(model,bridge,None,torch.zeros((1,1,96)),labels[:,:1],
            torch.zeros((1,1),dtype=torch.long),book_id='book-b',token_offset=0,chunk_tag_offset=0)
        raise AssertionError('cross-book stream accepted')
    except ValueError as exc:
        assert 'cross-book' in str(exc)
    print('R28 PASS: exact identity, future isolation, cross-book rejection, '
          '48-token four-chunk gradient and trainable mirror residual')

if __name__=='__main__':run()
