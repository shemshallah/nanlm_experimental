"""Test that prefix-replay optimization preserves future PAL scores EXACTLY."""
import torch
from r29_long_replay import warmed_piece
from r29_horizon_match import make_groups
from r23_stream_carry import ContinuousPALState
from test_r28_recurrent import ToyModel,ToyBridge

class ToyAdapter(torch.nn.Module):
    def __init__(self,dim=96,topk=2,capacity=8):
        super().__init__();self.topk=topk;self.capacity=capacity
        self.query=torch.nn.Linear(dim,dim);self.key=torch.nn.Linear(dim,dim)
        self.value=torch.nn.Linear(dim,dim);self.attn_out=torch.nn.Linear(dim,dim)
        self.ffn=torch.nn.Linear(dim,dim)
        self.attn_gain=torch.nn.Parameter(torch.ones(()))
        self.ffn_gain=torch.nn.Parameter(torch.ones(()))

class Book:
    book_id='toy'

class Wrapper:
    def __init__(self):
        self.model=ToyModel();self.bridge=ToyBridge();self.adapter=ToyAdapter()

def test():
    # Every arm uses the identical contiguous source material per optimizer step.
    plans=[(i,i+1,i,i+1) for i in range(8)]
    for horizon in (2,4,8):
        groups=make_groups(plans,0,horizon,8)
        assert len(groups)==8//horizon
        assert [segment for group in groups for segment in group]==plans
    torch.manual_seed(29029)
    for refined in (False,True):
        u=Wrapper()
        with torch.no_grad():u.model.r28.amplitude.fill_(.4)
        a=ContinuousPALState('toy');b=ContinuousPALState('toy')
        if refined:
            a.recurrent_refiner=u.model.r28;b.recurrent_refiner=u.model.r28
        with torch.no_grad():
            for k in range(4):
                # Exercise both word-boundaries and cross-segment KV retention.
                y=torch.randint(1,23,(1,12))
                h=torch.randn((1,3,96));tags=torch.tensor([[0]*4+[1]*3+[2]*5])
                seg=(3*k,3*k+3,12*k,12*k+12)
                output=a.forward_chunk(u.model,u.bridge,u.adapter,h,y,tags,
                    book_id='toy',token_offset=k*12,chunk_tag_offset=3*k)
                warmed_piece(u,Book(),seg,b,h,tags,y)
                b.detach();a.detach()
                assert torch.equal(a.state,b.state),('state',k,refined)
                assert torch.equal(a.previous,b.previous),('previous',k,refined)
                assert torch.equal(a.last_tag,b.last_tag),('word-tag',k,refined)
                assert a.cursor==b.cursor
                assert len(a.key_history)==len(b.key_history)
                for x,z in zip(a.key_history,b.key_history):assert torch.equal(x,z)
                for x,z in zip(a.value_history,b.value_history):assert torch.equal(x,z)
            y=torch.randint(1,23,(1,12));h=torch.randn((1,3,96))
            tags=torch.tensor([[0]*4+[1]*3+[2]*5])
            x=a.forward_chunk(u.model,u.bridge,u.adapter,h,y,tags,book_id='toy',token_offset=48,chunk_tag_offset=12)
            z=b.forward_chunk(u.model,u.bridge,u.adapter,h,y,tags,book_id='toy',token_offset=48,chunk_tag_offset=12)
            assert torch.equal(x,z),'query logits changed by accelerated warmup'
    print('R29 FAST REPLAY PASS: 2 modes, 4 contiguous chunks, exact recurrent K/V/state and future logits')
if __name__=='__main__':test()
