#!/usr/bin/env python3
"""Target-leakage and book-disjoint donor tests; requires no checkpoint."""
import torch
from fa1_r725_controls import unrelated_book_context,different_book_donors


def test():
    torch.set_num_threads(2)
    n=12
    # Imbalanced book clusters deliberately force donor reuse.
    books=torch.tensor([0]*9+[1]*2+[2])
    data={'target':torch.arange(n)+1000,
          'hidden':torch.arange(n*3).view(n,3).float(),
          'base':torch.arange(n).float(),
          'near_features':torch.arange(n*60).view(n,60).float(),
          'far_features':torch.arange(n*32).view(n,32).float(),
          'memory_modal_ids':torch.arange(n*12).view(n,12),
          'memory_modal_mask':torch.arange(n*12).view(n,12)%2==0,
          'book_ids':books}
    for width,key in [(32,'word_ids'),(128,'byte_ids'),
                      (8,'sentence_ids'),(8,'vertex_ids')]:
        data[key]=torch.arange(n*width).view(n,width)
    ctrl,donors=unrelated_book_context(data,725)
    assert torch.all(books[donors]!=books)
    assert torch.equal(donors,different_book_donors(books,725))
    assert torch.equal(data['target'],ctrl['target'])
    assert torch.equal(data['hidden'],ctrl['hidden'])
    assert torch.equal(data['base'],ctrl['base'])
    assert torch.equal(data['near_features'],ctrl['near_features'])
    assert torch.equal(data['far_features'],ctrl['far_features'])
    for k in ('word_ids','byte_ids','sentence_ids','vertex_ids'):
        assert torch.equal(ctrl[k],data[k][donors]),k
    assert torch.equal(ctrl['diamond_memory_features'],
        torch.cat((data['near_features'],data['far_features']),dim=-1)[donors])
    assert torch.equal(ctrl['diamond_modal_ids'],data['memory_modal_ids'][donors])
    assert torch.equal(ctrl['diamond_modal_mask'],data['memory_modal_mask'][donors])
    assert not torch.equal(donors,different_book_donors(books,726))
    for labels in [torch.zeros(n),torch.tensor([1])]:
        try:different_book_donors(labels)
        except ValueError:pass
        else:raise AssertionError('single-book history comparison leaked')
    # A supplied explicit source has precedence and is swapped as one unit.
    data['diamond_memory_features']=torch.randn(n,92)
    data['diamond_modal_ids']=torch.full((n,12),17)
    data['diamond_modal_mask']=torch.ones(n,12,dtype=torch.bool)
    ctrl,donors=unrelated_book_context(data,725)
    assert torch.equal(ctrl['diamond_memory_features'],data['diamond_memory_features'][donors])
    assert torch.equal(ctrl['diamond_modal_ids'],data['diamond_modal_ids'][donors])
    assert torch.equal(ctrl['diamond_modal_mask'],data['diamond_modal_mask'][donors])
    print('PASS: all Diamond donors strictly foreign-book; inherited predictions and labels unchanged')
    print('PASS: atomically swapped 32/128/8/8 histories + 92 memory + 12 causal modes')
    print('PASS: deterministic imbalanced-book sampling; single-book control fails closed')


if __name__=='__main__':test()
