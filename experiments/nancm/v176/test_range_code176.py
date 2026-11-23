#!/usr/bin/env python3
"""Exact context-dependent byte arithmetic coding, including top5 trie model."""
import math,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from range_code176 import encode,decode
from third_code176 import TopKTrie,normalize

def oracle(history):
 base=[0.001]*256
 key=history[-1] if history else 32
 base[ord('a')]=1.2 if key==32 else .01
 base[ord('b')]=1.2 if key==ord('a') else .01
 base[32]=1.2 if key==ord('b') else .01
 base[key]=.3
 return normalize(base)

def run():
 rng=random.Random(176);seqs=[b'',b'abc',b'aba baa cab ',b' ' * 500,bytes(rng.randrange(256) for _ in range(1200))]
 for seq in seqs:
  blob,n=encode(seq,oracle,b'header ')
  got=decode(blob,n,len(seq),oracle,b'header ')
  assert got==seq,(len(seq),got[:20],seq[:20])
 # Distinctive low-loss structured sequence compresses better than raw bytes.
 seq=b'ab '*300
 blob,bits=encode(seq,oracle)
 assert decode(blob,bits,len(seq),oracle)==seq
 assert bits<len(seq)*8,('not a compression improvement',bits,len(seq)*8)
 # The speculative trie can itself define a reversible target only if both
 # encoder and decoder use the SAME normalized Q; native v170 remains P.
 trie=TopKTrie([('the',.3),('then',.3),('there',.4)],escape=.18)
 def hybrid(h):
  prefix=h.rsplit(b' ',1)[-1]
  return trie.proposal(prefix,oracle(h))
 seq=b'the there then the '
 packed,nbits=encode(seq,hybrid,b'')
 assert decode(packed,nbits,len(seq),hybrid,b'')==seq
 print('PASS bit-exact 32-bit range coding, random 1200-byte payload, adaptive verifier, normalized top5 trie, structured compression')

if __name__=='__main__':run()