#!/usr/bin/env python3
"""Monte Carlo + exhaustive probability tests: exact speculative third coding."""
import math,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from third_code176 import *


def test():
 rng=random.Random(176)
 for k in range(15):
  p=normalize([rng.gammavariate(.8,1) for _ in range(256)])
  q=normalize([rng.gammavariate(.5,1) for _ in range(256)])
  a=sum(min(x,y) for x,y in zip(p,q));assert abs(a-analytic_acceptance(p,q))<1e-12
  r=residual(p,q);r_mass=1-a
  # Exact algebra: output P(b) = min(P,Q) + (1-A)R(b).
  for b in range(256):assert abs(min(p[b],q[b])+r_mass*r[b]-p[b])<2e-12
 t=TopKTrie([('there',.3),('their',.2),('the',.1),('there',.05),('then',.01)],escape=.07)
 p=normalize([1.]*256)
 for prefix in (b'',b't',b'th',b'the',b'ther',b'there',b'notincandidate'):
  q=t.proposal(prefix,p);assert len(q)==256 and abs(sum(q)-1)<1e-12
 assert t.match('their') and not t.match('every')
 assert t.proposal(b'badprefix',p)==p
 # Sample p once per byte; independent synthetic oracle will preserve its law.
 target=normalize([.5 if i==ord('t') else .5/255 for i in range(256)])
 draft=t.proposal(b'',target)
 n=90000;counts=[0]*256;accept=0
 for _ in range(n):
  i,y=speculative_one(target,draft,rng);counts[i]+=1;accept+=y
 assert abs(counts[ord('t')]/n-.5)<.008
 assert abs(accept/n-analytic_acceptance(target,draft))<.009
 def oracle(s):
  return normalize([.8 if j==ord('t') else .2/255 for j in range(256)])
 out,nacc,rejected=speculative_word(t,b'history ',oracle,rng,maxbytes=10)
 assert 1<=len(out)<=10 and 0<=nacc<=len(out)
 assert all(0<=r<=1 for r in fixed_draft_accept_probabilities(t,b'history ','there',oracle))
 beam,info=prefix_beam(t,b'history ',oracle,width=5,maxdepth=8)
 assert info['oracle_calls']==info['shared_prefixes']>0
 for bad in ([],[('bad word',1)], [('ABC',1)], [('abc',-1)]):
  try:TopKTrie(bad);raise AssertionError('accepted invalid')
  except ValueError:pass
 print('PASS: exhaustive 15x256 exact speculative residuals, 90K-sample target law, trie normalization, prefix cache, malformed inputs')

if __name__=='__main__':test()