import tempfile,pathlib,numpy as np
from discount import Discount
from clean_split import clean
rng=np.random.default_rng(57);docs=[rng.integers(1,10,128) for _ in range(5)];prior=np.ones(10)/10;m=Discount(docs,prior,5)
err=0.;checks=0
for d in docs:
 for i in range(1,len(d)):
  for n in (0,1,3,5):
   p=m.query(d[:i],n);err=max(err,abs(p.sum()-1));checks+=1
from collections import Counter
raw={n:Counter(tuple(d[i:i+n+1]) for d in docs for i in range(len(d)-n)) for n in range(1,6)}
base=np.bincount([k[-1] for k in raw[1]],minlength=10);base=.99*base/base.sum()+.01*prior
maxdiff=0
for d in docs:
 for i in range(5,20):
  h=d[:i];p=base.copy()
  for n in range(1,6):
   count=Counter()
   if n==5:
    for k,v in raw[n].items():
     if k[:-1]==tuple(h[-n:]):count[k[-1]]+=v
   else:
    for k in raw[n+1]:
     if k[1:-1]==tuple(h[-n:]):count[k[-1]]+=1
   if count:
    z=sum(count.values());p*=.75*len(count)/z
    for k,v in count.items():p[k]+=(v-.75)/z
  maxdiff=max(maxdiff,float(abs(p-m.query(h)).max()))
assert err<1e-12 and maxdiff<1e-12
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp)
 def file(name,words):p=root/name;p.write_text(' '.join(words));return name,p
 r=file('r',[f'r{i}' for i in range(160)])
 bad=file('bad',[f'x{i}' for i in range(100)]+[f'r{i}' for i in range(32)])
 good=file('good',[f'g{i}' for i in range(160)])
 dup=file('dup',[f'd{i}' for i in range(100)]+[f'g{i}' for i in range(32)])
 ok=file('ok',[f'o{i}' for i in range(160)])
 sel,manifest=clean([(*r,160)],[('cal',1,[bad,good]),('dev',1,[dup,ok])],lambda p,n:p.read_text().split()[:n])
 assert sel['cal']==[good] and sel['dev']==[ok]
 assert [x['reason'] for x in manifest['rejections']]==['reference_shingle','selected_shingle']
print(dict(normalization_checks=checks,max_mass_error=float(err),independent_direct_max_error=maxdiff,overlap_rejections=2))
