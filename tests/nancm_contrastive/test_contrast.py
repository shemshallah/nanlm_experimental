import numpy as np
from contrast import donors
p=np.array([.02,.48,.2,.1,.1,.1]);c=np.array([0,1,1,2,2,2]);parent=p.copy()
a,f,v=donors([3,4,3,4]*20,parent,p,c)
b,_,_=donors([1,2,1,2]*20,parent,p,c)
assert all(abs(x.sum()-1)<1e-14 for x in a+b)
assert a[2][5]>parent[5] and a[2][5]>b[2][5]
assert a[0][5]==0
assert all(x is parent for x in donors([],parent,p,c)[0])
print('mass error',max(abs(x.sum()-1) for x in a+b),'novel same-class',parent[5],a[2][5],b[2][5],'features',f[0].tolist())
