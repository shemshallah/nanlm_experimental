"""v98 transactional event decoding; retain model law and retry repeated trigrams.

A rejected candidate never updates the accepted recurrent state or conceptual
posterior. The sampler is changed, not the model's underlying probabilities.
"""
import re,math
import numpy as np

def words98(raw):
    return re.findall(rb'[A-Za-z]+',raw.lower())

def repeats(history,candidate):
    old=words98(history);new=words98(history+candidate)
    # History ends on a complete event except at final generation cap.
    if len(new)<=len(old):return False
    seen=set(zip(old,old[1:],old[2:]))
    for i in range(max(2,len(old)),len(new)):
        tri=tuple(new[i-2:i+1])
        if tri in seen:return True
        seen.add(tri)
    return False

def choose_event(propose,history,temperature,attempts):
    first=propose(temperature)
    if not attempts or not repeats(history,first['raw']):
        return first,0,False
    # For greedy decoding, retries are explicitly stochastic, seeded.
    # Rejection is on an entire candidate event rather than forcing a
    # nonsensical suffix by blocking a delimiter after a word is spelled.
    retry_temp=max(.8,temperature)
    for j in range(attempts):
        cand=propose(retry_temp)
        if not repeats(history,cand['raw']):return cand,j+1,False
    return first,attempts,True

def selftest():
    assert repeats(b'the sky and the sky ',b'and ')
    assert not repeats(b'the sky and the sky ',b'is ')
    assert not repeats(b'the sky and ',b'the ')
    assert not repeats(b'the sky and the sky ',b'.')
    history=b'the sky and the sky '
    items=iter([{'raw':b'and ','state':1},{'raw':b'and ','state':2},{'raw':b'is ','state':3}])
    calls=[]
    def propose(t):calls.append(t);return next(items)
    out,n,fallback=choose_event(propose,history,0.,4)
    assert out['state']==3 and n==2 and not fallback and calls==[0.,.8,.8]
    out,n,fallback=choose_event(lambda t:{'raw':b'and ','state':4},history,0.,2)
    assert out['state']==4 and n==2 and fallback
    out,n,fallback=choose_event(lambda t:{'raw':b'and ','state':5},history,0.,0)
    assert n==0 and not fallback and out['state']==5
    print('v98 event-level repeat detection, deterministic retry accounting and fallback passed',flush=True)
if __name__=='__main__':selftest()
