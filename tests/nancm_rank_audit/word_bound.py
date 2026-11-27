"""Conservative maximum-completion bound for normalized open word laws."""
import bisect, math
import numpy as np

class CompletionBound:
    def __init__(self, word):
        self.word=word;self.cache={}
        if hasattr(word,'outside') and hasattr(word,'p') and hasattr(word,'unknown'):
            self.kind='lift';self.words=word.words;self.p=word.p
        elif all(hasattr(word,k) for k in ('parent','child','g')):
            self.kind='blend';self.parent=CompletionBound(word.parent);self.child=CompletionBound(word.child)
        else:self.kind='generic'
    def __call__(self,prefix):
        if prefix in self.cache:return self.cache[prefix]
        if self.kind=='lift':
            a=bisect.bisect_left(self.words,prefix);b=bisect.bisect_left(self.words,prefix+'{')
            known=float(np.max(self.p[a:b])) if b>a else 0.
            outside=self.word.unknown*self.word.outside.mass(prefix)
            bound=max(known,outside)
        elif self.kind=='blend':bound=(1-self.word.g)*self.parent(prefix)+self.word.g*self.child(prefix)
        else:bound=self.word.mass(prefix)
        bound=math.nextafter(float(bound),math.inf)+1e-15*max(1.,float(bound))
        if len(self.cache)<32768:self.cache[prefix]=bound
        return bound

def rank(event,k=5,budget=2000,tight=True):
    import heapq
    from folded_event import LOWER
    bound=CompletionBound(event.word)
    heap=[(-1.,b'')];best=[];seen=0
    while heap and seen<budget:
        if len(best)>=k and -heap[0][0]<best[0][0]:break
        _,p=heapq.heappop(heap);seen+=1
        if p:
            item=(event.complete(p),p)
            if len(best)<k:heapq.heappush(best,item)
            elif item>best[0]:heapq.heapreplace(best,item)
        if len(p)<event.cap-1:
            for c in LOWER:
                h=p+bytes([c]);s=h.decode('ascii')
                if tight:
                    m=(1-event.weight)*math.fsum(event.states(h).values())+event.weight*bound(s)
                    m=math.nextafter(m,math.inf)+1e-15
                else:m=event.prefix_mass(h)
                if m>0 and (len(best)<k or m>=best[0][0]):heapq.heappush(heap,(-m,h))
    cert=not heap or (len(best)>=k and -heap[0][0]<best[0][0])
    return [p.decode('ascii') for _,p in sorted(best,reverse=True)],bool(cert),seen
