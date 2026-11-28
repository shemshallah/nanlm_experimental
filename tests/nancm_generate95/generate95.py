"""v95: genuine autoregressive generation using recovered NaNCM laws.

No answers, completion tokens or reference strings enter the predictor.
Prompt bytes are observed, then each newly sampled byte is fed back.
Output is saved verbatim, including undesirable control bytes and repetition.
"""
import time,re,math,collections
import numpy as np
import torch
import folded_event as FE

PROMPTS=[
    ('greeting','User: Hello! How are you?\nAssistant: '),
    ('instruction','User: Explain why the sky is blue in one sentence.\nAssistant: '),
    ('arithmetic','User: What is two plus three?\nAssistant: '),
    ('memory','User: My favorite color is green.\nAssistant: I will remember that.\nUser: What is my favorite color?\nAssistant: '),
    ('completion','Once upon a time, in a small village, '),
]
def events(prompt):
    start=0
    for i,b in enumerate(prompt):
        if not FE.IS_LETTER[b] or i-start==63:
            yield prompt[:start],prompt[start:i+1];start=i+1
    if start<len(prompt):yield prompt[:start],prompt[start:]

def draw(q,rng,temp,top_p):
    q=np.asarray(q,np.float64)
    assert q.shape==(256,) and q.min()>=0 and abs(q.sum()-1)<1e-8
    if temp==0:return int(np.argmax(q))
    log=np.log(np.maximum(q,1e-300))/temp;log-=log.max()
    p=np.exp(log);p/=p.sum()
    order=np.argsort(-p,kind='stable');n=int(np.searchsorted(np.cumsum(p[order]),top_p))+1
    order=order[:min(n,256)];mass=p[order];mass/=mass.sum()
    return int(rng.choice(order,p=mass))

def rollout(factory,prompt,limit,seed,temp,top_p):
    rng=np.random.default_rng(seed);t=time.perf_counter()
    src=prompt.encode('utf-8');root=b'';p=b'';e=None;count=0;warmup=0
    # Some inherited lexical retrieval routines require observed history.
    # Use the first complete prompt word as the initial observed root;
    # all remaining prompt events replay through the model.
    for prefix,event in events(src):
        assert prefix==root
        if not re.search(rb'[A-Za-z]+',root):
            root+=event;warmup+=len(event);continue
        e=factory.build(root);p=event
        if e.done(p):
            factory.observe(e,p);root+=p;p=b'';e=None
    assert root+p==src
    out=bytearray();logloss=0.;stop='byte_limit'
    for _ in range(limit):
        if e is None:e=factory.build(root)
        q=e.next(p);b=draw(q,rng,temp,top_p)
        logloss-=math.log2(max(float(q[b]),1e-300))
        p+=bytes([b]);out.append(b)
        if e.done(p):
            factory.observe(e,p);root+=p;p=b'';e=None;count+=1
        if out.endswith(b'\nUser:') or out.endswith(b'\nAssistant:'):
            stop='role_boundary';break
    assert root+p==src+bytes(out)
    raw=bytes(out);text=raw.decode('utf-8',errors='replace')
    words=re.findall(r'[A-Za-z]+',text.lower());tri=list(zip(words,words[1:],words[2:]))
    return {'prompt':prompt,'continuation':text,'continuation_hex':raw.hex(),
            'bytes':len(raw),'words':len(words),'completed_events':count,
            'prompt_warmup_bytes_skipped':warmup,'stop_reason':stop,
            'printable_ascii_fraction':sum(32<=b<127 or b in (9,10,13) for b in raw)/max(1,len(raw)),
            'repeated_trigram_fraction':1-len(set(tri))/len(tri) if tri else 0.,
            'sampled_mean_bits':logloss/max(1,len(raw)),
            'elapsed_seconds':time.perf_counter()-t,'temperature':temp,'top_p':top_p,'seed':seed}

def selftest():
    class E:
        def next(self,p):
            q=np.zeros(256);q[ord('a') if not p else ord(' ')]=1.;return q
        def done(self,p):return bool(p and (not FE.IS_LETTER[p[-1]] or len(p)>=64))
    class F:
        def __init__(self):self.roots=[];self.observed=[]
        def build(self,root):self.roots.append(root);return E()
        def observe(self,e,p):self.observed.append(p)
    a=F();x=rollout(a,'User: hi\nAssistant: ',8,95,0.,1.)
    assert x['continuation']=='a a a a ' and a.roots[-1].endswith(b'a a a ')
    assert x['completed_events']==4
    assert x['continuation_hex']==b'a a a a '.hex()
    print('v95 prompt replay, actual generated-byte feedback, raw output and deterministic sampling passed',flush=True)
if __name__=='__main__':selftest()
