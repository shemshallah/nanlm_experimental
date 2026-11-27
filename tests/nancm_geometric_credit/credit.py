import torch

def laws(logp,anchor,neural_share,copy,word_tilt,class_tilt,credit):
    p=logp.double().exp();p=p/p.sum(-1,keepdim=True)
    a=anchor+neural_share[:,None]*p
    def tilt(v):
        b=a*v;return b/b.sum(-1,keepdim=True)
    b=credit[:,0,None]*a+credit[:,1,None]*copy+credit[:,2,None]*tilt(word_tilt)+credit[:,3,None]*tilt(class_tilt)
    return a,b

def literal(logp,anchor,neural_share,copy,word_tilt,class_tilt,credit,event_vector,surface,raw,weights):
    a,b=laws(logp,anchor,neural_share,copy,word_tilt,class_tilt,credit)
    pa=(a*event_vector).sum(-1)*surface;pb=(b*event_vector).sum(-1)*surface
    return weights[0]*raw+weights[1]*pa+weights[2]*pb

def sequence(likelihood,initial,prior,hazard):
    # initial predictive weights derived from already observed teacher prefix.
    w=initial;loss=likelihood.new_zeros(())
    for q in likelihood:
        x=w*q;z=x.sum();loss=loss-z.log()
        w=(1-hazard)*x/z+hazard*prior
    return loss
