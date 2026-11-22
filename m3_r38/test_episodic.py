import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from m3_r38.episodic import CausalSuccessors,M3R38Predictor,corrected_logp,nll_grad,GROUPS
from m3_r38.optimizer import ContinuationAdam
from m3_r38.experiment import evaluate,fit,strict_gate


class FrozenParent:
    def __init__(self):
        self.history=[]
        self.lp=np.log(np.full(256,1./256,dtype=np.float64))
    def observe(self,b): self.history.append(b)
    def reset(self): self.history=[]
    def predict_logp(self): return self.lp.copy()


class R38Regression(unittest.TestCase):
    def test_zero_gain_bit_exact_frozen_parent_and_reset(self):
        base=FrozenParent()
        model=M3R38Predictor(base)
        for x in b'abracadabra abracadabra'*2:
            np.testing.assert_array_equal(model.predict_logp(),base.predict_logp())
            model.observe(x)
        model.reset()
        self.assertEqual(model.memory.observed,0)
        np.testing.assert_array_equal(model.predict_logp(),base.predict_logp())

    def test_future_target_does_not_change_prediction(self):
        model=M3R38Predictor(FrozenParent(),np.linspace(-0.2,0.4,GROUPS))
        for b in b'a long repeated phrase a long repeated phrase ': model.observe(b)
        n=model.memory.observed
        a=model.predict_logp()
        np.testing.assert_array_equal(model.predict_logp(),a)
        self.assertEqual(model.memory.observed,n)
        self.assertAlmostEqual(float(np.exp(a).sum()),1.0,places=12)

    def test_suffix_addresses_are_context_dependent(self):
        m=CausalSuccessors(orders=(2,),capacity=5,max_history=4)
        for c in b'abXabYabX':m.observe(c)
        m.history=bytearray(b'ab')
        f=m.features(np.log(np.full(256,1/256)))
        self.assertGreater(np.abs(f).sum(),0.)
        self.assertGreater(f[0,ord('X')],f[0,ord('Z')])
        m.reset()
        self.assertFalse(m.tables)

    def test_gradient_finite_difference(self):
        rng=np.random.default_rng(17)
        p=rng.random(256); p/=p.sum(); base=np.log(p)
        f=rng.normal(0,.1,size=(GROUPS,256))
        theta=rng.normal(0,.1,size=GROUPS)
        lp=corrected_logp(base,f,theta)
        grad=nll_grad(lp,f,91)
        for idx in (0,4,11):
            t=theta.copy(); t[idx]+=1e-6
            plus=-corrected_logp(base,f,t)[91]
            t[idx]-=2e-6
            minus=-corrected_logp(base,f,t)[91]
            self.assertAlmostEqual(grad[idx],(plus-minus)/2e-6,places=7)

    def test_optimizer_preserves_42_prefix_and_sha(self):
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root)/'r37.npz'
            kwargs={k: np.arange(42,dtype=np.float64) for k in ('theta','adam_m','adam_v')}
            kwargs['adam_steps']=np.arange(42,dtype=np.int64)
            kwargs['frozen_anchor']=np.arange(100,dtype=np.uint8)
            np.savez(parent,**kwargs)
            h=hashlib.sha256(parent.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): ContinuationAdam(parent, '0'*64)
            opt=ContinuationAdam(parent,h)
            opt.update(np.ones(GROUPS))
            path=Path(root)/'r38.npz'
            opt.save(path,fit_indices=(0,16),gate_indices=(16,20),dev_indices=(20,32))
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(z['theta'].shape,(54,))
                for key in ('theta','adam_m','adam_v','adam_steps'):
                    np.testing.assert_array_equal(z[key][:42],kwargs[key])
                np.testing.assert_array_equal(z['frozen_anchor'],kwargs['frozen_anchor'])

    def test_reranking_possible(self):
        base=np.log(np.ones(256)/256.)
        f=np.zeros((GROUPS,256)); f[0,65]=10.
        theta=np.zeros(GROUPS); theta[0]=1
        x=corrected_logp(base,f,theta)
        self.assertEqual(int(np.argmax(x)),65)
        self.assertAlmostEqual(float(np.exp(x).sum()),1.,places=12)

    def test_causal_wrong_successor_and_unrelated_nulls(self):
        books=[b"abC"*160,b"xyZ"*160]
        theta=np.zeros(GROUPS); theta[0]=1.
        out=evaluate(books,FrozenParent,theta,warmup=16)
        self.assertLess(out["r38"]["bpb"],out["wrong_successors"]["bpb"])
        self.assertLess(out["r38"]["bpb"],out["unrelated_history"]["bpb"])
        self.assertEqual(len(out["r38"]["whole_book_bootstrap_95pct_bits"]),2)

    def test_scoring_is_book_disjoint(self):
        sample=[b'abcdefghiabcdefghi'*12,b'zzzabcdefghiabcdefghi'*10]
        z=evaluate(sample,FrozenParent,np.zeros(GROUPS),warmup=16)
        self.assertEqual(z['books'],2)
        self.assertAlmostEqual(z['r38']['bpb'],8.0,places=10)
        self.assertAlmostEqual(z['r38']['gain_bpb'],0.,places=10)
        self.assertFalse(all(strict_gate(z).values()))

    def test_direct_real_training_reduces_synthetic_loss(self):
        class Opt:
            theta=np.zeros(GROUPS)
            steps=0
            def update(self,g):
                self.steps+=1
                self.theta -= .15*np.clip(g,-2,2)
        opt=Opt()
        # Two independent repeated-pattern books: memory should use document
        # locality while the frozen parent stays uniform.
        books=[(b'abC'*400),(b'xyZ'*400)]
        before=evaluate(books,FrozenParent,opt.theta,warmup=32)
        metrics=fit(books,FrozenParent,opt,warmup=32,batch=32)
        after=evaluate(books,FrozenParent,opt.theta,warmup=32)
        self.assertGreater(metrics['updates'],20)
        self.assertLess(after['r38']['bpb'],before['r38']['bpb']-.5)

if __name__=='__main__': unittest.main()