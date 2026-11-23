#!/usr/bin/env python3
import random,math
from macro_code177 import MacroByteModel
from range_code176 import encode as byte_encode,decode as byte_decode

def main():
    rng=random.Random(177)
    calls=0;roundtrips=0;probability=0;macros=0;quant_bits=[];oracle_bits=[]
    candidates=[b'the ',b'their ',b'there ',b'then ',b'they ']
    # Context-dependent 256-way byte predictor. All symbols have positive mass.
    # Reproducible, nonuniform and sensitive to causally observed prefix only.
    def p(s):
        nonlocal calls
        calls+=1
        h=(sum((i+1)*b for i,b in enumerate(s))+3*len(s))%127
        vals=[1+((j*23+h*19)%(1+((j%13)+1)*11)) for j in range(256)]
        for b in b'theriy ':
            vals[b]+=250+(h%17)*5
        z=sum(vals);return [v/z for v in vals]
    mod=MacroByteModel(candidates,p)
    assert abs(sum(mod.events)+mod.escape-1.)<1e-12
    tests=candidates+[b'hello ',b'heads ',b'elephant ',b'zzzzz ',b'theory ',b'a ',b'x ',b'ab ',b'thek ']
    tests+=[bytes(rng.choices(b'abcdefghijklmnopqrstuvwxyz',k=rng.randrange(1,20)))+b' ' for _ in range(170)]
    for word in tests:
        packed,bits=mod.encode(word)
        assert mod.decode(packed,bits)==word,(word,bits)
        plain,pbits=byte_encode(word,p)
        assert byte_decode(plain,pbits,len(word),p)==word
        original=mod.oracle_probability(word);macro=mod.macro_probability(word)
        if not math.isclose(original,macro,rel_tol=2e-10,abs_tol=1e-20):raise AssertionError((word,original,macro))
        roundtrips+=1;probability+=1;macros+=word in candidates
        oracle_bits.append(pbits);quant_bits.append(bits)
    # Check full 256-way escape distributions including shared prefixes.
    for pre in [b'',b't',b'th',b'the',b'thei',b'ther',b'z']:
        q=mod.conditional(pre)
        assert len(q)==256 and all(v>=0 for v in q) and abs(sum(q)-1)<1e-9
    print({'status':'PASS','lossless_roundtrips':roundtrips,'exact_law_examples':probability,
           'topfive_macro_cases':macros,'root_macro_events':len(candidates)+1,
           'orig_total_coding_bits':sum(oracle_bits),'macro_total_coding_bits':sum(quant_bits),
           'note':'Ideal probability law is preserved; finite precision coder lengths need not match.'},flush=True)
if __name__=='__main__':main()