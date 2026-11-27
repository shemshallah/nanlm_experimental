"""Reject exact shared 16-word passages within fixed evaluation prefixes."""
import hashlib

def shingle_set(words,n):
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}

def clean(reference,pools,load,n=16,limit=1200):
    wanted=set();cand={};seen_ids=set();reject=[]
    for name,count,files in pools:
        for bid,p in files:
            key=(name,bid)
            assert key not in cand
            words=load(p,limit);s=shingle_set(words,n);digest=hashlib.sha256(p.read_bytes()).hexdigest()
            cand[key]=(p,s,digest,len(words));wanted.update(s)
    ref_hits=set();ref_hashes=set()
    for bid,p,cap in reference:
        ref_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest());words=load(p,cap)
        for i in range(len(words)-n+1):
            k=tuple(words[i:i+n])
            if k in wanted:ref_hits.add(k)
    selected={};selected_shingles=set();hashes=set(ref_hashes);manifest={}
    for name,count,files in pools:
        selected[name]=[];manifest[name]=[]
        for bid,p in files:
            _,s,digest,length=cand[name,bid];reason=None
            if length<112:reason='short'
            elif bid in seen_ids or digest in hashes:reason='id_or_exact_file'
            elif s&ref_hits:reason='reference_shingle'
            elif s&selected_shingles:reason='selected_shingle'
            if reason:reject.append({'group':name,'id':bid,'reason':reason});continue
            selected[name].append((bid,p));manifest[name].append({'id':bid,'sha256':digest,'words':length});seen_ids.add(bid);hashes.add(digest);selected_shingles.update(s)
            if len(selected[name])==count:break
        if len(selected[name])<count:raise ValueError((name,'insufficient clean candidates',len(selected[name]),count))
    return selected,{'shingle_words':n,'scope':'reference indexed prefixes versus first 1200 words of selected documents; cross-cohort and within-cohort exact shingle exclusion','manifest':manifest,'rejections':reject,'reference_matching_unique_shingles':len(ref_hits)}
