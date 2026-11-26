"""Target documents for a learned readout must be absent from its retrieval index."""
import hashlib

def cohorts(books,tr,calbooks,extra):
    indexed=tr+extra
    idx={i for i,p in indexed};cal={i for i,p in calbooks}
    assert not idx&cal and len(idx)==len(indexed)
    spare=[(i,p) for i,p in books if i not in idx|cal]
    if len(spare)>=8:
        n=min(12,len(spare)//2)
        primary=spare[:n];secondary=spare[n:2*n];select=calbooks
        source='unused_train_bucket'
    else:
        assert len(calbooks)>=12
        primary=calbooks[:4];secondary=calbooks[4:8];select=calbooks[8:12]
        source='split_inherited_calibration'
    groups={'indexed':indexed,'fit_primary':primary,'fit_secondary':secondary,'selection':select}
    seen={};hashes={};manifest={}
    for name,files in groups.items():
        manifest[name]=[]
        for bid,p in files:
            assert bid not in seen,(bid,seen.get(bid),name)
            seen[bid]=name
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest not in hashes or hashes[digest]==name,('identical content across cohorts',bid,hashes.get(digest),name)
            hashes[digest]=name;manifest[name].append({'id':bid,'sha256':digest})
    return primary,secondary,select,{'source':source,'manifest':manifest,'unindexed_fit':True,'inherited_calibration_reused':True,'partial_overlap_not_checked':True}
