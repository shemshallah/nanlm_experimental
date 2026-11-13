#!/usr/bin/env python3
"""Latest-only FA2 R26 unified recovery archive, SHA-256 verified.
Original R19 accepted ancestor is a separate required dependency.
"""
import hashlib,json,pathlib,zipfile,argparse,os

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for z in iter(lambda:f.read(4<<20),b''):h.update(z)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--output',default='/mnt/data/FA2_R26_UNIFIED_LATEST_RECOVERY.zip')
    a=p.parse_args();root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.output)
    report=json.loads((root/'reports/r26_cpu_trial.json').read_text())
    if not(0<report['selected_updates']<=report['attempted_updates']) or report['optimizer_groups']!=11 or report['optimizer_states']!=137:
        raise ValueError('not verified R26 selection')
    files=(list(sorted((root/'source').glob('*.py')))+
        [root/'reports/fa2_r26_unified_candidate.pth',
         root/'reports/fa2_2_r21_gate_candidate.pth',root/'reports/r22_multiscale_candidate.pth',
         root/'reports/r26_cpu_trial.json',root/'reports/r26_lineage_audit.json',
         root/'reports/r26_longdistance_dev_970.json',root/'reports/r26_longdistance_dev_1444.json',
         root/'logs/r26_cpu100.log',root/'logs/r26_long970.log',root/'logs/r26_long1444.log',
         root/'R26_RECOVERY_README.md',root/'requirements.txt'])
    if not all(f.is_file() for f in files):
        raise FileNotFoundError([str(f) for f in files if not f.is_file()])
    if len(files)!=len(set(files)):raise ValueError('duplicate member')
    record={}
    for f in files:
        rel=f.relative_to(root).as_posix()
        record[rel]={'sha256':sha(f),'bytes':f.stat().st_size}
    if record['reports/fa2_r26_unified_candidate.pth']['sha256']!=report['candidate_sha256']:
        raise ValueError('selected checkpoint changed')
    man={'format':'FA2_R26_UNIFIED_RECOVERY_V1','selected_r26_updates':report['selected_updates'],
         'inherited_r25_step':120,'optimizer_groups':11,'optimizer_states':137,
         'accepted_model_promoted':False,'r18_parent_sha256':report['inherited_weights'] and
                 'dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f',
         'r25_parent_sha256':report['parent_r25_sha256'],
         'checkpoint_sha256':report['candidate_sha256'],
         'separate_r19_parent_required':True,'file_sha256':record}
    sums=''.join(f'{item["sha256"]}  {name}\n' for name,item in sorted(record.items()))
    out.parent.mkdir(exist_ok=True,parents=True);tmp=out.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for file in files:z.write(file,file.relative_to(root).as_posix())
        z.writestr('reports/r26_latest_manifest.json',json.dumps(man,sort_keys=True,indent=2)+'\n')
        z.writestr('SHA256SUMS.txt',sums)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)+2
        for name,item in record.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:
                raise ValueError('member SHA mismatch: '+name)
    tmp.replace(out)
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,
        'sha256':sha(out),'members':len(record)+2,
        'hashed_data_members':len(record),'checkpoint_sha256':report['candidate_sha256']}))
if __name__=='__main__':main()
