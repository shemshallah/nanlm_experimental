#!/usr/bin/env python3
"""R7 byte-reversible wordpiece tokenizer (train-only BPE, explicit whitespace bytes)."""
import argparse,hashlib,json,pathlib,random,re,collections
import numpy as np
import sentencepiece as spm

# 0..255 reserved for raw bytes in whitespace spans, SPM pieces start at 256.
TOKEN_PATTERN=re.compile(rb'\s+|\S+')

def encode_bytes(sp, src: bytes):
    out=[]
    for m in TOKEN_PATTERN.finditer(src):
        part=m.group()
        if part[:1] in b' \t\r\n\v\f':
            # Avoid classifying other non-ascii Unicode whitespace as byte spans
            if all(c in (9,10,11,12,13,32) for c in part):
                out.extend(int(c) for c in part);continue
        try:
            word=part.decode('utf8','strict')
            ids=sp.encode(word,out_type=int)
            if sp.decode(ids).encode('utf8')!=part:raise ValueError('non-reversible SentencePiece token')
            out.extend(256+i for i in ids)
        except (UnicodeDecodeError,ValueError):
            out.extend(int(c) for c in part)
    return out

def decode_bytes(sp, ids):
    data=bytearray();pieces=[]
    def flush():
        if pieces:data.extend(sp.decode(pieces).encode('utf8'));pieces.clear()
    for x in ids:
        x=int(x)
        if x<256:flush();data.append(x)
        else:pieces.append(x-256)
    flush();return bytes(data)

def build(root:pathlib.Path,vocab_size:int=4096,train_lines:int=24000):
    corpus=root/'corpus';out=root/'tokenizer';out.mkdir(exist_ok=True)
    rng=random.Random(7331)
    # Stream a deterministic reservoir from the TRAIN split only.
    reservoir=[];total=0
    with (corpus/'train.txt').open(encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line:continue
            total+=1
            if len(reservoir)<train_lines:reservoir.append(line[:1000])
            else:
                j=rng.randrange(total)
                if j<train_lines:reservoir[j]=line[:1000]
    sample=out/'bpe_training_subset.txt';sample.write_text('\n'.join(reservoir)+'\n',encoding='utf8')
    prefix=out/'gutenberg_r7'
    spm.SentencePieceTrainer.Train(input=str(sample),model_prefix=str(prefix),vocab_size=vocab_size,model_type='bpe',character_coverage=1.0,byte_fallback=True,normalization_rule_name='identity',remove_extra_whitespaces=False,add_dummy_prefix=False,split_by_whitespace=False,split_by_unicode_script=False,hard_vocab_limit=False,max_sentence_length=4096,shuffle_input_sentence=False,unk_id=0,bos_id=-1,eos_id=-1,pad_id=-1)
    sp=spm.SentencePieceProcessor(model_file=str(prefix)+'.model')
    stats={};verified=0
    for split in ('train','dev','test'):
        path=corpus/f'{split}.txt'; ids=[];h=hashlib.sha256();h2=hashlib.sha256()
        with path.open('rb') as f:
            for line in f:
                h.update(line); segment=encode_bytes(sp,line)
                restored=decode_bytes(sp,segment)
                if restored!=line:raise ValueError(f'roundtrip failure {split} line {verified}')
                ids.extend(segment);h2.update(restored);verified+=1
        arr=np.asarray(ids,dtype=np.int32);arr.tofile(out/f'{split}.subword.i32')
        if h.digest()!=h2.digest():raise ValueError('hash mismatch '+split)
        stats[split]={'tokens':int(len(arr)),'sha256_reconstructed':h2.hexdigest()}
    info={'token_count':sp.get_piece_size()+256,'sentencepiece_vocab':sp.get_piece_size(),'raw_byte_tokens':256,'reversible_lines':verified,'train_lines_sampled':len(reservoir),'train_source_lines_seen':total,'splits':stats,'trained_only_on':'corpus/train.txt'}
    (out/'tokenizer_audit.json').write_text(json.dumps(info,indent=2));return info
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--vocab',type=int,default=4096);p.add_argument('--train-lines',type=int,default=24000)
    a=p.parse_args();print(json.dumps(build(a.root,a.vocab,a.train_lines),indent=2))
