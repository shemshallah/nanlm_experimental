#!/usr/bin/env python3
"""Create/delete temporary fake books to exercise end-to-end split/causality audit.

These texts are deliberately synthetic fixtures, NOT independently heldout books.
"""
import argparse,json,pathlib,tempfile
from prepare_document_corpus import prepare
from fa1_r716_newbooks import audit_documents
from fa1_r716_overlap import screen

def main(root):
    with tempfile.TemporaryDirectory(prefix='fa1_r716_toybooks_') as tmp:
        root_tmp=pathlib.Path(tmp);inputdir=root_tmp/'books';inputdir.mkdir()
        for i in range(120):
            (inputdir/f'toy_{i:04d}.txt').write_text((f'chapter {i} the fox and the king talked about rivers and bright stars {i} '
                   + 'the old house beside the river was quiet and green. ' *4)*2)
        dest=root_tmp/'prepared';prepare(root,inputdir,dest,limit=120)
        docs=audit_documents(root,dest)
        overlap=screen(root,dest)
        assert [docs[k]['books'] for k in ('train','dev','test')]==[108,6,6]
        assert overlap['test_overlap_screen_passed']
        out={'toy_fixture_only':True,'pass':True,'book_counts':{k:docs[k]['books'] for k in docs},
             'legacy_16gram_screen_passed':overlap['test_overlap_screen_passed']}
        print(json.dumps(out))
        return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=pathlib.Path,required=True)
    main(a.parse_args().root)