"""Interruption-safe DEV reports never mix split identities or rerun books."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from compare_r70_future_winner_guard import run


class FakeModel:
    class cfg:
        pair_successor_future=True


def fake_score(model,tokens,book_id,*,future_winner_guard=False,**kwargs):
    base={'ce_1':3.,'ce_2':4.-(.01 if future_winner_guard else 0),
          'ce_3':5.-(.01 if future_winner_guard else 0),
          'acc_1':.3,'acc_2':.2,'acc_3':.1,
          'top5_1':.6,'top5_2':.5,'top5_3':.4,
          'bits_per_raw_byte':4.}
    return {'exact':dict(base),'neutral':dict(base)}


class R70ResumptionTests(unittest.TestCase):
    def test_rerun_skips_completed_book_and_rejects_mismatched_split(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            data=root/'data'
            data.mkdir()
            np.arange(64,dtype='<u2').tofile(data/'valid.bin')
            digest=hashlib.sha256((data/'valid.bin').read_bytes()).hexdigest()
            (data/'meta.json').write_text(json.dumps({'split_unit':'whole_book',
                'unit':'UTF8_bytes_plus_eos','token_dtype':'<u2',
                'valid_sha256':digest,'valid_doc_lengths':[32,32]}))
            (data/'split_manifest.json').write_text(json.dumps({'books':[
                {'split':'dev','archive_name':'a'},
                {'split':'dev','archive_name':'b'}]}))
            checkpoint=root/'checkpoint.pt'
            checkpoint.write_bytes(b'verified-elsewhere-test-placeholder')
            report=root/'progress.json'
            calls=[]
            def interrupted(*a,**kw):
                calls.append(a[2])
                if a[2]=='b':
                    raise RuntimeError('interrupted after first book')
                return fake_score(*a,**kw)
            with patch('compare_r70_future_winner_guard.load_active_fa2',
                       return_value=(FakeModel(),None,{'step':4465})),patch(
                       'compare_r70_future_winner_guard.evaluate_causal_book',
                       side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run(checkpoint,data,books=2,positions=4,warmup=4,
                        output_path=report)
            first=json.loads(report.read_text())
            self.assertEqual(first['status'],'incomplete')
            self.assertEqual([x['book_id'] for x in first['per_book']],['a'])
            resumed=[]
            def recording(*a,**kw):
                resumed.append(a[2])
                return fake_score(*a,**kw)
            with patch('compare_r70_future_winner_guard.load_active_fa2',
                       return_value=(FakeModel(),None,{'step':4465})),patch(
                       'compare_r70_future_winner_guard.evaluate_causal_book',
                       side_effect=recording):
                completed=run(checkpoint,data,books=2,positions=4,warmup=4,
                              output_path=report)
            self.assertEqual(resumed,['b','b'])
            self.assertEqual(completed['dev_books'],2)
            self.assertEqual(len(json.loads(report.read_text())['per_book']),2)
            with patch('compare_r70_future_winner_guard.load_active_fa2',
                       return_value=(FakeModel(),None,{'step':4465})):
                with self.assertRaisesRegex(ValueError,'another run'):
                    run(checkpoint,data,books=2,positions=3,warmup=4,
                        output_path=report)


if __name__=='__main__':
    unittest.main()
