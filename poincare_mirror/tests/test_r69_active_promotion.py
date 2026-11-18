"""R69 is the unified default with a fail-closed, hash-verified checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fa2_active import ACTIVE_MANIFEST, resolve_active_checkpoint


class R69ActivePromotionTests(unittest.TestCase):
    def test_source_manifest_contains_r69_override_and_caveats(self):
        info = json.loads(ACTIVE_MANIFEST.read_text())
        self.assertEqual((info['architecture'], info['active_round'], info['active_step']),
                         ('FA2_UNIFIED', 'R69', 4465))
        self.assertEqual(info['total_adam_states'], 198)
        self.assertEqual(info['inherited_adam_states'], 197)
        self.assertEqual(info['promotion'], 'explicit_user_override')
        self.assertFalse(info['strict_three_depth_gate_pass'])
        self.assertFalse(info['test_opened'])
        self.assertFalse(info['r69_dev_128_warmup_128_positions']['strict_DEV_gate_pass'])

    def test_r69_manifest_accepts_exact_hash_and_rejects_wrong_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'r69.pt'
            path.write_bytes(b'test-checkpoint-not-a-real-trained-weight')
            info = {'architecture': 'FA2_UNIFIED', 'active_round': 'R69',
                    'active_step': 4465, 'checkpoint_filename': path.name,
                    'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = Path(directory) / 'active.json'
            manifest.write_text(json.dumps(info))
            resolved, record = resolve_active_checkpoint(path, manifest)
            self.assertEqual((resolved, record['active_step']), (path, 4465))
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                resolve_active_checkpoint(path, manifest)

    def test_inconsistent_round_step_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'active.json'
            manifest.write_text(json.dumps({'architecture': 'FA2_UNIFIED',
                'active_round': 'R69', 'active_step': 4209,
                'checkpoint_filename': 'none.pt', 'checkpoint_sha256': '0' * 64}))
            with self.assertRaisesRegex(ValueError, 'lineage'):
                resolve_active_checkpoint(manifest_path=manifest)


if __name__ == '__main__':
    unittest.main()
