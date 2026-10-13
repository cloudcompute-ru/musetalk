import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'provision.sh'
PAYLOAD = b'fixture model bytes'


class DetectorDownloadTest(unittest.TestCase):
    def run_download(self, cached=None, download=PAYLOAD, curl_status=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'torch/hub/checkpoints/s3fd-619a316812.pth'
            target.parent.mkdir(parents=True)
            if cached is not None:
                target.write_bytes(cached)
            (root / 'response').write_bytes(download)
            binary = root / 'bin'
            binary.mkdir()
            curl = binary / 'curl'
            curl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
root = pathlib.Path(os.environ['TEST_ROOT'])
(root / 'called').touch()
assert '--max-time' in sys.argv and '--connect-timeout' in sys.argv
assert '--retry' in sys.argv
pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes((root / 'response').read_bytes())
sys.exit(int(os.environ['CURL_STATUS']))
''')
            curl.chmod(0o755)
            source = SCRIPT.read_text().split('download_s3fd() {', 1)[1].split('\n}\n', 1)[0]
            command = 'set -euo pipefail\nlog() { echo "$*"; }\ndownload_s3fd() {' + source + '\n}\ndownload_s3fd\n'
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                               TORCH_HOME=str(root / 'torch'), TEST_ROOT=str(root),
                               S3FD_SHA256=hashlib.sha256(PAYLOAD).hexdigest(), CURL_STATUS=str(curl_status))
            result = subprocess.run(['bash', '-c', command], env=environment, capture_output=True, text=True)
            return result, target.read_bytes() if target.exists() else None, (root / 'called').exists()

    def test_valid_download_is_promoted_to_the_model_cache(self):
        result, data, called = self.run_download()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(called)
        self.assertEqual(PAYLOAD, data)

    def test_valid_cached_model_skips_the_network(self):
        result, data, called = self.run_download(cached=PAYLOAD)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse(called)
        self.assertEqual(PAYLOAD, data)

    def test_corrupted_cached_model_is_replaced(self):
        result, data, called = self.run_download(cached=b'truncated')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(called)
        self.assertEqual(PAYLOAD, data)

    def test_invalid_download_never_becomes_a_cached_model(self):
        result, data, _ = self.run_download(download=b'truncated')
        self.assertNotEqual(0, result.returncode)
        self.assertIsNone(data)
        self.assertIn('checksum mismatch', result.stdout)

    def test_network_failure_never_becomes_a_cached_model(self):
        result, data, _ = self.run_download(curl_status=28)
        self.assertNotEqual(0, result.returncode)
        self.assertIsNone(data)


if __name__ == '__main__':
    unittest.main()
