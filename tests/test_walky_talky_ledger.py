from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).parents[1]
    / "skills"
    / "transcribe-audio-to-vault"
    / "scripts"
    / "walky_talky_ledger.py"
)


def load_ledger_module():
    spec = importlib.util.spec_from_file_location("walky_talky_ledger", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WalkyTalkyLedgerTest(unittest.TestCase):
    def test_cleanup_audio_removes_only_derived_audio_inside_raw_run(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vault = tmp_path / "vault"
            run_dir = vault / "sources" / "walky-talky" / "raw" / "run-1"
            review_dir = run_dir / "review"
            review_dir.mkdir(parents=True)
            source = tmp_path / "Voice Memos" / "source.m4a"
            source.parent.mkdir()
            source.write_bytes(b"source")
            (review_dir / "clip.wav").write_bytes(b"derived audio")
            (run_dir / "transcript.txt").write_text("transcript")

            result = ledger.main(
                [
                    "cleanup-audio",
                    "--vault",
                    str(vault),
                    "--run-dir",
                    str(run_dir),
                    "--source",
                    str(source),
                    "--json",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(source.exists())
            self.assertFalse((review_dir / "clip.wav").exists())
            self.assertTrue((run_dir / "transcript.txt").exists())

    def test_cleanup_audio_rejects_run_directory_outside_vault_raw(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vault = tmp_path / "vault"
            run_dir = tmp_path / "outside"
            run_dir.mkdir()
            source = tmp_path / "source.m4a"
            source.write_bytes(b"source")
            (run_dir / "clip.wav").write_bytes(b"derived audio")

            args = type(
                "Args",
                (),
                {"vault": str(vault), "run_dir": str(run_dir), "source": str(source)},
            )()

            with self.assertRaisesRegex(ValueError, "must be inside"):
                ledger.cmd_cleanup_audio(args)
            self.assertTrue((run_dir / "clip.wav").exists())


if __name__ == "__main__":
    unittest.main()
