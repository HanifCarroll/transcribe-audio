from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = (
    Path(__file__).parents[1]
    / "transcribe_audio"
    / "walky_talky_ledger.py"
)


def load_ledger_module():
    spec = importlib.util.spec_from_file_location("walky_talky_ledger", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WalkyTalkyLedgerTest(unittest.TestCase):
    def test_scan_keeps_unimported_audio_older_than_the_latest_import(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            older = root / "20260101 120000.m4a"
            older.write_bytes(b"not imported")
            state = root / "imports.jsonl"
            state.write_text(json.dumps({"scope": "walky-talky", "status": "imported",
                "source_unix_seconds": 9999999999, "sha256": "different"}) + "\n")
            args = SimpleNamespace(vault=tmp, ledger=str(state), scope="walky-talky", roots=[tmp])
            self.assertEqual(ledger.cmd_scan(args)["new_count"], 1)
            with patch.object(ledger, "VOICE_MEMOS_ROOT", tmp):
                self.assertEqual(ledger.scan_roots(root, None), [root])

    def test_explicit_voice_memos_root_still_excludes_deleted_recordings(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with ledger.sqlite3.connect(root / "CloudRecordings.db") as db:
                db.execute("CREATE TABLE ZCLOUDRECORDING (ZPATH TEXT, ZEVICTIONDATE TEXT)")
                db.execute("INSERT INTO ZCLOUDRECORDING VALUES ('deleted.m4a', 'deleted')")
            with patch.object(ledger, "VOICE_MEMOS_ROOT", tmp):
                deleted, warnings = ledger.deleted_voice_memo_paths([root], [tmp])
            self.assertEqual(deleted, {root / "deleted.m4a"})
            self.assertEqual(warnings, [])

    def test_append_rejects_missing_artifacts_without_marking_the_source_imported(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(vault=tmp, ledger=None, scope="walky-talky",
                note=str(Path(tmp) / "missing.md"), txt="missing.txt", json_artifact="missing.json")
            with self.assertRaises(FileNotFoundError):
                ledger.cmd_append(args)
            self.assertFalse((Path(tmp) / ".transcribe-audio").exists())

    def test_completed_import_is_recorded_once(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source.m4a", "note.md", "transcript.txt", "transcript.json"):
                (root / name).write_text("fixture")
            args = SimpleNamespace(vault=tmp, ledger=None, scope="walky-talky",
                source=str(root / "source.m4a"), note=str(root / "note.md"),
                txt=str(root / "transcript.txt"), json_artifact=str(root / "transcript.json"),
                run_dir=tmp, backend="spokenly", model="apple-speech-analyzer",
                imported_at=None, seeded=False, force=False)
            self.assertFalse(ledger.cmd_append(args)["duplicate"])
            self.assertTrue(ledger.cmd_append(args)["duplicate"])
            self.assertEqual(len(ledger.read_ledger(root / ".transcribe-audio/imports.jsonl", scope=args.scope)), 1)

    def test_cleanup_requires_a_real_source_and_one_run_directory(self) -> None:
        ledger = load_ledger_module()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "sources/walky-talky/raw"
            run = raw / "one-run"
            run.mkdir(parents=True)
            clip = run / "clip.wav"
            clip.write_bytes(b"audio")
            args = SimpleNamespace(vault=tmp, run_dir=str(run), source=str(Path(tmp) / "missing.m4a"))
            with self.assertRaises(FileNotFoundError):
                ledger.cmd_cleanup_audio(args)
            args.run_dir = str(raw)
            with self.assertRaises(ValueError):
                ledger.cmd_cleanup_audio(args)
            self.assertTrue(clip.exists())

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
