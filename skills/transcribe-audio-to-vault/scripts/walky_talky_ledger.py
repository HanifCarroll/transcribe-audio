#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
DEFAULT_SCOPE = "walky-talky"
VOICE_MEMOS_ROOT = (
    "~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"
)
VOICE_MEMOS_DB = "CloudRecordings.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="walky_talky_ledger.py",
        description="Track imported walky-talky voice notes for transcribe-audio-to-vault.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser(
        "scan", help="Find new walky-talky audio since the latest ledger source time."
    )
    add_common_args(scan)
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=cmd_scan)

    append = subparsers.add_parser("append", help="Append a successful import record.")
    add_common_args(append)
    append.add_argument("--source", required=True)
    append.add_argument("--note", required=True)
    append.add_argument("--txt", required=True)
    append.add_argument("--json-artifact", required=True)
    append.add_argument("--run-dir", required=True)
    append.add_argument("--backend", required=True)
    append.add_argument("--model", required=True)
    append.add_argument("--imported-at")
    append.add_argument("--seeded", action="store_true")
    append.add_argument("--force", action="store_true")
    append.add_argument("--json", action="store_true")
    append.set_defaults(func=cmd_append)

    cleanup_audio = subparsers.add_parser(
        "cleanup-audio",
        help="Delete derived audio from one completed walky-talky raw run.",
    )
    add_common_args(cleanup_audio)
    cleanup_audio.add_argument("--run-dir", required=True)
    cleanup_audio.add_argument("--source", required=True)
    cleanup_audio.add_argument("--json", action="store_true")
    cleanup_audio.set_defaults(func=cmd_cleanup_audio)

    args = parser.parse_args(argv)
    try:
        payload = args.func(args)
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload, indent=2), file=sys.stdout)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "scan":
        for item in payload["new_files"]:
            print(item["source_path"])
    elif args.command == "append":
        print(payload["ledger_path"])
    else:
        print(payload["removed_count"])
    return 0 if payload.get("ok") else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vault",
        default=os.getcwd(),
        help="Vault root. Defaults to the current directory.",
    )
    parser.add_argument("--ledger")
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Audio root to scan. Can be repeated. Defaults to Voice Memos plus sources/walky-talky/inbox.",
    )


def cmd_scan(args: argparse.Namespace) -> dict[str, Any]:
    vault = expand_path(args.vault)
    ledger_path = resolve_ledger(vault, args.ledger)
    entries = read_ledger(ledger_path, scope=args.scope)
    imported_hashes = {entry.get("sha256") for entry in entries if entry.get("sha256")}
    latest_source_time = latest_imported_source_time(entries)
    roots = scan_roots(vault, args.roots)
    deleted_voice_memos, warnings = deleted_voice_memo_paths(roots, args.roots)
    skipped_deleted_count = 0
    candidates: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in audio_files(root):
            if is_excluded(path, vault):
                continue
            if path.resolve() in deleted_voice_memos:
                skipped_deleted_count += 1
                continue
            identity = source_identity(path)
            source_time = identity["source_unix_seconds"]
            if latest_source_time is not None and source_time <= latest_source_time:
                continue
            if identity["sha256"] in imported_hashes:
                continue
            candidates.append(identity)

    candidates.sort(key=lambda item: (item["source_unix_seconds"], item["source_path"]))
    return {
        "ok": True,
        "scope": args.scope,
        "vault": str(vault),
        "ledger_path": str(ledger_path),
        "latest_source_unix_seconds": latest_source_time,
        "roots": [str(root) for root in roots if root.exists()],
        "skipped_deleted_count": skipped_deleted_count,
        "new_count": len(candidates),
        "new_files": candidates,
        "warnings": warnings,
    }


def cmd_append(args: argparse.Namespace) -> dict[str, Any]:
    vault = expand_path(args.vault)
    ledger_path = resolve_ledger(vault, args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_ledger(ledger_path, scope=args.scope)
    identity = source_identity(expand_path(args.source))
    duplicate = next(
        (entry for entry in existing if entry.get("sha256") == identity["sha256"]), None
    )
    if duplicate and not args.force:
        return {
            "ok": True,
            "duplicate": True,
            "ledger_path": str(ledger_path),
            "record": duplicate,
        }

    imported_at = args.imported_at or now_local_iso()
    record = {
        "schema_version": 1,
        "scope": args.scope,
        "status": "imported",
        "seeded": bool(args.seeded),
        "imported_at": imported_at,
        **identity,
        "note_path": str(expand_path(args.note)),
        "txt_path": str(expand_path(args.txt)),
        "json_path": str(expand_path(args.json_artifact)),
        "run_dir": str(expand_path(args.run_dir)),
        "backend": args.backend,
        "model": args.model,
    }
    with ledger_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "ok": True,
        "duplicate": False,
        "ledger_path": str(ledger_path),
        "record": record,
    }


def cmd_cleanup_audio(args: argparse.Namespace) -> dict[str, Any]:
    vault = expand_path(args.vault)
    raw_root = (vault / "sources" / DEFAULT_SCOPE / "raw").resolve()
    run_dir = expand_path(args.run_dir)
    source = expand_path(args.source)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not is_relative_to(run_dir, raw_root):
        raise ValueError(f"run directory must be inside {raw_root}: {run_dir}")

    removed_count = 0
    removed_bytes = 0
    preserved_source = False
    for path in audio_files(run_dir):
        resolved = path.resolve()
        if resolved == source:
            preserved_source = True
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_count += 1

    return {
        "ok": True,
        "vault": str(vault),
        "run_dir": str(run_dir),
        "source": str(source),
        "preserved_source": preserved_source,
        "removed_count": removed_count,
        "removed_bytes": removed_bytes,
    }


def scan_roots(vault: pathlib.Path, roots: list[str] | None) -> list[pathlib.Path]:
    if roots:
        return [expand_path(root) for root in roots]
    return [
        expand_path(VOICE_MEMOS_ROOT),
        vault / "sources" / DEFAULT_SCOPE / "inbox",
    ]


def audio_files(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in AUDIO_EXTENSIONS else []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def is_excluded(path: pathlib.Path, vault: pathlib.Path) -> bool:
    excluded = [
        vault / "sources" / "feedback" / "casamo",
        vault / "sources" / DEFAULT_SCOPE / "raw",
    ]
    resolved = path.resolve()
    return any(
        is_relative_to(resolved, root.resolve()) for root in excluded if root.exists()
    )


def deleted_voice_memo_paths(
    roots: list[pathlib.Path],
    explicit_roots: list[str] | None,
) -> tuple[set[pathlib.Path], list[str]]:
    if explicit_roots:
        return set(), []

    voice_memos_root = expand_path(VOICE_MEMOS_ROOT)
    if not any(root.resolve() == voice_memos_root for root in roots):
        return set(), []

    db_path = voice_memos_root / VOICE_MEMOS_DB
    if not db_path.exists():
        return set(), [f"Voice Memos deletion metadata not found: {db_path}"]

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                """
                SELECT ZPATH
                FROM ZCLOUDRECORDING
                WHERE ZEVICTIONDATE IS NOT NULL
                  AND ZPATH IS NOT NULL
                """
            ).fetchall()
    except sqlite3.Error:
        return set(), [f"Could not read Voice Memos deletion metadata: {db_path}"]

    deleted_paths = {(voice_memos_root / row[0]).resolve() for row in rows if row[0]}
    return deleted_paths, []


def read_ledger(path: pathlib.Path, *, scope: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if entry.get("scope") == scope and entry.get("status") == "imported":
            entries.append(entry)
    return entries


def latest_imported_source_time(entries: list[dict[str, Any]]) -> float | None:
    times = [
        value
        for value in (entry.get("source_unix_seconds") for entry in entries)
        if isinstance(value, int | float)
    ]
    return max(times) if times else None


def source_identity(path: pathlib.Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"source does not exist: {path}")
    stat = path.stat()
    recorded_at, source_unix_seconds = recorded_time_from_name(path)
    if source_unix_seconds is None:
        source_unix_seconds = stat.st_mtime
    return {
        "source_path": str(path),
        "source_name": path.name,
        "source_size": stat.st_size,
        "source_mtime": dt.datetime.fromtimestamp(stat.st_mtime)
        .astimezone()
        .isoformat(),
        "source_unix_seconds": source_unix_seconds,
        "source_recorded_at": recorded_at,
        "sha256": sha256_file(path),
    }


def recorded_time_from_name(path: pathlib.Path) -> tuple[str | None, float | None]:
    stem = path.stem
    if len(stem) < 15:
        return None, None
    prefix = stem[:15]
    try:
        parsed = dt.datetime.strptime(prefix, "%Y%m%d %H%M%S")
    except ValueError:
        return None, None
    aware = parsed.astimezone()
    return aware.isoformat(), aware.timestamp()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ledger(vault: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return expand_path(explicit)
    return vault / ".transcribe-audio" / "imports.jsonl"


def expand_path(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def now_local_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
