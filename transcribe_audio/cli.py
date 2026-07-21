from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import pathlib
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any


DEFAULT_MODEL_DIR = "~/.local/share/transcribe-audio/models"
SUPERWHISPER_SMALL = "~/Library/Application Support/superwhisper/ggml-small.bin"
DEFAULT_SPOKENLY_APP = "/Applications/Spokenly.app"
DEFAULT_SPOKENLY_MCP_URL = "http://127.0.0.1:51089"
DEFAULT_SPOKENLY_TDT_MODEL_DIR = (
    "~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3"
)
SPOKENLY_TDT_MODEL_ID = "parakeetTDT06"
DEFAULT_MLX_PYTHON = "3.11"
SUPPORTED_FORMATS = {"txt", "json", "srt", "vtt"}
MLX_SUPPORTED_FORMATS = {"txt", "json"}
SPOKENLY_SUPPORTED_FORMATS = {"txt", "json"}
SUPPORTED_AUDIO_EXTENSIONS = {"aac", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"}

SPOKENLY_MODELS = OrderedDict(
    [
        (
            "parakeet-tdt-0.6b-v3",
            {
                "id": SPOKENLY_TDT_MODEL_ID,
                "note": "Fast English file transcription through Spokenly's local MCP bridge.",
            },
        )
    ]
)

MLX_MODELS = OrderedDict(
    [
        (
            "large-v3",
            {
                "repo": "mlx-community/whisper-large-v3-mlx",
                "note": "Default backend/model for high-quality note transcripts on Apple Silicon.",
            },
        ),
        (
            "large-v3-turbo",
            {
                "repo": "mlx-community/whisper-large-v3-turbo",
                "note": "Faster MLX model with some quality tradeoff.",
            },
        ),
    ]
)

WHISPER_CPP_MODELS = OrderedDict(
    [
        (
            "large-v3",
            {
                "file": "ggml-large-v3.bin",
                "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
                "note": "Best whisper.cpp quality; slower than turbo.",
            },
        ),
        (
            "large-v3-turbo",
            {
                "file": "ggml-large-v3-turbo.bin",
                "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
                "note": "Fast whisper.cpp fallback for comparisons.",
            },
        ),
        (
            "medium",
            {
                "file": "ggml-medium.bin",
                "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
                "note": "Good fallback if large models are unavailable.",
            },
        ),
        (
            "small",
            {
                "file": "ggml-small.bin",
                "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
                "note": "Fast fallback; lower accuracy on names and technical terms.",
            },
        ),
    ]
)

MLX_DOWNLOAD_SCRIPT = """
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
force = sys.argv[2] == "1"
print(snapshot_download(repo_id=repo_id, force_download=force))
"""

MLX_TRANSCRIBE_SCRIPT = """
import json
import pathlib
import sys

import mlx_whisper

audio_path = sys.argv[1]
output_base = sys.argv[2]
model = sys.argv[3]
language = sys.argv[4]
formats = [part for part in sys.argv[5].split(",") if part]
prompt = sys.argv[6] or None
verbose = sys.argv[7] == "1"

result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo=model,
    verbose=verbose,
    temperature=0.0,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=False,
    initial_prompt=prompt,
    word_timestamps=True,
    hallucination_silence_threshold=2.0,
    language=None if language == "auto" else language,
)

base = pathlib.Path(output_base)
if "txt" in formats:
    pathlib.Path(str(base) + ".txt").write_text(result.get("text", "").strip() + "\\n", encoding="utf-8")
if "json" in formats:
    pathlib.Path(str(base) + ".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\\n",
        encoding="utf-8",
    )
"""


class ToolError(Exception):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        command: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.command = command
        self.exit_code = exit_code


@dataclasses.dataclass
class EnvPaths:
    model_dir: pathlib.Path
    superwhisper_small: pathlib.Path
    spokenly_app: pathlib.Path
    spokenly_tdt_model_dir: pathlib.Path
    spokenly_mcp_url: str
    ffmpeg: pathlib.Path | None
    ffprobe: pathlib.Path | None
    whisper_cli: pathlib.Path | None
    uv: pathlib.Path | None
    mlx_whisper_python: pathlib.Path | None
    curl: pathlib.Path | None


@dataclasses.dataclass
class SelectedModel:
    name: str
    locator: str
    path: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class BatchTask:
    index: int
    input_path: pathlib.Path
    output_name: str
    args: argparse.Namespace


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    json_on_error = "--json" in argv
    try:
        args = build_parser().parse_args(normalize_argv(argv))
        args.func(args)
        return 0
    except ToolError as err:
        if json_on_error:
            print_json(error_payload(err))
        else:
            print(f"transcribe-audio: {err.message}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-audio",
        description="Transcribe local audio with MLX Whisper, Spokenly, or whisper.cpp.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check tools and installed models.")
    add_backend_arg(doctor)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    models = subparsers.add_parser("models", help="List known and installed models.")
    models.add_argument(
        "--backend",
        choices=backend_choices(),
        help="Limit results to one backend.",
    )
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    download = subparsers.add_parser("download-model", help="Download a known model.")
    download.add_argument("name", choices=list(MLX_MODELS) + ["medium", "small"])
    add_backend_arg(download)
    download.add_argument("--force", action="store_true")
    download.add_argument("--quiet", action="store_true")
    download.add_argument("--json", action="store_true")
    download.set_defaults(func=cmd_download_model)

    preprocess = subparsers.add_parser(
        "preprocess", help="Remove long silence before transcription."
    )
    preprocess.add_argument("input", type=pathlib.Path)
    preprocess.add_argument("--output", type=pathlib.Path)
    preprocess.add_argument("--silence-threshold", default="-35dB")
    preprocess.add_argument("--silence-duration", type=float, default=2.0)
    preprocess.add_argument("--keep-silence", type=float, default=0.4)
    preprocess.add_argument("--json", action="store_true")
    preprocess.set_defaults(func=cmd_preprocess)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file.")
    transcribe.add_argument("input", type=pathlib.Path)
    add_transcribe_args(transcribe)
    transcribe.set_defaults(func=cmd_transcribe)

    batch = subparsers.add_parser("batch", help="Transcribe multiple audio files.")
    batch.add_argument("inputs", nargs="+", type=pathlib.Path)
    add_transcribe_args(batch, batch_mode=True)
    batch.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help="Number of raw transcription jobs to run concurrently. Defaults to 1.",
    )
    batch.add_argument("--fail-fast", action="store_true")
    batch.add_argument("--markdown", dest="markdown", action="store_true", default=True)
    batch.add_argument("--no-markdown", dest="markdown", action="store_false")
    batch.set_defaults(func=cmd_batch)

    discover = subparsers.add_parser("discover", help="Find likely audio files.")
    discover.add_argument("root", nargs="?", type=pathlib.Path)
    discover.add_argument("--since")
    discover.add_argument("--limit", type=int)
    discover.add_argument(
        "--recursive", dest="recursive", action="store_true", default=True
    )
    discover.add_argument("--no-recursive", dest="recursive", action="store_false")
    discover.add_argument("--json", action="store_true")
    discover.set_defaults(func=cmd_discover)

    quality = subparsers.add_parser(
        "quality", help="Check transcripts for common hallucination warnings."
    )
    quality.add_argument("input", type=pathlib.Path)
    quality.add_argument("--json", action="store_true")
    quality.set_defaults(func=cmd_quality)

    review = subparsers.add_parser(
        "review",
        help="Find timestamped transcript review windows and optionally extract clips.",
    )
    review.add_argument("transcript_json", type=pathlib.Path)
    review.add_argument("--audio", type=pathlib.Path)
    review.add_argument(
        "--phrase",
        dest="phrases",
        action="append",
        default=[],
        help="Phrase to locate in the transcript; can be repeated.",
    )
    review.add_argument("--min-word-probability", type=float, default=0.35)
    review.add_argument("--context", type=float, default=2.0)
    review.add_argument("--merge-gap", type=float, default=1.0)
    review.add_argument("--max-windows", type=int, default=12)
    review.add_argument("--output-dir", type=pathlib.Path)
    review.add_argument("--extract-clips", action="store_true")
    review.add_argument("--json", action="store_true")
    review.set_defaults(func=cmd_review)

    combine = subparsers.add_parser(
        "combine", help="Combine transcript text into Markdown."
    )
    combine.add_argument("inputs", nargs="+", type=pathlib.Path)
    combine.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("transcripts.md")
    )
    combine.add_argument("--title", default="Transcripts")
    combine.add_argument("--json", action="store_true")
    combine.set_defaults(func=cmd_combine)

    note = subparsers.add_parser(
        "note", help="Write an Obsidian-friendly transcript note."
    )
    note.add_argument("transcript", type=pathlib.Path)
    note.add_argument("--output", type=pathlib.Path, required=True)
    note.add_argument("--title", required=True)
    note.add_argument("--date")
    note.add_argument("--source")
    note.add_argument("--backend")
    note.add_argument("--model")
    note.add_argument("--raw-json")
    note.add_argument("--preprocessing-note")
    note.add_argument("--json", action="store_true")
    note.set_defaults(func=cmd_note)

    return parser


def backend_choices(*, include_spokenly: bool = True) -> list[str]:
    choices = ["mlx", "mlx-whisper", "whisper-cpp", "whisper.cpp", "cpp"]
    if include_spokenly:
        choices.extend(["spokenly", "parakeet", "parakeet-tdt"])
    return choices


def add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="mlx",
        choices=backend_choices(),
        help="Transcription backend. Defaults to MLX; use spokenly for Parakeet TDT.",
    )


def add_transcribe_args(
    parser: argparse.ArgumentParser, *, batch_mode: bool = False
) -> None:
    add_backend_arg(parser)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--formats", default="txt,json")
    parser.add_argument("--output-dir", type=pathlib.Path)
    if not batch_mode:
        parser.add_argument("--output-name")
        parser.add_argument("--markdown", action="store_true")
        parser.add_argument("--print", dest="print_text", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")


def normalize_argv(argv: list[str]) -> list[str]:
    commands = {
        "doctor",
        "models",
        "download-model",
        "preprocess",
        "transcribe",
        "batch",
        "discover",
        "quality",
        "review",
        "combine",
        "note",
    }
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        return ["transcribe", *argv]
    return argv


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer: {value}"
        ) from err
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def cmd_doctor(args: argparse.Namespace) -> None:
    env = env_paths()
    backend = normalize_backend(args.backend)
    spokenly_bridge_available, spokenly_bridge_error = spokenly_bridge_status(env)
    spokenly_selected_model = spokenly_selected_file_model_id()
    payload = {
        "ok": backend_ready(
            env,
            backend,
            spokenly_bridge_available=spokenly_bridge_available,
            spokenly_selected_model=spokenly_selected_model,
        ),
        "backend": backend,
        "tools": {
            "ffmpeg": str(env.ffmpeg) if env.ffmpeg else None,
            "ffprobe": str(env.ffprobe) if env.ffprobe else None,
            "whisper_cli": str(env.whisper_cli) if env.whisper_cli else None,
            "uv": str(env.uv) if env.uv else None,
            "mlx_whisper_python": str(env.mlx_whisper_python)
            if env.mlx_whisper_python
            else None,
            "curl": str(env.curl) if env.curl else None,
            "spokenly_bridge": spokenly_bridge_available,
            "spokenly_bridge_error": spokenly_bridge_error,
            "spokenly_file_model": spokenly_selected_model,
        },
        "paths": {
            "model_dir": str(env.model_dir),
            "superwhisper_small": str(env.superwhisper_small),
            "spokenly_app": str(env.spokenly_app),
            "spokenly_tdt_model_dir": str(env.spokenly_tdt_model_dir),
            "spokenly_mcp_url": env.spokenly_mcp_url,
        },
        "models": installed_models(env),
    }
    if args.json:
        print_json(payload)
        return
    print(f"ffmpeg: {payload['tools']['ffmpeg'] or 'missing'}")
    print(f"uv: {payload['tools']['uv'] or 'missing'}")
    print(f"whisper-cli: {payload['tools']['whisper_cli'] or 'missing'}")
    print(f"spokenly bridge: {'available' if spokenly_bridge_available else 'missing'}")
    print(f"spokenly file model: {spokenly_selected_model or 'unknown'}")
    for row in payload["models"]:
        status = "installed" if row["installed"] else "missing"
        print(f"{row['backend']} / {row['name']}: {status}")


def cmd_models(args: argparse.Namespace) -> None:
    backend = normalize_backend(args.backend) if args.backend else None
    rows = installed_models(env_paths())
    if backend:
        rows = [row for row in rows if row["backend"] == backend]
    recommended = (
        f"spokenly:{SPOKENLY_TDT_MODEL_ID}"
        if backend == "spokenly"
        else "mlx-whisper:large-v3"
    )
    payload = {
        "ok": True,
        "recommended": recommended,
        "models": rows,
    }
    if args.json:
        print_json(payload)
        return
    print(f"Recommended: {recommended}")
    for row in payload["models"]:
        status = "installed" if row["installed"] else "missing"
        print(f"- {row['backend']} / {row['name']} ({status}): {row['note']}")
        for path in row["paths"]:
            print(f"  {path}")


def cmd_download_model(args: argparse.Namespace) -> None:
    backend = normalize_backend(args.backend)
    env = env_paths()
    if backend == "spokenly":
        raise ToolError(
            "download",
            "Install NVIDIA Parakeet TDT 0.6B V3 from Spokenly's model manager; "
            "transcribe-audio does not download Spokenly models.",
        )
    if backend == "mlx-whisper":
        payload = download_mlx_model(args.name, env, force=args.force, quiet=args.quiet)
    else:
        payload = download_whisper_cpp_model(
            args.name, env, force=args.force, quiet=args.quiet
        )
    if args.json:
        print_json(payload)
    else:
        action = "Installed" if payload["downloaded"] else "Already installed"
        print(f"{action} {payload['model']}: {payload['path']}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    payload = preprocess_audio(
        args.input,
        output=args.output,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        keep_silence=args.keep_silence,
    )
    if args.json:
        print_json(payload)
        return
    print(f"Wrote {payload['output']}")
    if payload["removed_seconds"] is not None:
        print(f"Removed about {payload['removed_seconds']:.1f}s of silence")


def cmd_transcribe(args: argparse.Namespace) -> None:
    payload = transcribe_one(args, args.input)
    if args.json:
        print_json(payload)
    else:
        print(f"Backend: {payload['backend']}")
        print(f"Model: {payload['model']['name']}")
        for fmt, path in payload["outputs"].items():
            print(f"{fmt}: {path}")
        print(f"Done in {payload['duration_seconds']}s")
    if getattr(args, "print_text", False) and "txt" in payload["outputs"]:
        print()
        print(pathlib.Path(payload["outputs"]["txt"]).read_text(encoding="utf-8"))


def cmd_batch(args: argparse.Namespace) -> None:
    run_dir = (
        expand_path(args.output_dir)
        if args.output_dir
        else pathlib.Path.cwd() / "transcripts" / f"run-{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    tasks = build_batch_tasks(args, run_dir, formats)
    jobs = min(args.jobs, len(tasks))
    if normalize_backend(args.backend) == "spokenly" and jobs > 1:
        raise ToolError("cli", "Spokenly file transcription requires --jobs 1")
    if jobs > 1 and args.fail_fast:
        raise ToolError("cli", "--fail-fast requires --jobs 1")
    items = run_batch_tasks(tasks, jobs=jobs, fail_fast=args.fail_fast)
    successful_txt = [
        (input_label(pathlib.Path(item["input"])), pathlib.Path(item["outputs"]["txt"]))
        for item in items
        if item["ok"] and "txt" in item["outputs"]
    ]
    markdown = None
    if args.markdown and successful_txt:
        markdown_path = run_dir / "transcripts.md"
        write_combined_markdown(successful_txt, markdown_path, "Transcripts")
        markdown = str(markdown_path)
    success_count = sum(1 for item in items if item["ok"])
    payload = {
        "ok": success_count == len(items),
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / "run.json"),
        "markdown": markdown,
        "jobs": jobs,
        "input_count": len(items),
        "success_count": success_count,
        "failure_count": len(items) - success_count,
        "items": items,
    }
    pathlib.Path(payload["manifest_path"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print_json(payload)
    else:
        print(
            f"Transcribed {success_count}/{len(items)} files. Manifest: {payload['manifest_path']}"
        )


def build_batch_tasks(
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    formats: list[str],
) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    reserved_output_names: set[str] = set()
    for index, input_path in enumerate(args.inputs, start=1):
        output_name = unique_output_name(
            input_path, index, run_dir, reserved_output_names
        )
        reserved_output_names.add(output_name)
        child_args = argparse.Namespace(**vars(args))
        child_args.input = input_path
        child_args.output_dir = run_dir
        child_args.output_name = output_name
        child_args.markdown = False
        child_args.print_text = False
        child_args.formats = ",".join(formats)
        tasks.append(
            BatchTask(
                index=index,
                input_path=input_path,
                output_name=output_name,
                args=child_args,
            )
        )
    return tasks


def run_batch_tasks(
    tasks: list[BatchTask],
    *,
    jobs: int,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    if jobs <= 1:
        items = []
        for task in tasks:
            item = run_batch_task(task)
            items.append(item)
            if fail_fast and not item["ok"]:
                break
        return items

    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_batch_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            results[task.index] = future.result()
    return [results[task.index] for task in tasks]


def run_batch_task(task: BatchTask) -> dict[str, Any]:
    try:
        payload = transcribe_one(task.args, task.input_path)
    except ToolError as err:
        return {
            "ok": False,
            "input": str(expand_path(task.input_path)),
            "output_name": task.output_name,
            "error": error_payload(err),
        }
    return {
        "ok": True,
        "input": payload["input"],
        "output_name": task.output_name,
        "outputs": payload["outputs"],
        "duration_seconds": payload["duration_seconds"],
    }


def cmd_discover(args: argparse.Namespace) -> None:
    root = (
        expand_path(args.root)
        if args.root
        else expand_path(pathlib.Path("~/Downloads"))
    )
    if not root.exists():
        raise ToolError("discover", f"discovery root does not exist: {root}")
    min_mtime = time.time() - parse_age(args.since) if args.since else None
    files = discover_audio(root, recursive=args.recursive, min_mtime=min_mtime)
    files.sort(key=lambda row: row["modified_unix_seconds"] or 0, reverse=True)
    if args.limit is not None:
        files = files[: args.limit]
    payload = {"ok": True, "root": str(root), "count": len(files), "files": files}
    if args.json:
        print_json(payload)
    else:
        for row in files:
            print(row["path"])


def cmd_quality(args: argparse.Namespace) -> None:
    path = expand_path(args.input)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ToolError("quality", f"failed to read transcript {path}: {err}") from err
    warnings = transcript_quality_warnings(text)
    payload = {
        "ok": not warnings,
        "input": str(path),
        "text_chars": len(text),
        "warnings": warnings,
    }
    if args.json:
        print_json(payload)
    elif warnings:
        print(f"Transcript has {len(warnings)} warning(s).")
        for warning in warnings:
            print(f"- {warning['code']}: {warning['message']}")
    else:
        print("No transcript quality warnings.")


def cmd_review(args: argparse.Namespace) -> None:
    transcript_json = expand_path(args.transcript_json)
    data = read_json_file(transcript_json, stage="review")
    windows = build_review_windows(
        data,
        phrases=args.phrases,
        min_word_probability=args.min_word_probability,
        context_seconds=args.context,
        merge_gap_seconds=args.merge_gap,
        max_windows=args.max_windows,
    )
    audio = expand_path(args.audio) if args.audio else None
    output_dir = (
        expand_path(args.output_dir)
        if args.output_dir
        else transcript_json.with_name(f"{transcript_json.stem}-review")
    )
    if args.extract_clips:
        if audio is None:
            raise ToolError("review", "--extract-clips requires --audio")
        if not audio.exists():
            raise ToolError("review", f"audio file does not exist: {audio}")
        env = env_paths()
        if env.ffmpeg is None:
            raise ToolError("doctor", "ffmpeg is required to extract review clips")
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, window in enumerate(windows, start=1):
            label = review_window_label(index, window)
            clip = output_dir / f"{label}.wav"
            extract_audio_clip(env, audio, clip, window["start"], window["end"])
            window["clip"] = str(clip)
            window["transcribe_commands"] = review_transcribe_commands(
                clip, output_dir, label
            )
    elif args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "ok": True,
        "transcript_json": str(transcript_json),
        "audio": str(audio) if audio else None,
        "output_dir": str(output_dir)
        if args.extract_clips or args.output_dir
        else None,
        "window_count": len(windows),
        "windows": windows,
    }
    if args.json:
        print_json(payload)
        return
    if not windows:
        print("No review windows found.")
        return
    for window in windows:
        print(f"{window['id']}: {window['start']:.2f}-{window['end']:.2f}s")
        print(f"  reasons: {', '.join(reason['code'] for reason in window['reasons'])}")
        print(f"  text: {window['text']}")
        if "clip" in window:
            print(f"  clip: {window['clip']}")


def cmd_combine(args: argparse.Namespace) -> None:
    inputs = collect_transcript_inputs(args.inputs)
    output = expand_path(args.output)
    write_combined_markdown(inputs, output, args.title)
    payload = {
        "ok": True,
        "output": str(output),
        "input_count": len(inputs),
        "inputs": [str(path) for _, path in inputs],
    }
    if args.json:
        print_json(payload)
    else:
        print(f"Wrote {output}")


def cmd_note(args: argparse.Namespace) -> None:
    transcript = expand_path(args.transcript)
    try:
        transcript_text = transcript.read_text(encoding="utf-8")
    except OSError as err:
        raise ToolError(
            "note", f"failed to read transcript {transcript}: {err}"
        ) from err
    output = expand_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    date = args.date or dt.date.today().isoformat()
    markdown = [
        "---",
        f"date: {yaml_scalar(date)}",
        "type: transcript",
        f"source_transcript: {yaml_scalar(str(transcript))}",
    ]
    if args.source:
        markdown.append(f"source_audio: {yaml_scalar(args.source)}")
    if args.backend:
        markdown.append(f"backend: {yaml_scalar(args.backend)}")
    if args.model:
        markdown.append(f"model: {yaml_scalar(args.model)}")
    if args.raw_json:
        markdown.append(f"raw_json: {yaml_scalar(args.raw_json)}")
    if args.preprocessing_note:
        markdown.append(f"preprocessing: {yaml_scalar(args.preprocessing_note)}")
    markdown.extend(["---", "", f"# {args.title.strip()}", ""])
    if args.preprocessing_note:
        markdown.extend([f"_Preprocessing: {args.preprocessing_note.strip()}_", ""])
    markdown.append(transcript_text.strip())
    markdown.append("")
    output.write_text("\n".join(markdown), encoding="utf-8")
    payload = {"ok": True, "output": str(output), "transcript": str(transcript)}
    if args.json:
        print_json(payload)
    else:
        print(f"Wrote {output}")


def transcribe_one(
    args: argparse.Namespace, input_path: pathlib.Path
) -> dict[str, Any]:
    env = env_paths()
    backend = normalize_backend(args.backend)
    ensure_tools(env, backend)
    input_path = expand_path(input_path)
    if not input_path.exists():
        raise ToolError("input", f"input file does not exist: {input_path}")
    input_path = input_path.resolve()
    formats = parse_formats(args.formats)
    validate_formats(formats, backend)
    model = choose_model(args.model, backend, env)
    prompt = resolve_prompt(args.prompt, args.prompt_file)
    if backend == "spokenly":
        validate_spokenly_options(args.language, prompt)
    output_dir = (
        expand_path(args.output_dir)
        if args.output_dir
        else input_path.parent / "transcripts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = getattr(args, "output_name", None) or slugify(input_path.name)
    output_base = output_dir / output_name
    start = time.time()

    with tempfile.TemporaryDirectory(prefix="transcribe-audio-") as tmp:
        tmp_wav = pathlib.Path(tmp) / f"{output_name}.wav"
        convert_audio(env, input_path, tmp_wav)
        converted_wav = None
        if args.keep_wav:
            kept = output_path_for_format(output_base, "wav")
            shutil.copyfile(tmp_wav, kept)
            converted_wav = str(kept)
        if backend == "mlx-whisper":
            run_mlx_whisper(
                env,
                tmp_wav,
                output_base,
                model.locator,
                args.language,
                formats,
                prompt,
                args.verbose,
            )
        elif backend == "whisper-cpp":
            run_whisper_cpp(
                env,
                tmp_wav,
                output_base,
                model,
                args.language,
                formats,
                prompt,
                args.no_gpu,
                args.verbose,
            )
        else:
            run_spokenly(
                env,
                tmp_wav,
                output_base,
                model,
                formats,
            )

    outputs = wait_for_outputs(output_base, formats)
    markdown = None
    if getattr(args, "markdown", False):
        if "txt" not in outputs:
            raise ToolError("outputs", "--markdown requires txt output format")
        markdown_path = output_path_for_format(output_base, "md")
        write_combined_markdown(
            [(input_label(input_path), pathlib.Path(outputs["txt"]))],
            markdown_path,
            "Transcript",
        )
        markdown = str(markdown_path)
    return {
        "ok": True,
        "input": str(input_path),
        "backend": backend,
        "model": {"name": model.name, "path": model.locator},
        "language": args.language,
        "formats": formats,
        "outputs": outputs,
        "duration_seconds": round(time.time() - start, 3),
        "converted_wav": converted_wav,
        "markdown": markdown,
    }


def download_mlx_model(
    name: str, env: EnvPaths, *, force: bool, quiet: bool
) -> dict[str, Any]:
    meta = MLX_MODELS.get(name)
    if meta is None:
        raise ToolError("download", f"unknown MLX model: {name}")
    cache_path = mlx_model_cache_path(meta["repo"])
    if cache_path.exists() and not force:
        return {
            "ok": True,
            "downloaded": False,
            "backend": "mlx-whisper",
            "model": name,
            "path": str(cache_path),
        }
    cmd = mlx_python_command(env)
    cmd.extend(["-c", MLX_DOWNLOAD_SCRIPT, meta["repo"], "1" if force else "0"])
    completed = run(cmd, "download", quiet=quiet, capture=True)
    path = (
        completed.stdout.strip().splitlines()[-1]
        if completed.stdout.strip()
        else str(cache_path)
    )
    return {
        "ok": True,
        "downloaded": True,
        "backend": "mlx-whisper",
        "model": name,
        "path": path,
    }


def download_whisper_cpp_model(
    name: str, env: EnvPaths, *, force: bool, quiet: bool
) -> dict[str, Any]:
    meta = WHISPER_CPP_MODELS.get(name)
    if meta is None:
        raise ToolError("download", f"unknown whisper.cpp model: {name}")
    if env.curl is None:
        raise ToolError("download", "curl is required to download whisper.cpp models")
    env.model_dir.mkdir(parents=True, exist_ok=True)
    target = env.model_dir / meta["file"]
    part = pathlib.Path(f"{target}.part")
    if target.exists() and not force:
        return {
            "ok": True,
            "downloaded": False,
            "backend": "whisper-cpp",
            "model": name,
            "path": str(target),
        }
    cmd = [str(env.curl)]
    if quiet:
        cmd.extend(["--silent", "--show-error"])
    cmd.extend(
        [
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(part),
            meta["url"],
        ]
    )
    run(cmd, "download")
    part.replace(target)
    return {
        "ok": True,
        "downloaded": True,
        "backend": "whisper-cpp",
        "model": name,
        "path": str(target),
    }


def preprocess_audio(
    input_path: pathlib.Path,
    *,
    output: pathlib.Path | None,
    silence_threshold: str,
    silence_duration: float,
    keep_silence: float,
) -> dict[str, Any]:
    env = env_paths()
    if env.ffmpeg is None:
        raise ToolError("doctor", "ffmpeg is required for preprocessing")
    input_path = expand_path(input_path)
    if not input_path.exists():
        raise ToolError("input", f"input file does not exist: {input_path}")
    input_path = input_path.resolve()
    output_path = (
        expand_path(output)
        if output
        else input_path.with_name(f"{input_label(input_path)}-desilenced.flac")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = probe_duration_seconds(env, input_path)
    audio_filter = (
        "silenceremove="
        f"start_periods=1:start_duration=0:start_threshold={silence_threshold}:start_silence=0:"
        f"stop_periods=-1:stop_duration={silence_duration}:stop_threshold={silence_threshold}:"
        f"stop_silence={keep_silence}:detection=rms:window=0.05"
    )
    run(
        [
            str(env.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-af",
            audio_filter,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "flac",
            str(output_path),
        ],
        "preprocess",
    )
    after = probe_duration_seconds(env, output_path)
    removed = (
        round(max(before - after, 0.0), 1)
        if before is not None and after is not None
        else None
    )
    return {
        "ok": True,
        "input": str(input_path),
        "output": str(output_path),
        "silence_threshold": silence_threshold,
        "silence_duration_seconds": silence_duration,
        "keep_silence_seconds": keep_silence,
        "input_duration_seconds": before,
        "output_duration_seconds": after,
        "removed_seconds": removed,
    }


def run_mlx_whisper(
    env: EnvPaths,
    input_wav: pathlib.Path,
    output_base: pathlib.Path,
    model: str,
    language: str,
    formats: list[str],
    prompt: str | None,
    verbose: bool,
) -> None:
    cmd = mlx_python_command(env)
    cmd.extend(
        [
            "-c",
            MLX_TRANSCRIBE_SCRIPT,
            str(input_wav),
            str(output_base),
            model,
            language,
            ",".join(formats),
            prompt or "",
            "1" if verbose else "0",
        ]
    )
    run(cmd, "mlx-whisper", quiet=not verbose)


def run_whisper_cpp(
    env: EnvPaths,
    input_wav: pathlib.Path,
    output_base: pathlib.Path,
    model: SelectedModel,
    language: str,
    formats: list[str],
    prompt: str | None,
    no_gpu: bool,
    verbose: bool,
) -> None:
    if env.whisper_cli is None:
        raise ToolError("doctor", "whisper-cli is required for whisper.cpp")
    if model.path is None:
        raise ToolError("model", "whisper.cpp backend requires a local ggml model path")
    cmd = [
        str(env.whisper_cli),
        "-m",
        str(model.path),
        "-f",
        str(input_wav),
        "-l",
        language,
        "-of",
        str(output_base),
        "--no-prints",
    ]
    if prompt:
        cmd.extend(["--prompt", prompt])
    if no_gpu:
        cmd.append("--no-gpu")
    if "txt" in formats:
        cmd.append("-otxt")
    if "json" in formats:
        cmd.append("-oj")
    if "srt" in formats:
        cmd.append("-osrt")
    if "vtt" in formats:
        cmd.append("-ovtt")
    run(cmd, "whisper", quiet=not verbose)


def run_spokenly(
    env: EnvPaths,
    input_wav: pathlib.Path,
    output_base: pathlib.Path,
    model: SelectedModel,
    formats: list[str],
) -> None:
    response = spokenly_rpc(
        env.spokenly_mcp_url,
        "tools/call",
        {
            "name": "transcribe_file",
            "arguments": {
                "file_path": str(input_wav.resolve()),
                "format": "json",
            },
        },
        timeout=600.0,
    )
    transcript = parse_spokenly_transcription(response)
    reported_model = transcript.get("modelId")
    if reported_model != model.locator:
        raise ToolError(
            "model",
            "Spokenly returned model "
            f"{reported_model or 'unknown'}; select NVIDIA Parakeet TDT 0.6B V3 "
            f"({model.locator}) for file transcription and retry.",
        )
    text = spokenly_transcript_text(transcript)
    if "txt" in formats:
        output_path_for_format(output_base, "txt").write_text(
            text + "\n",
            encoding="utf-8",
        )
    if "json" in formats:
        output_path_for_format(output_base, "json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def spokenly_rpc(
    url: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    request_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise ToolError(
            "spokenly",
            "Spokenly's local MCP bridge is unavailable. Open Spokenly, enable its local "
            f"MCP bridge, and retry ({url}): {err}",
        ) from err
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as err:
        raise ToolError("spokenly", f"Spokenly returned invalid JSON: {err}") from err
    if not isinstance(payload, dict):
        raise ToolError("spokenly", "Spokenly returned a non-object JSON-RPC response")
    if payload.get("error"):
        raise ToolError("spokenly", f"Spokenly MCP error: {payload['error']}")
    return payload


def parse_spokenly_transcription(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, dict):
        raise ToolError("spokenly", "Spokenly response is missing a result object")
    if result.get("isError"):
        raise ToolError("spokenly", "Spokenly reported a transcription error")
    content = result.get("content")
    if not isinstance(content, list):
        raise ToolError("spokenly", "Spokenly response is missing MCP content")
    text_blocks = [
        block.get("text")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if len(text_blocks) != 1:
        raise ToolError(
            "spokenly",
            f"Spokenly returned {len(text_blocks)} text blocks; expected exactly one JSON transcript",
        )
    try:
        transcript = json.loads(text_blocks[0])
    except json.JSONDecodeError as err:
        raise ToolError(
            "spokenly", f"Spokenly transcript content is invalid JSON: {err}"
        ) from err
    if not isinstance(transcript, dict):
        raise ToolError("spokenly", "Spokenly transcript content is not a JSON object")
    return transcript


def spokenly_transcript_text(transcript: dict[str, Any]) -> str:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ToolError(
            "spokenly", "Spokenly transcript is missing timestamped segments"
        )
    parts: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ToolError("spokenly", f"Spokenly segment {index} is not an object")
        if not isinstance(segment.get("text"), str):
            raise ToolError("spokenly", f"Spokenly segment {index} is missing text")
        if not isinstance(segment.get("start"), (int, float)) or not isinstance(
            segment.get("end"), (int, float)
        ):
            raise ToolError(
                "spokenly", f"Spokenly segment {index} is missing timestamps"
            )
        parts.append(segment["text"])
    text = normalize_spaces(" ".join(parts))
    if not text:
        raise ToolError("spokenly", "Spokenly returned an empty transcript")
    return text


def convert_audio(
    env: EnvPaths, input_path: pathlib.Path, output_wav: pathlib.Path
) -> None:
    if env.ffmpeg is None:
        raise ToolError("convert", "ffmpeg is required for audio conversion")
    run(
        [
            str(env.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav),
        ],
        "convert",
    )


def choose_model(model_arg: str, backend: str, env: EnvPaths) -> SelectedModel:
    if backend == "spokenly":
        return choose_spokenly_model(model_arg, env)
    if backend == "mlx-whisper":
        return choose_mlx_model(model_arg)
    return choose_whisper_cpp_model(model_arg, env)


def choose_spokenly_model(model_arg: str, env: EnvPaths) -> SelectedModel:
    aliases = {
        "auto",
        "parakeet",
        "parakeet-tdt",
        "parakeet-tdt-0.6b-v3",
        SPOKENLY_TDT_MODEL_ID,
    }
    if model_arg not in aliases:
        raise ToolError(
            "model",
            "Spokenly backend supports only NVIDIA Parakeet TDT 0.6B V3 "
            f"({SPOKENLY_TDT_MODEL_ID}); got {model_arg}.",
        )
    return SelectedModel(
        name="parakeet-tdt-0.6b-v3",
        locator=SPOKENLY_TDT_MODEL_ID,
        path=env.spokenly_tdt_model_dir,
    )


def choose_mlx_model(model_arg: str) -> SelectedModel:
    if model_arg == "auto":
        env_model = os.environ.get("TRANSCRIBE_AUDIO_MLX_MODEL")
        if env_model:
            return SelectedModel(name=model_label(env_model), locator=env_model)
        repo = MLX_MODELS["large-v3"]["repo"]
        return SelectedModel(name="large-v3", locator=repo)
    expanded = expand_path(pathlib.Path(model_arg))
    if expanded.exists():
        return SelectedModel(
            name=model_label(model_arg), locator=str(expanded), path=expanded
        )
    if model_arg in MLX_MODELS:
        return SelectedModel(name=model_arg, locator=MLX_MODELS[model_arg]["repo"])
    if "/" in model_arg:
        return SelectedModel(name=model_label(model_arg), locator=model_arg)
    raise ToolError(
        "model",
        f"MLX model is not known: {model_arg}. Known MLX models: {', '.join(MLX_MODELS)}.",
    )


def choose_whisper_cpp_model(model_arg: str, env: EnvPaths) -> SelectedModel:
    if model_arg == "auto":
        env_model = os.environ.get("TRANSCRIBE_AUDIO_MODEL")
        if env_model:
            selected = whisper_cpp_model_path_for(env_model, env)
            if selected:
                return selected
            raise ToolError(
                "model", f"TRANSCRIBE_AUDIO_MODEL does not exist: {env_model}"
            )
        for name in WHISPER_CPP_MODELS:
            selected = whisper_cpp_model_path_for(name, env)
            if selected:
                return selected
        raise ToolError(
            "model",
            "no whisper.cpp model found. Run `transcribe-audio download-model large-v3 --backend whisper-cpp`.",
        )
    selected = whisper_cpp_model_path_for(model_arg, env)
    if selected:
        return selected
    raise ToolError(
        "model",
        f"whisper.cpp model is not installed: {model_arg}. Run `transcribe-audio download-model {model_arg} --backend whisper-cpp` or pass a model path.",
    )


def whisper_cpp_model_path_for(
    name_or_path: str, env: EnvPaths
) -> SelectedModel | None:
    expanded = expand_path(pathlib.Path(name_or_path))
    if expanded.exists():
        return SelectedModel(name=expanded.name, locator=str(expanded), path=expanded)
    meta = WHISPER_CPP_MODELS.get(name_or_path)
    if meta is None:
        return None
    candidate = env.model_dir / meta["file"]
    if candidate.exists():
        return SelectedModel(name=name_or_path, locator=str(candidate), path=candidate)
    if name_or_path == "small" and env.superwhisper_small.exists():
        return SelectedModel(
            name="small",
            locator=str(env.superwhisper_small),
            path=env.superwhisper_small,
        )
    return None


def env_paths() -> EnvPaths:
    return EnvPaths(
        model_dir=expand_path(
            pathlib.Path(
                os.environ.get("TRANSCRIBE_AUDIO_MODEL_DIR", DEFAULT_MODEL_DIR)
            )
        ),
        superwhisper_small=expand_path(pathlib.Path(SUPERWHISPER_SMALL)),
        spokenly_app=expand_path(
            pathlib.Path(os.environ.get("SPOKENLY_APP", DEFAULT_SPOKENLY_APP))
        ),
        spokenly_tdt_model_dir=expand_path(
            pathlib.Path(
                os.environ.get(
                    "SPOKENLY_TDT_MODEL_DIR",
                    DEFAULT_SPOKENLY_TDT_MODEL_DIR,
                )
            )
        ),
        spokenly_mcp_url=os.environ.get("SPOKENLY_MCP_URL", DEFAULT_SPOKENLY_MCP_URL),
        ffmpeg=env_tool("FFMPEG", "ffmpeg"),
        ffprobe=env_tool("FFPROBE", "ffprobe"),
        whisper_cli=env_tool("WHISPER_CLI", "whisper-cli"),
        uv=env_tool("UV", "uv"),
        mlx_whisper_python=pathlib.Path(os.environ["MLX_WHISPER_PYTHON"]).expanduser()
        if os.environ.get("MLX_WHISPER_PYTHON")
        else None,
        curl=env_tool("CURL", "curl"),
    )


def env_tool(env_name: str, command: str) -> pathlib.Path | None:
    if os.environ.get(env_name):
        return pathlib.Path(os.environ[env_name]).expanduser()
    found = shutil.which(command)
    return pathlib.Path(found) if found else None


def ensure_tools(env: EnvPaths, backend: str) -> None:
    missing = []
    if env.ffmpeg is None:
        missing.append("ffmpeg")
    if backend == "mlx-whisper" and not mlx_runner_available(env):
        missing.append("uv or MLX_WHISPER_PYTHON")
    if backend == "whisper-cpp" and env.whisper_cli is None:
        missing.append("whisper-cli")
    if backend == "spokenly":
        if not env.spokenly_app.exists():
            missing.append("Spokenly.app")
        if not env.spokenly_tdt_model_dir.exists():
            missing.append("NVIDIA Parakeet TDT 0.6B V3 in Spokenly")
        bridge_available, _ = spokenly_bridge_status(env)
        if not bridge_available:
            missing.append("Spokenly local MCP bridge")
        selected_model = spokenly_selected_file_model_id()
        if selected_model != SPOKENLY_TDT_MODEL_ID:
            missing.append(
                "NVIDIA Parakeet TDT 0.6B V3 selected for Spokenly file transcription "
                f"(found {selected_model or 'unknown'})"
            )
    if missing:
        raise ToolError("doctor", f"missing required tool(s): {', '.join(missing)}")


def mlx_runner_available(env: EnvPaths) -> bool:
    return env.mlx_whisper_python is not None or env.uv is not None


def spokenly_bridge_status(env: EnvPaths) -> tuple[bool, str | None]:
    try:
        response = spokenly_rpc(
            env.spokenly_mcp_url,
            "tools/list",
            {},
            timeout=2.0,
        )
        tools = response.get("result", {}).get("tools")
        if not isinstance(tools, list):
            return False, "tools/list response is missing tools"
        available = any(
            isinstance(tool, dict) and tool.get("name") == "transcribe_file"
            for tool in tools
        )
        if not available:
            return False, "transcribe_file tool is unavailable"
        return True, None
    except ToolError as err:
        return False, err.message


def backend_ready(
    env: EnvPaths,
    backend: str,
    *,
    spokenly_bridge_available: bool,
    spokenly_selected_model: str | None,
) -> bool:
    if env.ffmpeg is None:
        return False
    if backend == "spokenly":
        return (
            env.spokenly_app.exists()
            and env.spokenly_tdt_model_dir.exists()
            and spokenly_bridge_available
            and spokenly_selected_model == SPOKENLY_TDT_MODEL_ID
        )
    if backend == "mlx-whisper":
        return mlx_runner_available(env)
    return env.whisper_cli is not None


def spokenly_selected_file_model_id() -> str | None:
    defaults = shutil.which("defaults")
    if defaults is None:
        return None
    try:
        completed = subprocess.run(
            [defaults, "export", "app.spokenly", "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        preferences = plistlib.loads(completed.stdout)
    except plistlib.InvalidFileException:
        return None
    if not isinstance(preferences, dict):
        return None
    return decode_spokenly_model_preference(
        preferences.get("fileTranscriptionVoiceModelID")
    )


def decode_spokenly_model_preference(value: Any) -> str | None:
    if isinstance(value, bytes):
        try:
            raw = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(value, str):
        raw = value
    else:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) else None


def mlx_python_command(env: EnvPaths) -> list[str]:
    if env.mlx_whisper_python:
        return [str(env.mlx_whisper_python)]
    if env.uv:
        return [
            str(env.uv),
            "run",
            "--quiet",
            "--with",
            "mlx-whisper",
            "--with",
            "huggingface-hub",
            "--python",
            DEFAULT_MLX_PYTHON,
            "python",
        ]
    raise ToolError(
        "doctor", "MLX transcription requires uv or MLX_WHISPER_PYTHON=/path/to/python"
    )


def installed_models(env: EnvPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "backend": "spokenly",
            "name": "parakeet-tdt-0.6b-v3",
            "file": SPOKENLY_TDT_MODEL_ID,
            "installed": env.spokenly_tdt_model_dir.exists(),
            "paths": [str(env.spokenly_tdt_model_dir)]
            if env.spokenly_tdt_model_dir.exists()
            else [],
            "note": SPOKENLY_MODELS["parakeet-tdt-0.6b-v3"]["note"],
        }
    )
    for name, meta in MLX_MODELS.items():
        cache = mlx_model_cache_path(meta["repo"])
        paths = [str(cache)] if cache.exists() else []
        rows.append(
            {
                "backend": "mlx-whisper",
                "name": name,
                "file": meta["repo"],
                "installed": bool(paths),
                "paths": paths,
                "note": meta["note"],
            }
        )
    for name, meta in WHISPER_CPP_MODELS.items():
        candidates = [env.model_dir / meta["file"]]
        if name == "small":
            candidates.append(env.superwhisper_small)
        paths = [str(path) for path in candidates if path.exists()]
        rows.append(
            {
                "backend": "whisper-cpp",
                "name": name,
                "file": meta["file"],
                "installed": bool(paths),
                "paths": paths,
                "note": meta["note"],
            }
        )
    return rows


def normalize_backend(raw: str) -> str:
    if raw in {"spokenly", "parakeet", "parakeet-tdt"}:
        return "spokenly"
    if raw in {"mlx", "mlx-whisper"}:
        return "mlx-whisper"
    if raw in {"whisper-cpp", "whisper.cpp", "cpp"}:
        return "whisper-cpp"
    raise ToolError(
        "cli",
        f"unsupported backend: {raw}. Use spokenly, mlx, or whisper-cpp.",
    )


def parse_formats(raw: str) -> list[str]:
    values = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value not in SUPPORTED_FORMATS:
            raise ToolError("cli", f"unsupported output format: {value}")
        if value not in values:
            values.append(value)
    return values or ["txt", "json"]


def validate_formats(formats: list[str], backend: str) -> None:
    if backend == "mlx-whisper":
        unsupported = [fmt for fmt in formats if fmt not in MLX_SUPPORTED_FORMATS]
        if unsupported:
            raise ToolError(
                "cli",
                f"MLX backend supports only txt,json formats. Unsupported: {', '.join(unsupported)}",
            )
    if backend == "spokenly":
        unsupported = [fmt for fmt in formats if fmt not in SPOKENLY_SUPPORTED_FORMATS]
        if unsupported:
            raise ToolError(
                "cli",
                "Spokenly Parakeet TDT supports only txt,json formats. Unsupported: "
                f"{', '.join(unsupported)}",
            )


def validate_spokenly_options(language: str, prompt: str | None) -> None:
    if language.lower() not in {"auto", "en", "english"}:
        raise ToolError(
            "language",
            f"Spokenly Parakeet TDT supports English only; got {language}.",
        )
    if prompt:
        raise ToolError(
            "prompt",
            "Spokenly Parakeet TDT does not support transcription prompts; remove "
            "--prompt/--prompt-file and review names from the timestamped JSON instead.",
        )


def resolve_prompt(prompt: str | None, prompt_file: pathlib.Path | None) -> str | None:
    file_prompt = None
    if prompt_file:
        prompt_path = expand_path(prompt_file)
        try:
            file_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as err:
            raise ToolError(
                "prompt", f"failed to read prompt file {prompt_path}: {err}"
            ) from err
    inline = prompt.strip() if prompt else None
    parts = [part for part in [file_prompt, inline] if part]
    return "\n\n".join(parts) if parts else None


def run(
    cmd: list[str],
    stage: str,
    *,
    quiet: bool = False,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    printable = subprocess.list2cmdline(cmd)
    stdout = subprocess.PIPE if capture else (subprocess.DEVNULL if quiet else None)
    stderr = subprocess.PIPE if capture else (subprocess.DEVNULL if quiet else None)
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as err:
        raise ToolError(
            stage, f"failed to start command: {err}", command=printable
        ) from err
    if completed.returncode != 0:
        detail = completed.stderr.strip() if capture and completed.stderr else printable
        raise ToolError(
            stage,
            f"command failed ({completed.returncode}): {detail}",
            command=printable,
            exit_code=completed.returncode,
        )
    return completed


def wait_for_outputs(output_base: pathlib.Path, formats: list[str]) -> dict[str, str]:
    deadline = time.time() + 10
    while True:
        outputs = {
            fmt: str(output_path_for_format(output_base, fmt))
            for fmt in formats
            if output_path_for_format(output_base, fmt).exists()
        }
        if len(outputs) == len(formats):
            return outputs
        if time.time() >= deadline:
            missing = [fmt for fmt in formats if fmt not in outputs]
            raise ToolError(
                "outputs", f"missing transcript output(s): {', '.join(missing)}"
            )
        time.sleep(0.05)


def probe_duration_seconds(env: EnvPaths, path: pathlib.Path) -> float | None:
    if env.ffprobe is None:
        return None
    completed = run(
        [
            str(env.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "probe",
        capture=True,
    )
    try:
        return round(float(completed.stdout.strip()), 3)
    except ValueError:
        return None


def discover_audio(
    root: pathlib.Path, *, recursive: bool, min_mtime: float | None
) -> list[dict[str, Any]]:
    pattern = "**/*" if recursive else "*"
    rows = []
    for path in root.glob(pattern):
        if not path.is_file() or not is_supported_audio_path(path):
            continue
        stat = path.stat()
        if min_mtime is not None and stat.st_mtime < min_mtime:
            continue
        rows.append(
            {
                "path": str(path),
                "modified_unix_seconds": int(stat.st_mtime),
                "size_bytes": stat.st_size,
            }
        )
    return rows


def read_json_file(path: pathlib.Path, *, stage: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ToolError(stage, f"failed to read JSON {path}: {err}") from err
    except json.JSONDecodeError as err:
        raise ToolError(stage, f"failed to parse JSON {path}: {err}") from err


def build_review_windows(
    data: dict[str, Any],
    *,
    phrases: list[str],
    min_word_probability: float,
    context_seconds: float,
    merge_gap_seconds: float,
    max_windows: int,
) -> list[dict[str, Any]]:
    segments = transcript_segments(data)
    raw_windows: list[dict[str, Any]] = []
    normalized_phrases = [
        normalize_spaces(phrase).lower()
        for phrase in phrases
        if normalize_spaces(phrase)
    ]
    for segment in segments:
        text = segment["text"]
        lower_text = normalize_spaces(text).lower()
        for phrase in normalized_phrases:
            if phrase in lower_text:
                raw_windows.append(
                    review_window(
                        segment["start"],
                        segment["end"],
                        context_seconds,
                        text,
                        {
                            "code": "phrase",
                            "phrase": phrase,
                        },
                        priority=0,
                    )
                )
        low_words = [
            word
            for word in segment.get("words", [])
            if word.get("probability") is not None
            and float(word["probability"]) < min_word_probability
            and word.get("start") is not None
            and word.get("end") is not None
        ]
        if low_words:
            raw_windows.append(
                review_window(
                    min(float(word["start"]) for word in low_words),
                    max(float(word["end"]) for word in low_words),
                    context_seconds,
                    text,
                    {
                        "code": "low-word-probability",
                        "threshold": min_word_probability,
                        "count": len(low_words),
                        "min_probability": round(
                            min(float(word["probability"]) for word in low_words), 4
                        ),
                        "words": [
                            {
                                "word": normalize_spaces(str(word.get("word", ""))),
                                "probability": round(float(word["probability"]), 4),
                            }
                            for word in low_words[:12]
                        ],
                    },
                    priority=1,
                )
            )
    merged = merge_review_windows(raw_windows, merge_gap_seconds=merge_gap_seconds)
    selected = sorted(
        merged, key=lambda row: (row["_priority"], row["start"], row["end"])
    )[: max(max_windows, 0)]
    selected.sort(key=lambda row: (row["start"], row["end"]))
    for index, window in enumerate(selected, start=1):
        window["id"] = f"review-{index:03d}"
        window["duration_seconds"] = round(window["end"] - window["start"], 3)
        window.pop("_priority", None)
    return selected


def transcript_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("segments"), list):
        rows = []
        for segment in data["segments"]:
            if "start" not in segment or "end" not in segment:
                continue
            rows.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": normalize_spaces(str(segment.get("text", ""))),
                    "words": segment.get("words", []),
                }
            )
        if rows:
            return rows
    if isinstance(data.get("transcription"), list):
        rows = []
        for segment in data["transcription"]:
            offsets = segment.get("offsets", {})
            if "from" not in offsets or "to" not in offsets:
                continue
            rows.append(
                {
                    "start": float(offsets["from"]) / 1000.0,
                    "end": float(offsets["to"]) / 1000.0,
                    "text": normalize_spaces(str(segment.get("text", ""))),
                    "words": [],
                }
            )
        if rows:
            return rows
    raise ToolError("review", "transcript JSON does not contain timestamped segments")


def review_window(
    start: float,
    end: float,
    context_seconds: float,
    text: str,
    reason: dict[str, Any],
    *,
    priority: int,
) -> dict[str, Any]:
    window_start = max(0.0, start - max(context_seconds, 0.0))
    window_end = max(window_start, end + max(context_seconds, 0.0))
    return {
        "start": round(window_start, 3),
        "end": round(window_end, 3),
        "text": normalize_spaces(text),
        "reasons": [reason],
        "_priority": priority,
    }


def merge_review_windows(
    windows: list[dict[str, Any]], *, merge_gap_seconds: float
) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda row: (row["start"], row["end"]))
    merged: list[dict[str, Any]] = [ordered[0]]
    for window in ordered[1:]:
        current = merged[-1]
        if window["start"] <= current["end"] + max(merge_gap_seconds, 0.0):
            current["end"] = max(current["end"], window["end"])
            current["text"] = join_unique_text(current["text"], window["text"])
            current["reasons"].extend(window["reasons"])
            current["_priority"] = min(current["_priority"], window["_priority"])
            continue
        merged.append(window)
    return merged


def join_unique_text(left: str, right: str) -> str:
    left = normalize_spaces(left)
    right = normalize_spaces(right)
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left} {right}"


def review_window_label(index: int, window: dict[str, Any]) -> str:
    words = re.findall(r"[A-Za-z0-9]+", window.get("text", ""))
    suffix = slugify(" ".join(words[:6])) if words else "clip"
    return f"{index:03d}-{suffix}"


def extract_audio_clip(
    env: EnvPaths, audio: pathlib.Path, output: pathlib.Path, start: float, end: float
) -> None:
    duration = max(end - start, 0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(env.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        "review",
    )


def review_transcribe_commands(
    clip: pathlib.Path, output_dir: pathlib.Path, label: str
) -> dict[str, list[str]]:
    return {
        "parakeet_tdt": [
            "transcribe-audio",
            "transcribe",
            str(clip),
            "--backend",
            "spokenly",
            "--model",
            SPOKENLY_TDT_MODEL_ID,
            "--language",
            "en",
            "--formats",
            "txt,json",
            "--output-dir",
            str(output_dir),
            "--output-name",
            f"{label}-parakeet-tdt",
            "--json",
        ],
        "mlx_large_v3": [
            "transcribe-audio",
            "transcribe",
            str(clip),
            "--backend",
            "mlx",
            "--model",
            "large-v3",
            "--language",
            "en",
            "--formats",
            "txt,json",
            "--output-dir",
            str(output_dir),
            "--output-name",
            f"{label}-mlx-large-v3",
            "--json",
        ],
    }


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def collect_transcript_inputs(
    inputs: list[pathlib.Path],
) -> list[tuple[str, pathlib.Path]]:
    transcripts = []
    for raw in inputs:
        path = expand_path(raw)
        if path.is_dir():
            for entry in sorted(path.glob("*.txt")):
                transcripts.append((input_label(entry), entry))
        elif path.exists():
            transcripts.append((input_label(path), path))
        else:
            raise ToolError("combine", f"transcript input does not exist: {path}")
    if not transcripts:
        raise ToolError("combine", "no transcript .txt files found")
    return transcripts


def write_combined_markdown(
    inputs: list[tuple[str, pathlib.Path]], output: pathlib.Path, title: str
) -> None:
    if not inputs:
        raise ToolError("combine", "no transcript inputs to combine")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title.strip()}", ""]
    for label, path in inputs:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as err:
            raise ToolError(
                "combine", f"failed to read transcript {path}: {err}"
            ) from err
        parts.extend([f"## {label.strip()}", "", content, ""])
    output.write_text("\n".join(parts), encoding="utf-8")


def transcript_quality_warnings(text: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if "[BLANK_AUDIO]" in text:
        warnings.append(
            {
                "code": "blank-audio-marker",
                "message": "Transcript contains [BLANK_AUDIO], often a silence hallucination marker.",
                "count": text.count("[BLANK_AUDIO]"),
            }
        )
    thanks_count = len(
        re.findall(r"\b(thank you|thanks for watching)\b", text, flags=re.IGNORECASE)
    )
    if thanks_count:
        warnings.append(
            {
                "code": "thanks-marker",
                "message": "Transcript contains generic thanks phrases that can be hallucinated during silence.",
                "count": thanks_count,
            }
        )
    repeats = adjacent_repeated_line_count(text)
    if repeats:
        warnings.append(
            {
                "code": "adjacent-repeated-line",
                "message": "Transcript has adjacent repeated lines.",
                "count": repeats,
            }
        )
    return warnings


def adjacent_repeated_line_count(text: str) -> int:
    repeats = 0
    previous = None
    streak = 1
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if not normalized:
            continue
        if normalized == previous:
            streak += 1
            if streak >= 3:
                repeats += 1
        else:
            previous = normalized
            streak = 1
    return repeats


def parse_age(raw: str) -> int:
    match = re.fullmatch(r"(\d+)([smhdw])", raw.strip())
    if not match:
        raise ToolError(
            "cli", f"invalid --since value: {raw}. Use formats like 30m, 6h, 2d, or 1w."
        )
    value = int(match.group(1))
    unit = match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def slugify(value: str) -> str:
    stem = pathlib.Path(value).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return slug or "transcript"


def unique_output_name(
    input_path: pathlib.Path,
    index: int,
    output_dir: pathlib.Path,
    reserved: set[str] | None = None,
) -> str:
    base = slugify(input_path.name)
    if output_name_available(output_dir, base, reserved):
        return base
    candidate = f"{index:03}-{base}"
    if output_name_available(output_dir, candidate, reserved):
        return candidate
    suffix = 2
    while True:
        candidate = f"{index:03}-{suffix}-{base}"
        if output_name_available(output_dir, candidate, reserved):
            return candidate
        suffix += 1


def output_name_available(
    output_dir: pathlib.Path,
    output_name: str,
    reserved: set[str] | None = None,
) -> bool:
    if reserved is not None and output_name in reserved:
        return False
    formats = sorted(SUPPORTED_FORMATS | {"md", "wav"})
    return not any(
        output_path_for_format(output_dir / output_name, fmt).exists()
        for fmt in formats
    )


def output_path_for_format(output_base: pathlib.Path, fmt: str) -> pathlib.Path:
    return pathlib.Path(f"{output_base}.{fmt}")


def input_label(path: pathlib.Path) -> str:
    return path.stem or path.name


def model_label(value: str) -> str:
    return pathlib.Path(value).name or value


def mlx_model_cache_path(repo_id: str) -> pathlib.Path:
    return (
        pathlib.Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
    )


def expand_path(path: pathlib.Path | None) -> pathlib.Path:
    if path is None:
        raise ToolError("cli", "missing path")
    return pathlib.Path(os.path.expandvars(str(path))).expanduser()


def is_supported_audio_path(path: pathlib.Path) -> bool:
    return path.suffix.lower().lstrip(".") in SUPPORTED_AUDIO_EXTENSIONS


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def error_payload(err: ToolError) -> dict[str, Any]:
    payload = {"ok": False, "stage": err.stage, "error": err.message}
    if err.command:
        payload["command"] = err.command
    if err.exit_code is not None:
        payload["exit_code"] = err.exit_code
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
