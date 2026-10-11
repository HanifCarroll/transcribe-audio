# transcribe-audio

Local-first audio transcription workbench for coding agents and personal
automation.

`transcribe-audio` is a Python CLI for turning local recordings into inspectable
transcript artifacts. It supports [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
Spokenly's on-device Apple Speech Analyzer and Parakeet TDT file transcription, and whisper.cpp. It keeps
`txt` and `json` outputs for agent review and includes helpers for silence
preprocessing, transcript quality checks, batch runs, timestamped review clips,
and Obsidian-friendly Markdown notes.

The tool is intentionally a workbench instead of a fully automatic note taker:
it handles repeatable audio plumbing while leaving judgment about prompts,
preprocessing, retries, titles, and final notes to the agent or human running it.

## Set It Up With An Agent

Copy this prompt into your coding agent:

```text
Set up this local transcription workbench for me:
https://github.com/HanifCarroll/transcribe-audio

Clone the repo if needed, install the Python CLI with uv, run the doctor check,
select the backend for the task, and verify its doctor check. Download large-v3
only when using MLX. Use the repository README and CLI help as the operating
contract. Keep audio files, transcripts, generated artifacts, and model files
out of git.
```

## Features

- MLX Whisper `large-v3` default for high-quality Apple Silicon transcription
- Optional Spokenly backend with Apple Speech Analyzer or NVIDIA Parakeet TDT 0.6B V3 for local English transcription
- Optional whisper.cpp fallback for comparison and subtitle formats
- Audio normalization through `ffmpeg`
- Silence-removal preprocessing for recordings with long dead air
- JSON summaries designed for automation and coding agents
- Batch transcription with a run manifest and optional parallel workers
- Lightweight quality warnings for common silence hallucination markers
- Timestamped review windows and clip extraction for uncertain transcript text
- Markdown note writer with date, source, backend, model, and raw artifact
  provenance

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [`ffmpeg`](https://ffmpeg.org/)
- Apple Silicon for the default MLX backend
- Optional: [Spokenly](https://spokenly.app/) with the requested Apple Speech Analyzer or Parakeet TDT file model selected and its local MCP bridge enabled
- Optional: `whisper-cli` from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  for fallback comparisons, SRT, and VTT output

## Install

From this checkout:

```sh
uv tool install --force .
```

Or run without installing:

```sh
uv run transcribe-audio doctor --json
```

Check the local environment:

```sh
transcribe-audio doctor --json
```

## Defaults

CLI default backend:

```text
mlx-whisper
```

Default model:

```text
mlx-community/whisper-large-v3-mlx
```

Download the default model:

```sh
transcribe-audio download-model large-v3
```

The CLI runs MLX through `uv` automatically:

```sh
uv run --with mlx-whisper --with huggingface-hub --python 3.11 python
```

To use an existing Python environment with `mlx-whisper` installed:

```sh
MLX_WHISPER_PYTHON=/path/to/python transcribe-audio audio.m4a
```

Whisper.cpp models are stored in:

```text
~/.local/share/transcribe-audio/models
```

## Quick Start

Transcribe one file with the MLX default:

```sh
transcribe-audio "/path/to/audio.m4a" --json
```

Use a prompt for names, products, or technical vocabulary:

```sh
transcribe-audio "/path/to/audio.m4a" \
  --prompt "Names: Casamo, Codex, MLX Whisper" \
  --json
```

Use Spokenly's Apple Speech Analyzer backend when the app and local MCP bridge are available:

```sh
transcribe-audio doctor --backend spokenly --json
transcribe-audio transcribe "/path/to/audio.m4a" \
  --backend spokenly \
  --model apple-speech-analyzer \
  --formats txt,json \
  --json
```

Apple Speech Analyzer supports English `txt,json` output and does not accept transcription prompts. The command fails if Spokenly reports a different selected file model. NVIDIA Parakeet TDT remains available with `--model parakeetTDT06`.

Preprocess long silence before retrying:

```sh
transcribe-audio preprocess "/path/to/audio.m4a" --json
```

Check a transcript for common hallucination markers:

```sh
transcribe-audio quality ./transcripts/audio.txt --json
```

Transcribe several independent files at once:

```sh
transcribe-audio batch "/path/one.m4a" "/path/two.m4a" \
  --output-dir ./transcripts/current \
  --jobs 2 \
  --json
```

Build review windows for low-confidence words or suspect phrases, then extract
short clips for targeted re-transcription:

```sh
transcribe-audio review ./transcripts/audio.json \
  --audio ./transcripts/audio-desilenced.wav \
  --phrase "Probably the trackpad" \
  --extract-clips \
  --output-dir ./transcripts/review \
  --json
```

Wrap a transcript into an Obsidian note with frontmatter:

```sh
transcribe-audio note ./transcripts/audio.txt \
  --output "/path/to/vault/sources/voice-journal/Descriptive Title.md" \
  --title "Descriptive Title" \
  --date 2026-07-03 \
  --source "/path/to/audio.m4a" \
  --backend mlx-whisper \
  --model large-v3
```

## Agent Workflow

1. Read the destination workflow and select its backend explicitly. For Hanif’s English vault imports, prefer `--backend spokenly --model apple-speech-analyzer`; the CLI default remains MLX. Run the matching doctor check.
2. Use `discover` if the user points at a folder or vague location.
3. Use `preprocess` when long silence is likely or quality warnings appear.
4. Run `transcribe` with the selected backend/model and keep `txt,json` artifacts.
5. For multiple independent files, use `batch`; MLX/whisper.cpp can use `--jobs 2` for raw transcription, while Spokenly requires one worker.
6. Run `quality` on the text transcript.
7. When raw JSON contains timestamps, use `review` for garbled text or uncertain names/tools. Use the same audio that was transcribed, including the desilenced file when used.
8. Inspect uncertain audio or use targeted re-transcription as evidence before correcting the note copy. Model agreement and quality warnings do not establish the original wording. The agent reviews and edits text directly.
9. Choose a descriptive note title from the content.
10. Run `note` with the recording date, original source locator, backend and model. Include raw artifact paths only when the destination retains those files; for temporary evidence, put recording identity and quality warnings in the note instead.
11. If warnings or review uncertainties remain, report them explicitly.

The destination owns reviewed-note creation, discovery scope, retention and downstream follow-up. For Hanif’s vault, read `sources/voice-journal/AGENTS.md`: save the reviewed Markdown note and any required daily link; keep intermediate TXT/JSON and derived audio in a temporary directory outside the vault and delete them after verification. Do not create raw folders or import ledgers there. Match recordings against existing note metadata before importing.

## Optional import ledger

Use this facility only for a destination whose owning workflow explicitly retains raw evidence and an import ledger. Hanif’s vault uses note metadata instead. The package defines a `walky-talky-ledger` entry point. From this checkout, use `python -m transcribe_audio.walky_talky_ledger`; its implementation is `transcribe_audio/walky_talky_ledger.py`. Run its `--help` and each subcommand’s help for required fields. `scan` finds unimported recordings, `append` records a completed import, and `cleanup-audio` removes derived audio within a completed raw run while preserving the supplied source.

Automatic discovery defaults to Apple Voice Memos; it excludes deleted recordings and identifies imported sources by their saved hashes. Use `--root` only for a source Hanif supplies. A Voice Memos scan can specify the root explicitly:

```sh
python -m transcribe_audio.walky_talky_ledger scan --vault "/path/to/vault" \
  --root "$HOME/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings" --json
```

The ledger defaults to `.transcribe-audio/imports.jsonl` under the supplied vault. Append only after the reviewed note, raw TXT/JSON and any required journal link exist. Cleanup requires a single run under that vault’s `sources/walky-talky/raw/` and an existing original source; preserve original recordings and non-audio evidence. These commands do not authorize task extraction or downstream processing.

## Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Check required tools and installed models |
| `models` | List known Spokenly, MLX, and whisper.cpp models |
| `download-model` | Download/cache a known model |
| `discover` | Find recent audio files in a folder |
| `preprocess` | Remove long silence from an audio file |
| `transcribe` | Transcribe one audio file |
| `batch` | Transcribe multiple files into a run directory; use `--jobs N` for parallel raw transcription |
| `quality` | Flag common hallucination and repetition markers |
| `review` | Find uncertain timestamp windows and optionally extract clips |
| `combine` | Combine transcript text files into Markdown |
| `note` | Write a vault-ready Markdown transcript note |

Examples:

```sh
transcribe-audio models --json
transcribe-audio discover ~/Downloads --since 2d --json
transcribe-audio transcribe "/path/to/audio.m4a" --formats txt,json --json
transcribe-audio batch "/path/one.m4a" "/path/two.m4a" --output-dir ./transcripts/current --jobs 2 --json
transcribe-audio combine ./transcripts/current --output ./transcripts/current/transcripts.md
```

Use whisper.cpp explicitly:

```sh
transcribe-audio download-model large-v3 --backend whisper-cpp
transcribe-audio transcribe "/path/to/audio.m4a" \
  --backend whisper-cpp \
  --model large-v3 \
  --formats txt,json,srt \
  --json
```

## Output And Privacy

Transcripts, audio files, converted WAVs, and run artifacts are ignored by git.
The CLI does not send audio to hosted transcription APIs; transcription runs
locally through MLX Whisper, Spokenly's on-device model bridge, or whisper.cpp.

JSON mode returns structured failures:

```json
{
  "ok": false,
  "stage": "input",
  "error": "input file does not exist: /tmp/missing.m4a"
}
```

Subprocess failures include the command and exit code when available.

## Development

Run tests:

```sh
python -m unittest discover -s tests
```

Smoke-check the CLI:

```sh
uv run transcribe-audio doctor --json
uv run transcribe-audio --help
```

The repository is safe to publish publicly. Do not commit local audio,
transcripts, model files, generated run artifacts, credentials, or private
configuration.

## License

MIT
