# transcribe-audio

Local-first audio transcription workbench for coding agents and personal
automation.

`transcribe-audio` is a Python CLI for turning local recordings into inspectable
transcript artifacts. It uses [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
by default on Apple Silicon, keeps `txt` and `json` outputs for agent review, and
includes helpers for silence preprocessing, transcript quality checks, batch
runs, timestamped review clips, and Obsidian-friendly Markdown notes.

The tool is intentionally a workbench instead of a fully automatic note taker:
it handles repeatable audio plumbing while leaving judgment about prompts,
preprocessing, retries, titles, and final notes to the agent or human running it.

## Set It Up With An Agent

Copy this prompt into your coding agent:

```text
Set up this local transcription workbench for me:
https://github.com/HanifCarroll/transcribe-audio

Clone the repo if needed, install the Python CLI with uv, run the doctor check,
download the default MLX large-v3 model, and verify that `transcribe-audio
doctor --json` works. If your environment supports skills, install or adapt the
public skill at `skills/transcribe-audio-to-vault/SKILL.md` so future agents can
use the CLI to turn audio files into sourced Markdown notes. Keep audio files,
transcripts, generated artifacts, and model files out of git.
```

## Features

- MLX Whisper `large-v3` default for high-quality Apple Silicon transcription
- Optional whisper.cpp fallback for comparison and subtitle formats
- Audio normalization through `ffmpeg`
- Silence-removal preprocessing for recordings with long dead air
- JSON summaries designed for automation and coding agents
- Batch transcription with a run manifest
- Lightweight quality warnings for common silence hallucination markers
- Timestamped review windows and clip extraction for uncertain transcript text
- Markdown note writer with date, source, backend, model, and raw artifact
  provenance

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [`ffmpeg`](https://ffmpeg.org/)
- Apple Silicon for the default MLX backend
- Optional: `whisper-cli` from [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
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

Default backend:

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

Preprocess long silence before retrying:

```sh
transcribe-audio preprocess "/path/to/audio.m4a" --json
```

Check a transcript for common hallucination markers:

```sh
transcribe-audio quality ./transcripts/audio.txt --json
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
  --output "./inbox/walky-talky/Descriptive Title.md" \
  --title "Descriptive Title" \
  --date 2026-07-03 \
  --source "/path/to/audio.m4a" \
  --backend mlx-whisper \
  --model large-v3 \
  --raw-json ./transcripts/audio.json
```

## Agent Workflow

1. Run `transcribe-audio doctor --json`.
2. Use `discover` if the user points at a folder or vague location.
3. Use `preprocess` when long silence is likely or quality warnings appear.
4. Run `transcribe` with the MLX default and keep `txt,json` artifacts.
5. Run `quality` on the text transcript.
6. Run `review` on raw JSON when text looks garbled or names/tools are
   uncertain. Use the same audio file that was transcribed, such as the
   preprocessed/desilenced file when one was used.
7. Use extracted clips and emitted re-transcription command arguments to compare
   MLX and whisper.cpp outputs before correcting the note copy.
8. Choose a descriptive note title from the content.
9. Run `note` with date, source, backend, model, and raw JSON provenance.
10. If warnings or review uncertainties remain, report them explicitly.

## Agent Skill

This repo includes a public Codex-style skill at:

```text
skills/transcribe-audio-to-vault/SKILL.md
```

The skill gives agents a repeatable workflow for using this CLI to turn audio
files into reviewed, sourced Markdown notes. It assumes `transcribe-audio` is
installed on `PATH` or can be run from a checkout with `uv run`.

## Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Check required tools and installed models |
| `models` | List known MLX and whisper.cpp models |
| `download-model` | Download/cache a known model |
| `discover` | Find recent audio files in a folder |
| `preprocess` | Remove long silence from an audio file |
| `transcribe` | Transcribe one audio file |
| `batch` | Transcribe multiple files into a run directory |
| `quality` | Flag common hallucination and repetition markers |
| `review` | Find uncertain timestamp windows and optionally extract clips |
| `combine` | Combine transcript text files into Markdown |
| `note` | Write a vault-ready Markdown transcript note |

Examples:

```sh
transcribe-audio models --json
transcribe-audio discover ~/Downloads --since 2d --json
transcribe-audio transcribe "/path/to/audio.m4a" --formats txt,json --json
transcribe-audio batch "/path/one.m4a" "/path/two.m4a" --output-dir ./transcripts/current --json
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
locally through MLX Whisper or whisper.cpp.

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
