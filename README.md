# transcribe-audio

Local-first audio transcription CLI for agents and automation.

`transcribe-audio` wraps `ffmpeg` and `whisper-cli`, converts incoming audio to
16 kHz mono WAV, runs a local whisper.cpp model, and writes machine-readable
artifacts that are easy for coding agents to inspect.

It is built for workflows like:

- downloading WhatsApp voice notes
- finding recent audio files in Downloads
- transcribing one file or a batch
- producing `run.json` manifests for automation
- combining transcript text into a clean Markdown handoff

## Install

Prerequisites:

- Rust toolchain
- `ffmpeg`
- `whisper-cli` from whisper.cpp
- at least one local whisper.cpp model

Install from a checkout:

```sh
cargo install --path . --force
```

Verify:

```sh
transcribe-audio doctor --json
```

## Model Location

By default, models live in:

```text
~/.local/share/transcribe-audio/models
```

Resolution order:

1. `large-v3-turbo`
2. `medium`
3. `small`

The CLI also checks Superwhisper's local small model as a fallback:

```text
~/Library/Application Support/superwhisper/ggml-small.bin
```

Override model behavior with:

```sh
TRANSCRIBE_AUDIO_MODEL=/path/to/ggml-model.bin transcribe-audio audio.opus
TRANSCRIBE_AUDIO_MODEL_DIR=/path/to/models transcribe-audio doctor
```

## Quick Start

Transcribe one file:

```sh
transcribe-audio "/path/to/audio.opus" --print
```

Write text, JSON, and SRT outputs:

```sh
transcribe-audio "/path/to/audio.opus" \
  --output-dir /tmp/transcripts \
  --formats txt,json,srt \
  --json
```

Write a Markdown transcript too:

```sh
transcribe-audio "/path/to/audio.opus" --markdown --json
```

Use a prompt file for project names or vocabulary:

```sh
transcribe-audio "/path/to/audio.opus" \
  --prompt-file ./prompt.txt \
  --json
```

## Agent Workflow

Find likely audio files:

```sh
transcribe-audio discover ~/Downloads --since 2d --json
```

Transcribe a batch into one run directory:

```sh
transcribe-audio batch ~/Downloads/*.opus \
  --output-dir ./transcripts/current \
  --prompt-file ./prompt.txt \
  --formats txt,json,srt \
  --json
```

Batch output includes:

```text
transcripts/current/
  run.json
  transcripts.md
  <audio-name>.txt
  <audio-name>.json
  <audio-name>.srt
```

`run.json` records each input, output paths, durations, and per-file errors. The
top-level `ok` flag is `false` if any file failed.

Combine existing transcript files:

```sh
transcribe-audio combine ./transcripts/current \
  --output ./transcripts/current/transcripts.md \
  --title "Client Voice Notes" \
  --json
```

## Commands

```sh
transcribe-audio doctor --json
transcribe-audio models --json
transcribe-audio download-model large-v3-turbo
transcribe-audio discover ~/Downloads --since 2d --limit 10 --json
transcribe-audio "/path/to/audio.opus" --print
transcribe-audio transcribe "/path/to/audio.opus" --markdown --json
transcribe-audio batch "/path/one.opus" "/path/two.m4a" --output-dir ./transcripts/current --json
transcribe-audio combine ./transcripts/current --output ./transcripts/current/transcripts.md
```

## JSON Errors

JSON mode returns structured failures:

```json
{
  "ok": false,
  "stage": "input",
  "error": "input file does not exist: /tmp/missing.opus"
}
```

Subprocess failures include the command and exit code:

```json
{
  "ok": false,
  "stage": "whisper",
  "error": "command failed (1): ...",
  "command": "...",
  "exit_code": 1
}
```

Common stages:

- `cli`
- `doctor`
- `input`
- `prompt`
- `model`
- `convert`
- `whisper`
- `outputs`
- `discover`
- `combine`
- `download`

## Development

```sh
cargo fmt -- --check
cargo test
cargo clippy -- -D warnings
cargo build --release
```

End-to-end smoke:

```sh
rm -rf /tmp/transcribe-audio-smoke
mkdir -p /tmp/transcribe-audio-smoke
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i anullsrc=r=16000:cl=mono -t 0.4 \
  /tmp/transcribe-audio-smoke/input.wav

transcribe-audio batch /tmp/transcribe-audio-smoke/input.wav \
  --model small \
  --output-dir /tmp/transcribe-audio-smoke/run \
  --formats txt,json \
  --json
```
