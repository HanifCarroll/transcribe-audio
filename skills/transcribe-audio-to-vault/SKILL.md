---
name: transcribe-audio-to-vault
description: Audio transcription workflow for turning recordings and voice notes into reviewed, sourced Markdown or Obsidian notes. Use when the user asks to transcribe audio files, process voice memos or meeting recordings, create transcript notes in a vault folder, review transcript text for likely transcription issues, compare MLX Whisper with whisper.cpp, preprocess silence, or preserve transcript provenance.
---

# Transcribe Audio To Vault

Use `transcribe-audio` as a workbench. Let the tool handle repeatable audio steps; use agent judgment for prompts, retries, review, note titles, and final cleanup.

## Workflow

1. Locate the inputs and destination. Expand paths, create the requested note folder when needed, and finish this step when every audio input and target folder path is known.

2. Check the workbench:

```sh
transcribe-audio doctor --json
transcribe-audio models --json
```

If running from a repo checkout instead of an installed command, use:

```sh
uv run transcribe-audio doctor --json
uv run transcribe-audio models --json
```

If the default MLX model is missing, install it:

```sh
transcribe-audio download-model large-v3 --json
```

3. Prepare difficult audio only when evidence points there: long silence, first-pass quality warnings, or user context that the recording has dead air.

```sh
transcribe-audio preprocess "$AUDIO" --json
```

Use the preprocessed output for the next transcription and record that fact in the final note.

4. Transcribe with MLX as the default:

```sh
transcribe-audio transcribe "$AUDIO" \
  --output-dir "$RUN_DIR" \
  --formats txt,json \
  --json
```

Add `--prompt` or `--prompt-file` when names, products, places, or technical vocabulary are known. Use `--backend whisper-cpp` for explicit comparison, unavailable MLX, unresolved quality problems, or subtitle output. MLX writes `txt,json`; request `srt` or `vtt` through whisper.cpp.

5. Inspect quality:

```sh
transcribe-audio quality "$TXT" --json
```

Treat warnings as review prompts. Retry with preprocessing or backend comparison when that is likely to improve the transcript; otherwise report the warning with the note.

6. Review unclear text when needed. Use the same audio file that was passed to `transcribe`; if a desilenced file was transcribed, pass that desilenced file as `--audio`.

```sh
transcribe-audio review "$JSON" \
  --audio "$TRANSCRIBED_AUDIO" \
  --phrase "suspect phrase" \
  --extract-clips \
  --output-dir "$RUN_DIR/review" \
  --json
```

Use extracted clips and emitted re-transcription command arguments to compare MLX and whisper.cpp outputs. Apply a correction only when transcript context and audio review make it strong enough; otherwise mark the item as `Needs audio check`.

7. Choose the note title from the transcript content, then write the note:

```sh
transcribe-audio note "$TXT" \
  --output "$DEST/Descriptive Title.md" \
  --title "Descriptive Title" \
  --date "$DATE" \
  --source "$AUDIO" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --raw-json "$JSON" \
  --json
```

Include `--preprocessing-note` when the transcript came from a desilenced or otherwise processed audio file.

8. Review the note before finishing. Break transcript text into readable paragraphs by topic or natural pause. Do not change the raw `.txt` or `.json` artifacts. Apply only high-confidence cleanup in the note copy: duplicated words or sentences, clear punctuation fixes, product-name casing, and obvious misrecognitions grounded in local context.

## Completion

Finish only when the requested folder exists, every input has a descriptively named Markdown note, each note has `date` frontmatter, transcript provenance is recorded, raw `txt,json` artifacts are retained, the note transcript is readable, and remaining quality warnings or `Needs audio check` items are either resolved or explicitly reported.
