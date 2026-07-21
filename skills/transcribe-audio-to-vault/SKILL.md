---
name: transcribe-audio-to-vault
description: Canonical audio transcription workflow for turning recordings, videos, and voice notes into reviewed, sourced Obsidian or vault notes, with optional automatic walky-talky import and follow-up thread handoff. Use for local audio transcription, video transcript notes, voice memo import, no-argument walky-talky import, transcript note creation, quality review, Spokenly Parakeet TDT transcription, explicit Whisper fallback comparison, silence preprocessing, provenance, launching the walky-talky follow-up workflow after reviewed notes are created, or launching the video transcript workflow after reviewed video transcripts are created.
---

# Transcribe Audio To Vault

Use the local tool as a workbench, then apply agent judgment to naming, retries, and final note shape:

```sh
TOOL_DIR="/Users/hanifcarroll/projects/tools/transcribe-audio"
LEDGER_TOOL="/Users/hanifcarroll/.codex/skills/transcribe-audio-to-vault/scripts/walky_talky_ledger.py"
```

## Workflow

1. Locate the inputs and destination.

   - If the user gives audio paths, expand those paths and use the requested destination.
   - If the skill is invoked without explicit audio paths, run automatic walky-talky import from the current vault:

```sh
python3 "$LEDGER_TOOL" scan --vault "$PWD" --json
```

   Automatic walky-talky import scans Apple Voice Memos and `sources/walky-talky/inbox` only. It excludes `sources/feedback/casamo`, broad Downloads folders, and `sources/walky-talky/raw`. New means: source content hash is absent from `.transcribe-audio/imports.jsonl` and the source recording time is newer than the latest successful walky-talky ledger record. Process returned files oldest-first. If no files are returned, stop after reporting that there are no new walky-talky voice notes.

   Finish this step when every audio input and target folder path is known. For automatic walky-talky import, the target folder is `sources/walky-talky` and raw `txt,json` transcript artifacts go in `sources/walky-talky/raw`. Apple Voice Memos remains the audio authority; do not copy source audio into the vault.

2. Check the bench:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio doctor --backend spokenly --json)
(cd "$TOOL_DIR" && uv run transcribe-audio models --backend spokenly --json)
```

Continue only when the doctor reports `ok: true` and the model list reports `spokenly:parakeetTDT06` as installed. If either check fails, open Spokenly, install and select NVIDIA Parakeet TDT 0.6B V3 for file transcription, enable Spokenly's local MCP bridge, and rerun both checks.

3. Prepare difficult audio only when evidence points there: long silence, first-pass quality warnings, or user context that the recording has dead air. Run:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio preprocess "$AUDIO" --json)
```

Use the preprocessed output for the next transcription and record that fact in the final note.

4. Transcribe with Spokenly Parakeet TDT as the default:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio transcribe "$AUDIO" \
  --backend spokenly \
  --model parakeetTDT06 \
  --output-dir "$RUN_DIR" \
  --formats txt,json \
  --json)
```

Parakeet TDT accepts English audio and writes `txt,json`; it does not accept `--prompt` or `--prompt-file`. Use MLX `large-v3` only when Parakeet is unavailable or an unresolved quality issue needs a comparison. Use whisper.cpp only when `srt` or `vtt` output is required.

For multiple independent files, batch raw transcription sequentially because Spokenly accepts one file job at a time:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio batch "$AUDIO_ONE" "$AUDIO_TWO" \
  --backend spokenly \
  --model parakeetTDT06 \
  --output-dir "$RUN_DIR" \
  --formats txt,json \
  --jobs 1 \
  --json)
```

Treat the batch manifest as the transcript artifact list. For automatic walky-talky imports, preserve the scanned oldest-first order for quality checks, review, vault note writing, ledger appends, and follow-up handoff.

5. Inspect quality:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio quality "$TXT" --json)
```

Treat warnings as review prompts. Retry with preprocessing or backend comparison when that is likely to improve the transcript; otherwise report the warning with the note.

6. Choose the note title from the transcript content, then write the vault note:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio note "$TXT" \
  --output "$DEST/Descriptive Title.md" \
  --title "Descriptive Title" \
  --date "$DATE" \
  --source "$AUDIO" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --raw-json "$JSON" \
  --json)
```

Include `--preprocessing-note` when the transcript came from a desilenced or otherwise processed audio file.

7. Review the vault note transcript text before finishing:

   - Break the transcript into readable paragraphs by topic or natural pause. Do not change the raw `.txt` or `.json` artifacts.
   - Apply obvious high-confidence cleanup only in the vault note copy: duplicated words or sentences, product-name casing, clear punctuation fixes, and clear misrecognitions where the intended phrase is obvious from local context.
   - For unclear names, tool names, technical terms, or garbled phrases, use the review helper against the raw JSON and the same audio file that was passed to `transcribe`:

```sh
(cd "$TOOL_DIR" && uv run transcribe-audio review "$JSON" \
  --audio "$TRANSCRIBED_AUDIO" \
  --phrase "suspect phrase" \
  --extract-clips \
  --output-dir "$RUN_DIR/review" \
  --json)
```

   The review command emits timestamped windows, temporary extracted clips, and command arguments for Parakeet TDT plus MLX `large-v3` fallback re-transcription. Run those commands when the review evidence is needed. If a desilenced file was transcribed, pass that desilenced file as `--audio`; JSON timestamps match the transcribed input, not necessarily the original source.
   - Do not guess unclear phrases. Apply a correction only when transcript context, low-confidence evidence, and clip re-transcription make the correction strong enough; otherwise flag it as `Needs audio check`.
   - Report any high-confidence edits made and any remaining `Needs audio check` items.

8. For automatic walky-talky import, append the ledger record only after the note exists, the transcript artifacts are retained, and review is complete:

```sh
python3 "$LEDGER_TOOL" append \
  --vault "$PWD" \
  --source "$AUDIO" \
  --note "$NOTE" \
  --txt "$TXT" \
  --json-artifact "$JSON" \
  --run-dir "$RUN_DIR" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --json
```

   The ledger is the source of truth for repeat imports. Do not mark a file imported before the final note has been written and reviewed.

9. Delete derived walky-talky audio after the successful ledger append:

```sh
python3 "$LEDGER_TOOL" cleanup-audio \
  --vault "$PWD" \
  --run-dir "$RUN_DIR" \
  --source "$AUDIO" \
  --json
```

   Cleanup is mandatory even when the reviewed note still contains `Needs audio check`. Preserve the original Voice Memo, raw `txt,json` transcripts, reviewed note, and ledger record; do not retain extracted review clips, desilenced copies, or other derived audio in the vault.

10. For each reviewed walky-talky source note, start a new Codex thread to run the follow-up workflow. Use the Codex Desktop `create_thread` tool path only: search for the `create_thread` thread tool if it is not already callable, then create a user-owned thread that appears in the Codex Desktop sidebar. Do not use `multi_agent_v1.spawn_agent` or other sub-agent tools for this step; sub-agents are not Codex threads. Do not substitute low-level or headless thread mechanisms such as `thread/start`, `thread/name/set`, `turn/start`, manual app-server WebSocket calls, `codex debug app-server send-message-v2`, or a manually started `codex app-server`. Title the thread `Walky-Talky Follow-Up: <note title>` and send this prompt:

```text
Run the walky-talky follow-up workflow for this reviewed source note:

Vault: /Users/hanifcarroll/Library/Mobile Documents/iCloud~md~obsidian/Documents/hanif-md
Workflow: workflows/sources/Walky-Talky Follow-Up Workflow.md
Source note: <absolute source note path>

Read AGENTS.md and the workflow first. Then run the walky-talky wrapper: run Vault Context Preflight, Atomic Note Extraction, and Source Follow-Up Research when the source contains follow-up work. Capture source-faithful questions and investigation prompts, extract deduped task candidates, and offer Things and Codex-thread handoffs for selected tasks. Do not mutate external accounts or perform outreach without explicit approval.
```

   If `create_thread` is unavailable, stop the thread-handoff step and report the exact prompt above so the follow-up can be launched manually. Do not run atomic extraction, source follow-up research, Things handoff, or Codex-startable task triage inside this transcription workflow unless Hanif explicitly asks for that fallback.

11. For each reviewed video transcript note in `sources/video/transcripts/`, start a new Codex thread to run the video transcript atomic extraction workflow. Use the Codex Desktop `create_thread` tool path only: search for the `create_thread` thread tool if it is not already callable, then create a user-owned thread that appears in the Codex Desktop sidebar. Do not use `multi_agent_v1.spawn_agent` or other sub-agent tools for this step; sub-agents are not Codex threads. Do not substitute low-level or headless thread mechanisms such as `thread/start`, `thread/name/set`, `turn/start`, manual app-server WebSocket calls, `codex debug app-server send-message-v2`, or a manually started `codex app-server`. Title the thread `Video Transcript Atomic Extraction: <note title>` and send this prompt:

```text
Run the video transcript atomic extraction workflow for this reviewed source note:

Vault: /Users/hanifcarroll/Library/Mobile Documents/iCloud~md~obsidian/Documents/hanif-md
Workflow: workflows/sources/Video Transcript Atomic Extraction Workflow.md
Source note: <absolute source note path>

Read AGENTS.md and the workflow first. Then run the video transcript wrapper: run Vault Context Preflight and Atomic Note Extraction for this reviewed transcript. If the transcript contains substantive questions, project implications, strategy implications, tool/workflow ideas, public-facing content implications, or execution follow-ups, route to Source Follow-Up Research. Do not select clips, edit video, create short-form packaging, publish, or mutate external accounts without explicit approval.
```

   If `create_thread` is unavailable, stop the thread-handoff step and report the exact prompt above so the workflow can be launched manually. Do not run the video transcript workflow inside this transcription workflow unless Hanif explicitly asks for that fallback.

## Completion

Finish only when the requested folder exists, every input has a descriptively named Markdown note, each note has `date` frontmatter, transcript provenance is recorded, raw `txt,json` artifacts are retained, the note transcript is paragraphized and reviewed, automatic walky-talky imports have ledger records and zero derived audio left in their vault run directories, the walky-talky follow-up workflow has been launched or a manual launch prompt has been reported for each walky-talky note, the video transcript workflow has been launched or a manual launch prompt has been reported for each reviewed video transcript note, and remaining quality warnings or `Needs audio check` items are either resolved or explicitly reported.
