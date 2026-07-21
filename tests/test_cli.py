import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from transcribe_audio import cli


class CliHelpersTest(unittest.TestCase):
    def test_slugify_keeps_dotted_times(self) -> None:
        self.assertEqual(
            cli.slugify("WhatsApp Audio 2026-04-20 at 07.50.30.opus"),
            "WhatsApp-Audio-2026-04-20-at-07.50.30",
        )

    def test_parse_formats_defaults_and_deduplicates(self) -> None:
        self.assertEqual(cli.parse_formats(" txt, json,txt "), ["txt", "json"])
        self.assertEqual(cli.parse_formats(""), ["txt", "json"])

    def test_normalize_argv_adds_transcribe_for_paths(self) -> None:
        self.assertEqual(
            cli.normalize_argv(["/tmp/audio.m4a", "--json"]),
            ["transcribe", "/tmp/audio.m4a", "--json"],
        )
        self.assertEqual(cli.normalize_argv(["doctor", "--json"]), ["doctor", "--json"])
        self.assertEqual(
            cli.normalize_argv(["review", "/tmp/audio.json"]),
            ["review", "/tmp/audio.json"],
        )

    def test_batch_accepts_jobs(self) -> None:
        args = cli.build_parser().parse_args(
            ["batch", "one.m4a", "two.m4a", "--jobs", "2"]
        )

        self.assertEqual(args.jobs, 2)

    def test_spokenly_backend_aliases_and_model(self) -> None:
        self.assertEqual(cli.normalize_backend("spokenly"), "spokenly")
        self.assertEqual(cli.normalize_backend("parakeet-tdt"), "spokenly")

        selected = cli.choose_spokenly_model("auto", cli.env_paths())

        self.assertEqual(selected.name, "parakeet-tdt-0.6b-v3")
        self.assertEqual(selected.locator, "parakeetTDT06")

    def test_spokenly_model_preference_requires_json_string(self) -> None:
        self.assertEqual(
            cli.decode_spokenly_model_preference(b'"parakeetTDT06"'),
            "parakeetTDT06",
        )
        self.assertIsNone(cli.decode_spokenly_model_preference(b"parakeetTDT06"))
        self.assertIsNone(
            cli.decode_spokenly_model_preference({"model": "parakeetTDT06"})
        )

    def test_spokenly_transcript_requires_exact_timestamped_segments(self) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "modelId": "parakeetTDT06",
                                "segments": [
                                    {"start": 0.0, "end": 0.5, "text": "Hello"},
                                    {"start": 0.5, "end": 1.0, "text": "world."},
                                ],
                            }
                        ),
                    }
                ]
            },
        }

        transcript = cli.parse_spokenly_transcription(response)

        self.assertEqual(transcript["modelId"], "parakeetTDT06")
        self.assertEqual(cli.spokenly_transcript_text(transcript), "Hello world.")

    def test_spokenly_rejects_prompts_and_subtitle_formats(self) -> None:
        with self.assertRaises(cli.ToolError):
            cli.validate_spokenly_options("en", "Names: Codex")
        with self.assertRaises(cli.ToolError):
            cli.validate_formats(["txt", "srt"], "spokenly")

    def test_spokenly_model_download_routes_to_app(self) -> None:
        args = cli.build_parser().parse_args(
            ["download-model", "large-v3", "--backend", "spokenly"]
        )

        with self.assertRaisesRegex(cli.ToolError, "Spokenly's model manager"):
            cli.cmd_download_model(args)

    def test_review_commands_use_parakeet_tdt_as_primary(self) -> None:
        commands = cli.review_transcribe_commands(
            pathlib.Path("/tmp/clip.wav"),
            pathlib.Path("/tmp/review"),
            "clip",
        )

        self.assertIn("parakeet_tdt", commands)
        self.assertEqual(
            commands["parakeet_tdt"][commands["parakeet_tdt"].index("--model") + 1],
            "parakeetTDT06",
        )

    def test_unique_output_name_reserves_duplicate_batch_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            reserved: set[str] = set()

            first = cli.unique_output_name(
                pathlib.Path("same.m4a"), 1, output_dir, reserved
            )
            reserved.add(first)
            second = cli.unique_output_name(
                pathlib.Path("same.m4a"), 2, output_dir, reserved
            )

        self.assertEqual(first, "same")
        self.assertEqual(second, "002-same")

    def test_batch_parallel_preserves_input_order_in_manifest(self) -> None:
        original_transcribe_one = cli.transcribe_one
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            output_dir = root / "out"
            inputs = [root / "slow.m4a", root / "fast.m4a"]
            for input_path in inputs:
                input_path.write_text("synthetic", encoding="utf-8")

            def fake_transcribe_one(args, input_path):
                output_base = pathlib.Path(args.output_dir) / args.output_name
                txt = cli.output_path_for_format(output_base, "txt")
                txt.write_text(f"{pathlib.Path(input_path).stem}\n", encoding="utf-8")
                return {
                    "ok": True,
                    "input": str(pathlib.Path(input_path).resolve()),
                    "outputs": {"txt": str(txt)},
                    "duration_seconds": 0.001,
                }

            stdout = io.StringIO()
            cli.transcribe_one = fake_transcribe_one
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "batch",
                            str(inputs[0]),
                            str(inputs[1]),
                            "--output-dir",
                            str(output_dir),
                            "--jobs",
                            "2",
                            "--json",
                        ]
                    )
            finally:
                cli.transcribe_one = original_transcribe_one

            payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["jobs"], 2)
        self.assertEqual(
            [pathlib.Path(item["input"]).name for item in payload["items"]],
            ["slow.m4a", "fast.m4a"],
        )
        self.assertEqual(payload["success_count"], 2)
        self.assertTrue(payload["markdown"].endswith("transcripts.md"))

    def test_batch_parallel_rejects_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.build_parser().parse_args(
                [
                    "batch",
                    "one.m4a",
                    "two.m4a",
                    "--output-dir",
                    tmp,
                    "--jobs",
                    "2",
                    "--fail-fast",
                ]
            )

            with self.assertRaises(cli.ToolError):
                cli.cmd_batch(args)

    def test_batch_parallel_rejects_spokenly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.build_parser().parse_args(
                [
                    "batch",
                    "one.m4a",
                    "two.m4a",
                    "--backend",
                    "spokenly",
                    "--output-dir",
                    tmp,
                    "--jobs",
                    "2",
                ]
            )

            with self.assertRaises(cli.ToolError):
                cli.cmd_batch(args)

    def test_quality_flags_common_silence_hallucinations(self) -> None:
        warnings = cli.transcript_quality_warnings("[BLANK_AUDIO]\nThank you\n")
        self.assertEqual(
            {warning["code"] for warning in warnings},
            {"blank-audio-marker", "thanks-marker"},
        )

    def test_review_windows_include_phrase_and_low_confidence_words(self) -> None:
        data = {
            "segments": [
                {
                    "start": 10.0,
                    "end": 12.0,
                    "text": "Probably the trackpad.",
                    "words": [
                        {
                            "word": "Probably",
                            "start": 10.1,
                            "end": 10.4,
                            "probability": 0.7,
                        },
                        {
                            "word": "trackpad",
                            "start": 11.0,
                            "end": 11.4,
                            "probability": 0.1,
                        },
                    ],
                }
            ]
        }

        windows = cli.build_review_windows(
            data,
            phrases=["trackpad"],
            min_word_probability=0.35,
            context_seconds=1.0,
            merge_gap_seconds=0.5,
            max_windows=5,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start"], 9.0)
        self.assertEqual(windows[0]["end"], 13.0)
        self.assertEqual(
            {reason["code"] for reason in windows[0]["reasons"]},
            {"phrase", "low-word-probability"},
        )

    def test_transcript_segments_supports_whisper_cpp_json(self) -> None:
        data = {
            "transcription": [
                {
                    "offsets": {"from": 1500, "to": 2750},
                    "text": " short clip ",
                }
            ]
        }

        self.assertEqual(
            cli.transcript_segments(data),
            [{"start": 1.5, "end": 2.75, "text": "short clip", "words": []}],
        )

    def test_review_windows_prioritize_explicit_phrases_over_low_confidence_cap(
        self,
    ) -> None:
        data = {
            "segments": [
                {
                    "start": float(index * 10),
                    "end": float(index * 10 + 1),
                    "text": f"low confidence segment {index}",
                    "words": [
                        {
                            "word": f"low{index}",
                            "start": float(index * 10),
                            "end": float(index * 10 + 0.5),
                            "probability": 0.1,
                        }
                    ],
                }
                for index in range(3)
            ]
            + [
                {
                    "start": 100.0,
                    "end": 101.0,
                    "text": "Probably the trackpad.",
                    "words": [],
                }
            ]
        }

        windows = cli.build_review_windows(
            data,
            phrases=["trackpad"],
            min_word_probability=0.35,
            context_seconds=0.0,
            merge_gap_seconds=0.0,
            max_windows=1,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["text"], "Probably the trackpad.")

    def test_supported_audio_extensions_are_case_insensitive(self) -> None:
        self.assertTrue(cli.is_supported_audio_path(pathlib.Path("voice.OPUS")))
        self.assertTrue(cli.is_supported_audio_path(pathlib.Path("clip.webm")))
        self.assertFalse(cli.is_supported_audio_path(pathlib.Path("notes.txt")))

    def test_yaml_scalar_quotes_and_escapes(self) -> None:
        self.assertEqual(cli.yaml_scalar('a "quote"'), '"a \\"quote\\""')


if __name__ == "__main__":
    unittest.main()
