import pathlib
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
        self.assertEqual(cli.normalize_argv(["/tmp/audio.m4a", "--json"]), ["transcribe", "/tmp/audio.m4a", "--json"])
        self.assertEqual(cli.normalize_argv(["doctor", "--json"]), ["doctor", "--json"])
        self.assertEqual(cli.normalize_argv(["review", "/tmp/audio.json"]), ["review", "/tmp/audio.json"])

    def test_quality_flags_common_silence_hallucinations(self) -> None:
        warnings = cli.transcript_quality_warnings("[BLANK_AUDIO]\nThank you\n")
        self.assertEqual({warning["code"] for warning in warnings}, {"blank-audio-marker", "thanks-marker"})

    def test_review_windows_include_phrase_and_low_confidence_words(self) -> None:
        data = {
            "segments": [
                {
                    "start": 10.0,
                    "end": 12.0,
                    "text": "Probably the trackpad.",
                    "words": [
                        {"word": "Probably", "start": 10.1, "end": 10.4, "probability": 0.7},
                        {"word": "trackpad", "start": 11.0, "end": 11.4, "probability": 0.1},
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

    def test_review_windows_prioritize_explicit_phrases_over_low_confidence_cap(self) -> None:
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
