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

    def test_quality_flags_common_silence_hallucinations(self) -> None:
        warnings = cli.transcript_quality_warnings("[BLANK_AUDIO]\nThank you\n")
        self.assertEqual({warning["code"] for warning in warnings}, {"blank-audio-marker", "thanks-marker"})

    def test_supported_audio_extensions_are_case_insensitive(self) -> None:
        self.assertTrue(cli.is_supported_audio_path(pathlib.Path("voice.OPUS")))
        self.assertTrue(cli.is_supported_audio_path(pathlib.Path("clip.webm")))
        self.assertFalse(cli.is_supported_audio_path(pathlib.Path("notes.txt")))

    def test_yaml_scalar_quotes_and_escapes(self) -> None:
        self.assertEqual(cli.yaml_scalar('a "quote"'), '"a \\"quote\\""')


if __name__ == "__main__":
    unittest.main()
