import tempfile
import unittest
from pathlib import Path

from papastop import (
    SubtitleBlock,
    convert_srt_text,
    convert_subtitle_file,
    ensure_subtitle_extension,
    format_timestamp,
    parse_srt_blocks,
    recommend_transcribe_settings,
    render_srt,
    render_srt_blocks,
)


class SubtitleUtilityTests(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(3661.25), "01:01:01,250")

    def test_render_srt(self):
        rendered = render_srt([(0, 1.5, "  Hello   PapaStop  ")])
        self.assertEqual(rendered, "1\n00:00:00,000 --> 00:00:01,500\nHello PapaStop\n")

    def test_parse_and_render_blocks(self):
        source = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        blocks = parse_srt_blocks(source)
        self.assertEqual(blocks, [SubtitleBlock("1", "00:00:00,000 --> 00:00:01,000", ["Hello"])])
        self.assertEqual(render_srt_blocks(blocks), "1\n00:00:00,000 --> 00:00:01,000\nHello\n")

    def test_convert_srt_to_vtt(self):
        source = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        self.assertEqual(convert_srt_text(source, "vtt"), "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n")

    def test_convert_srt_to_text(self):
        source = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello\nworld\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nPapaStop\n"
        )
        self.assertEqual(convert_srt_text(source, "txt"), "Hello world\n\nPapaStop\n")

    def test_convert_subtitle_file_adds_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "sample.srt"
            target_path = temp_path / "sample-converted"
            source_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

            saved_path = convert_subtitle_file(
                source_path,
                target_path,
                "txt",
                on_line=lambda _line: None,
                on_progress=lambda _value: None,
            )

            self.assertEqual(saved_path, temp_path / "sample-converted.txt")
            self.assertEqual(saved_path.read_text(encoding="utf-8"), "Hello\n")

    def test_convert_subtitle_file_writes_vtt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "sample.srt"
            target_path = temp_path / "sample-web"
            source_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

            saved_path = convert_subtitle_file(
                source_path,
                target_path,
                "vtt",
                on_line=lambda _line: None,
                on_progress=lambda _value: None,
            )

            self.assertEqual(saved_path, temp_path / "sample-web.vtt")
            self.assertEqual(
                saved_path.read_text(encoding="utf-8"),
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n",
            )

    def test_ensure_subtitle_extension(self):
        self.assertEqual(ensure_subtitle_extension(Path("sample"), "vtt"), Path("sample.vtt"))
        self.assertEqual(ensure_subtitle_extension(Path("sample.srt"), "srt"), Path("sample.srt"))

    def test_recommend_gpu_settings(self):
        settings, _reason = recommend_transcribe_settings(["NVIDIA Test GPU"])
        self.assertEqual(settings["device"], "cuda")
        self.assertEqual(settings["compute"], "float16")

    def test_recommend_cpu_settings(self):
        settings, _reason = recommend_transcribe_settings([])
        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["compute"], "int8")


if __name__ == "__main__":
    unittest.main()
