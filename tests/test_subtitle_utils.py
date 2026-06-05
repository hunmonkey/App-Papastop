import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from papastop import (
    ASREngine,
    OperationCancelled,
    SubtitleSegment,
    TranslationEngine,
    build_eta_text,
    build_resource_settings,
    build_subtitle_overwrite_prompt,
    clean_translation_for_subtitle,
    clean_text_for_srt,
    default_srt_path,
    format_bytes_as_gib,
    format_duration_short,
    format_srt_time,
    format_vga_spec,
    is_http_media_source,
    merge_vga_specs,
    nllb_lang,
    parse_nvidia_smi_csv,
    save_float_audio_to_wav,
    source_stem,
    source_stem_from_url,
    write_pcm16_to_wav,
    write_srt,
)


class FakeASREngine(ASREngine):
    def _load_model(self, on_line=None):
        self._model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [
                    SimpleNamespace(start=0.0, end=1.0, text=" first line "),
                    SimpleNamespace(start=1.0, end=2.0, text="second line"),
                ],
                SimpleNamespace(language="en"),
            )
        )


class FakeLlama:
    def __init__(self, content: str):
        self.content = content
        self.messages = []

    def create_chat_completion(self, *, messages, **_kwargs):
        self.messages.append(messages)
        return {"choices": [{"message": {"content": self.content}}]}


class SubtitleUtilityTests(unittest.TestCase):
    def test_format_srt_time(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")
        self.assertEqual(format_srt_time(3661.25), "01:01:01,250")

    def test_clean_text_for_srt(self):
        self.assertEqual(clean_text_for_srt("  Hello     world  "), "Hello world")

    def test_clean_translation_for_subtitle_removes_source_lines(self):
        text = "日本語: 今日はいい天気ですね\n한국어: 오늘은 날씨가 좋네요."

        self.assertEqual(
            clean_translation_for_subtitle(text, "今日はいい天気ですね"),
            "오늘은 날씨가 좋네요.",
        )

        self.assertEqual(
            clean_translation_for_subtitle(
                "Japanese: 今日はいい天気ですね Korean: 오늘은 날씨가 좋네요.",
                "今日はいい天気ですね",
            ),
            "오늘은 날씨가 좋네요.",
        )

    def test_clean_translation_for_subtitle_rejects_source_only_japanese(self):
        self.assertEqual(clean_translation_for_subtitle("今日はいい天気ですね", "今日はいい天気ですね"), "")

    def test_llm_numbered_translation_parser_keeps_korean_only(self):
        engine = TranslationEngine()

        parsed = engine._parse_llm_numbered_translations(
            "1. Japanese: 今日はいい天気ですね\n한국어: 오늘은 날씨가 좋네요.\n2. 번역: 같이 가요.",
            ["今日はいい天気ですね", "行きましょう"],
        )

        self.assertEqual(parsed, ["오늘은 날씨가 좋네요.", "같이 가요."])

    def test_llm_translation_batches_lines_for_context(self):
        engine = TranslationEngine()
        fake = FakeLlama("1. 오늘은 날씨가 좋네요.\n2. 같이 가요.")
        engine._load_llama = lambda _key: fake

        translated = engine._translate_llama(["今日はいい天気ですね", "行きましょう"], "ja", "qwen2.5-14b")

        self.assertEqual(translated, ["오늘은 날씨가 좋네요.", "같이 가요."])
        user_prompt = fake.messages[0][1]["content"]
        self.assertIn("1. 今日はいい天気ですね", user_prompt)
        self.assertIn("2. 行きましょう", user_prompt)
        self.assertIn("Return exactly 2 lines", user_prompt)

    def test_write_srt_filters_empty_lines_and_indexes(self):
        output = Path(tempfile.gettempdir()) / "papastop_test.srt"
        segments = [
            SubtitleSegment(0.0, 1.0, "  첫 번째  줄  "),
            SubtitleSegment(1.5, 2.0, "   "),
        ]
        write_srt(segments, output)
        try:
            text = output.read_text(encoding="utf-8")
            self.assertIn("1\n00:00:00,000 --> 00:00:01,000", text)
            self.assertNotIn("\n2\n", text)
            self.assertIn("첫 번째 줄", text)
        finally:
            if output.exists():
                output.unlink()

    def test_default_srt_path_url_and_file(self):
        self.assertEqual(default_srt_path("https://example.com/sample.mp4").name, "sample.srt")
        self.assertEqual(default_srt_path("https://example.com/stream", Path("out")).name, "stream.srt")
        self.assertEqual(default_srt_path("movie_clip.mp4", Path("tmp_out")).name, "movie_clip.srt")

    def test_subtitle_overwrite_prompt_names_existing_subtitles(self):
        prompt = build_subtitle_overwrite_prompt([Path("movie.srt"), Path("clip.srt")])

        self.assertIn("같은 이름의 자막 파일", prompt)
        self.assertIn("movie.srt", prompt)
        self.assertIn("clip.srt", prompt)
        self.assertIn("덮어쓸까요?", prompt)

    def test_is_http_media_source(self):
        self.assertTrue(is_http_media_source("https://example.com/video"))
        self.assertTrue(is_http_media_source("rtsp://camera.local/stream"))
        self.assertFalse(is_http_media_source("C:\\\\music\\\\song.mp3"))

    def test_source_stem_and_lang_map(self):
        self.assertEqual(source_stem("/media/song.mp3"), "song")
        self.assertEqual(source_stem("https://example.com/media/file.webm"), "file")
        self.assertEqual(source_stem_from_url("https://example.com/media/file.webm?x=1"), "file")
        self.assertEqual(source_stem_from_url("rtsp://host/stream"), "stream")
        self.assertEqual(nllb_lang("EN-US"), "eng_Latn")
        self.assertEqual(nllb_lang(None), "eng_Latn")

    def test_write_pcm16_to_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pcm.wav"
            write_pcm16_to_wav(b"\x00\x00\xff\xff", 16_000, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_save_float_audio_to_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "float.wav"
            audio = np.array([[0.1, -0.2], [0.4, -0.4]], dtype=np.float32)
            save_float_audio_to_wav(audio, 16_000, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_operation_cancelled(self):
        with self.assertRaises(OperationCancelled):
            raise OperationCancelled("cancelled")

    def test_system_spec_formatters(self):
        self.assertEqual(format_bytes_as_gib(8 * 1024**3), "8.0 GB")
        self.assertEqual(format_bytes_as_gib(None), "알 수 없음")
        self.assertEqual(
            format_vga_spec("NVIDIA RTX", 12 * 1024**3, "1.2.3"),
            "NVIDIA RTX (VRAM 12.0 GB, 드라이버 1.2.3)",
        )

    def test_nvidia_smi_parser_and_vga_merge(self):
        parsed = parse_nvidia_smi_csv("NVIDIA RTX, 16303, 610.47\n")

        self.assertEqual(parsed, [("NVIDIA RTX", 16303 * 1024 * 1024, "610.47")])
        self.assertEqual(
            merge_vga_specs(
                ["NVIDIA RTX (VRAM 15.9 GB, 드라이버 610.47)"],
                ["NVIDIA RTX (VRAM 15.6 GB)", "Intel UHD (VRAM 1.0 GB)"],
            ),
            ["NVIDIA RTX (VRAM 15.9 GB, 드라이버 610.47)", "Intel UHD (VRAM 1.0 GB)"],
        )

    def test_eta_text_uses_progress_speed(self):
        self.assertEqual(format_duration_short(65), "1분 5초")
        self.assertEqual(build_eta_text(0, 1000, 1060), "예상 시간: 계산 중")
        self.assertIn("약 1분 0초 남음", build_eta_text(50, 1000, 1060))
        self.assertEqual(build_eta_text(100, 1000, 1060), "예상 시간: 완료")

    def test_max_resource_settings_use_gpu_and_all_cpu_threads(self):
        with (
            mock.patch("papastop.os.cpu_count", return_value=8),
            mock.patch("papastop.get_primary_cuda_vram_bytes", return_value=16 * 1024**3),
            mock.patch("papastop.get_ctranslate2_cuda_available", return_value=True),
            mock.patch("papastop.get_ctranslate2_compute_types", return_value={"float16", "int8_float16"}),
            mock.patch("papastop.get_torch_cuda_available", return_value=False),
            mock.patch("papastop.has_module", return_value=True),
        ):
            settings = build_resource_settings("max")

        self.assertEqual(settings.cpu_threads, 8)
        self.assertEqual(settings.asr_device, "cuda")
        self.assertEqual(settings.asr_compute_type, "float16")
        self.assertEqual(settings.translation_device, -1)

    def test_asr_transcribe_emits_segments_for_preview(self):
        engine = FakeASREngine()
        preview_segments = []

        segments, language = engine.transcribe(
            Path("fake.wav"),
            on_segment=preview_segments.append,
        )

        self.assertEqual(language, "en")
        self.assertEqual([segment.text for segment in segments], ["first line", "second line"])
        self.assertEqual([segment.text for segment in preview_segments], ["first line", "second line"])

    def test_asr_model_selection_changes_model_label_without_download(self):
        engine = ASREngine()

        engine.set_model("small")

        self.assertEqual(engine.model_name, "small")
        self.assertEqual(engine.model_label(), "균형 (small)")


if __name__ == "__main__":
    unittest.main()
