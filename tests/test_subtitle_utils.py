import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from papastop import (
    ASREngine,
    OperationCancelled,
    SubtitleSegment,
    build_subtitle_overwrite_prompt,
    clean_text_for_srt,
    default_srt_path,
    format_srt_time,
    is_http_media_source,
    nllb_lang,
    save_float_audio_to_wav,
    source_stem,
    source_stem_from_url,
    write_pcm16_to_wav,
    write_srt,
)


class FakeASREngine(ASREngine):
    def _load_model(self):
        self._model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [
                    SimpleNamespace(start=0.0, end=1.0, text=" first line "),
                    SimpleNamespace(start=1.0, end=2.0, text="second line"),
                ],
                SimpleNamespace(language="en"),
            )
        )


class SubtitleUtilityTests(unittest.TestCase):
    def test_format_srt_time(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")
        self.assertEqual(format_srt_time(3661.25), "01:01:01,250")

    def test_clean_text_for_srt(self):
        self.assertEqual(clean_text_for_srt("  Hello     world  "), "Hello world")

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
