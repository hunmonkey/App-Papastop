import queue
import builtins
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papastop import LiveCaptionService, SubtitleSegment, _UtteranceSegmenter


class FakeRecorder:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.entered = False
        self.exited = False
        self.record_calls = 0
        self.recorded_after_enter = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True

    def record(self, numframes=None):
        if not self.entered:
            raise AttributeError("'_Recorder' object has no attribute '_pending_chunk'")

        self.record_calls += 1
        self.recorded_after_enter = True
        if self.record_calls >= 2:
            self.stop_event.set()
        return np.zeros((numframes or 1, 1), dtype=np.float32)


class LiveCaptionServiceTests(unittest.TestCase):
    def test_pc_speaker_recorder_is_opened_before_recording(self):
        stop_event = threading.Event()
        recorder = FakeRecorder(stop_event)
        call_order = []
        microphone = types.SimpleNamespace(recorder=lambda samplerate: recorder)
        soundcard = types.SimpleNamespace(
            default_speaker=lambda: call_order.append("speaker") or types.SimpleNamespace(name="test speaker"),
            get_microphone=lambda id, include_loopback: microphone,
        )

        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=10, chunk_seconds=0.1)
        events = []

        previous_soundcard = sys.modules.get("soundcard")
        sys.modules["soundcard"] = soundcard
        real_import = builtins.__import__

        def recording_import(name, *args, **kwargs):
            if name == "soundcard":
                call_order.append("import")
            return real_import(name, *args, **kwargs)

        try:
            service._chunk_queue = queue.Queue(maxsize=8)
            # 실제 silero VAD 로딩을 피하려고 발화가 없다고 답하는 가짜 분할기를 주입한다.
            service._segmenter = _UtteranceSegmenter(
                service.sample_rate,
                on_utterance=lambda s, c: service._safe_put_chunk((s, c)),
                speech_timestamps_fn=lambda audio, sr: [],
            )

            def initialize_audio_thread():
                call_order.append("init")
                return lambda: call_order.append("cleanup")

            with (
                mock.patch("papastop.os.name", "nt"),
                mock.patch("papastop.has_module", return_value=True),
                mock.patch("builtins.__import__", side_effect=recording_import),
                mock.patch("papastop._initialize_windows_audio_thread", side_effect=initialize_audio_thread),
            ):
                service._capture_loop(lambda kind, payload: events.append((kind, payload)), stop_event)

            # 무음만 들어왔으므로 발화 청크는 없고 종료 신호만 큐에 남는다.
            sentinel = service._chunk_queue.get_nowait()
            self.assertEqual(sentinel, (None, None))
        finally:
            if previous_soundcard is None:
                sys.modules.pop("soundcard", None)
            else:
                sys.modules["soundcard"] = previous_soundcard

        self.assertTrue(recorder.recorded_after_enter)
        self.assertTrue(recorder.exited)
        self.assertEqual(call_order[0], "import")
        self.assertLess(call_order.index("import"), call_order.index("init"))
        self.assertLess(call_order.index("init"), call_order.index("speaker"))
        self.assertEqual(call_order[-1], "cleanup")
        self.assertNotIn("error", [kind for kind, _ in events])
        logs = [payload for kind, payload in events if kind == "log"]
        self.assertIn("PC 스피커 캡처를 준비합니다.", logs)
        self.assertIn("PC 스피커 캡처를 시작했습니다.", logs)
        self.assertIn("PC 스피커 캡처 장치를 정리했습니다.", logs)
        self.assertIn("PC 스피커 캡처 스레드가 종료되었습니다.", logs)


    def test_stop_prepares_windows_audio_thread_before_closing_recorder(self):
        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=10, chunk_seconds=0.1)
        call_order = []
        service._stop_event = threading.Event()
        service._loopback_recorder = types.SimpleNamespace(close=lambda: call_order.append("close"))

        def initialize_audio_thread():
            call_order.append("init")
            return lambda: call_order.append("cleanup")

        with mock.patch("papastop._initialize_windows_audio_thread", side_effect=initialize_audio_thread):
            events = []
            service.stop(timeout=0, emit=lambda kind, payload: events.append((kind, payload)))

        self.assertEqual(call_order, ["init", "close", "cleanup"])
        self.assertIsNone(service._loopback_recorder)
        logs = [payload for kind, payload in events if kind == "log"]
        self.assertIn("실시간 자막 내부 작업에 중지 신호를 보냈습니다.", logs)
        self.assertIn("PC 스피커 캡처 장치를 닫는 중입니다.", logs)
        self.assertIn("PC 스피커 캡처 장치를 닫았습니다.", logs)
        self.assertIn("실시간 자막 내부 작업 정리가 완료되었습니다.", logs)


    def test_safe_put_chunk_counts_dropped_chunks(self):
        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=16000, chunk_seconds=1.0)
        service._chunk_queue = queue.Queue(maxsize=2)
        # 큐를 가득 채운 뒤 더 넣으면 오래된 청크를 버린다(드롭 카운트 증가).
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            for i in range(2):
                service._safe_put_chunk((i, base / f"c{i}.wav"))
            self.assertEqual(service._dropped_chunks, 0)
            service._safe_put_chunk((2, base / "c2.wav"))
            service._safe_put_chunk((3, base / "c3.wav"))
            self.assertEqual(service._dropped_chunks, 2)
            # 종료 신호(None)는 드롭으로 세지 않는다.
            service._safe_put_chunk((None, None))
            self.assertEqual(service._dropped_chunks, 2)

    def test_report_processing_lag_warns_and_throttles(self):
        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=16000, chunk_seconds=5.0)
        events = []
        emit = lambda kind, payload: events.append((kind, payload))

        # 발화 길이보다 빠르게 처리하면 경고 없음 (elapsed, 발화길이, 인식시간, 번역시간)
        service._report_processing_lag(1.0, 5.0, 0.5, 0.5, emit)
        self.assertEqual(events, [])

        # 발화 길이보다 느리면 경고(live_status + log), 한 번만
        service._dropped_chunks = 3
        service._report_processing_lag(8.0, 5.0, 3.0, 5.0, emit)
        kinds = [k for k, _ in events]
        self.assertIn("live_status", kinds)
        self.assertIn("log", kinds)
        self.assertIn("건너뜀 3", events[0][1])
        # 경고에 인식·번역 분리 시간이 표시된다
        self.assertIn("인식 3.0초", events[0][1])
        self.assertIn("번역 5.0초", events[0][1])

        # 10초 throttle: 곧바로 다시 호출하면 추가 경고 없음
        before = len(events)
        service._report_processing_lag(9.0, 5.0, 3.0, 6.0, emit)
        self.assertEqual(len(events), before)

    def test_append_live_srt_offsets_by_start_time(self):
        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=16000, chunk_seconds=5.0)
        events = []
        emit = lambda kind, payload: events.append((kind, payload))

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "live.srt"
            service._srt_path = srt_path
            service._srt_rows = []
            service._srt_emitted = False

            # 발화 시작 시각(절대 초)을 직접 오프셋으로 사용한다.
            service._append_live_srt(10.0, [(SubtitleSegment(0.0, 1.5, "원문무시"), "안녕하세요")], emit)
            service._append_live_srt(20.0, [(SubtitleSegment(0.5, 2.0, "원문무시"), "반갑습니다")], emit)

            content = srt_path.read_text(encoding="utf-8")

        # 누적·시간 보정 확인
        self.assertEqual(len(service._srt_rows), 2)
        self.assertIn("00:00:10,000 --> 00:00:11,500", content)  # 10 + 0.0 ~ 10 + 1.5
        self.assertIn("00:00:20,500 --> 00:00:22,000", content)  # 20 + 0.5 ~ 20 + 2.0
        self.assertIn("안녕하세요", content)
        self.assertIn("반갑습니다", content)
        # 저장 경로는 한 번만 알림
        self.assertEqual([k for k, _ in events].count("live_subtitle_output"), 1)


class UtteranceSegmenterTests(unittest.TestCase):
    def _make(self, fake_fn, **kwargs):
        results = []
        segmenter = _UtteranceSegmenter(
            16000,
            on_utterance=lambda start, clip: results.append((start, len(clip))),
            speech_timestamps_fn=fake_fn,
            **kwargs,
        )
        return segmenter, results

    def test_emits_utterance_after_trailing_silence(self):
        # 버퍼가 1초(16000샘플) 이상이면 [0,16000) 구간이 음성이라고 답하는 가짜 VAD
        fake = lambda audio, sr: [{"start": 0, "end": 16000}] if len(audio) >= 16000 else []
        segmenter, results = self._make(fake, min_silence_ms=400, max_utterance_s=8.0)
        frame = np.zeros(8000, dtype=np.float32)  # 0.5초

        segmenter.add_frame(frame)   # 버퍼 8000: 음성 없음
        self.assertEqual(results, [])
        segmenter.add_frame(frame)   # 버퍼 16000: 마지막 구간이 끝에 닿아 진행 중 → 보류
        self.assertEqual(results, [])
        segmenter.add_frame(frame)   # 버퍼 24000: 끝에 침묵(8000) 확보 → 확정 방출
        self.assertEqual(results, [(0.0, 16000)])

    def test_force_flush_at_max_utterance(self):
        # 버퍼 전체가 늘 음성(침묵이 안 끝남)이면 최대 길이에서 강제 분할
        fake = lambda audio, sr: [{"start": 0, "end": len(audio)}]
        segmenter, results = self._make(fake, min_silence_ms=2000, max_utterance_s=1.0)
        frame = np.zeros(8000, dtype=np.float32)

        segmenter.add_frame(frame)   # 버퍼 8000 < 최대(16000): 진행 중 → 보류
        self.assertEqual(results, [])
        segmenter.add_frame(frame)   # 버퍼 16000 >= 최대: 강제 확정
        self.assertEqual(results, [(0.0, 16000)])

    def test_flush_remaining_emits_pending(self):
        fake = lambda audio, sr: [{"start": 0, "end": len(audio)}]
        segmenter, results = self._make(fake, min_silence_ms=2000, max_utterance_s=8.0)

        segmenter.add_frame(np.zeros(8000, dtype=np.float32))  # 진행 중 → 보류
        self.assertEqual(results, [])
        segmenter.flush_remaining()  # 종료 시 남은 음성 방출
        self.assertEqual(results, [(0.0, 8000)])

    def test_stereo_frame_is_downmixed_to_mono(self):
        # 스테레오(2채널) 프레임이 들어와도 mono로 합쳐 길이가 채널 수만큼 줄지 않는다.
        fake = lambda audio, sr: [{"start": 0, "end": len(audio)}]
        segmenter, results = self._make(fake, min_silence_ms=2000, max_utterance_s=8.0)

        segmenter.add_frame(np.zeros((8000, 2), dtype=np.float32))
        segmenter.flush_remaining()
        self.assertEqual(results, [(0.0, 8000)])


if __name__ == "__main__":
    unittest.main()
