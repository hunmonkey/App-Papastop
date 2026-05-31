import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papastop import LiveCaptionService, SubtitleSegment


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
        microphone = types.SimpleNamespace(recorder=lambda samplerate: recorder)
        soundcard = types.SimpleNamespace(
            default_speaker=lambda: types.SimpleNamespace(name="test speaker"),
            get_microphone=lambda id, include_loopback: microphone,
        )

        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=10, chunk_seconds=0.1)
        events = []

        previous_soundcard = sys.modules.get("soundcard")
        sys.modules["soundcard"] = soundcard
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                service._temp_dir = Path(temp_dir)
                service._chunk_queue = queue.Queue(maxsize=8)

                with mock.patch("papastop.os.name", "nt"), mock.patch("papastop.has_module", return_value=True):
                    service._capture_loop(lambda kind, payload: events.append((kind, payload)), stop_event)

                chunk_index, chunk_path = service._chunk_queue.get_nowait()
                sentinel = service._chunk_queue.get_nowait()

                self.assertEqual(chunk_index, 0)
                self.assertTrue(chunk_path.exists())
                self.assertEqual(sentinel, (None, None))
        finally:
            if previous_soundcard is None:
                sys.modules.pop("soundcard", None)
            else:
                sys.modules["soundcard"] = previous_soundcard

        self.assertTrue(recorder.recorded_after_enter)
        self.assertTrue(recorder.exited)
        self.assertNotIn("error", [kind for kind, _ in events])


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

        # 빠르면 경고 없음
        service._report_processing_lag(1.0, emit)
        self.assertEqual(events, [])

        # 느리면 경고(live_status + log), 한 번만
        service._dropped_chunks = 3
        service._report_processing_lag(8.0, emit)
        kinds = [k for k, _ in events]
        self.assertIn("live_status", kinds)
        self.assertIn("log", kinds)
        self.assertIn("건너뜀 3", events[0][1])

        # 10초 throttle: 곧바로 다시 호출하면 추가 경고 없음
        before = len(events)
        service._report_processing_lag(9.0, emit)
        self.assertEqual(len(events), before)

    def test_append_live_srt_offsets_by_chunk_index(self):
        service = LiveCaptionService(asr=object(), translator=object(), sample_rate=16000, chunk_seconds=5.0)
        events = []
        emit = lambda kind, payload: events.append((kind, payload))

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "live.srt"
            service._srt_path = srt_path
            service._srt_rows = []
            service._srt_emitted = False

            # 청크 2 (오프셋 10초), 청크 4 (오프셋 20초)
            service._append_live_srt(2, [(SubtitleSegment(0.0, 1.5, "원문무시"), "안녕하세요")], emit)
            service._append_live_srt(4, [(SubtitleSegment(0.5, 2.0, "원문무시"), "반갑습니다")], emit)

            content = srt_path.read_text(encoding="utf-8")

        # 누적·시간 보정 확인
        self.assertEqual(len(service._srt_rows), 2)
        self.assertIn("00:00:10,000 --> 00:00:11,500", content)  # 청크 2 = 0+10 ~ 1.5+10
        self.assertIn("00:00:20,500 --> 00:00:22,000", content)  # 청크 4 = 0.5+20 ~ 2.0+20
        self.assertIn("안녕하세요", content)
        self.assertIn("반갑습니다", content)
        # 저장 경로는 한 번만 알림
        self.assertEqual([k for k, _ in events].count("live_subtitle_output"), 1)


if __name__ == "__main__":
    unittest.main()
