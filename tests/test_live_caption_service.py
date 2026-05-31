import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papastop import LiveCaptionService


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


if __name__ == "__main__":
    unittest.main()
