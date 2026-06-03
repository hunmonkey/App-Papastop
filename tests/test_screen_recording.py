import unittest
from pathlib import Path

import numpy as np

from papastop import (
    build_screen_capture_command,
    default_recording_path,
    float_audio_to_pcm16_bytes,
)


class ScreenRecordingTests(unittest.TestCase):
    def test_lossless_screen_capture_command_uses_gdigrab_and_ffv1(self):
        output = Path("capture.mkv")
        command = build_screen_capture_command("ffmpeg", output, 30)

        self.assertIn("gdigrab", command)
        self.assertIn("desktop", command)
        self.assertIn("ffv1", command)
        self.assertIn("bgra", command)
        self.assertEqual(command[-1], str(output))

    def test_default_recording_path_uses_mkv(self):
        path = default_recording_path(Path("recordings"))

        self.assertEqual(path.parent, Path("recordings"))
        self.assertEqual(path.suffix, ".mkv")
        self.assertTrue(path.name.startswith("screen-"))

    def test_float_audio_to_pcm16_bytes_keeps_channels(self):
        audio = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
        pcm_bytes, channels, frames = float_audio_to_pcm16_bytes(audio)

        self.assertEqual(channels, 2)
        self.assertEqual(frames, 2)
        self.assertEqual(len(pcm_bytes), 8)


if __name__ == "__main__":
    unittest.main()
