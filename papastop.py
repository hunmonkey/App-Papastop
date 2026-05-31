from __future__ import annotations

import importlib.util
import os
import queue
import re
import shutil
import subprocess
import hashlib
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_NAME = "PapaStop"
APP_ROOT = Path(__file__).resolve().parent
PAPASTOP_DATA = APP_ROOT / ".papastop_data"
HF_HOME = PAPASTOP_DATA / "huggingface"

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_RECORD_SAMPLE_RATE = 48_000
DEFAULT_RECORD_FPS = 30

WHISPER_MODEL_ORDER = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
WHISPER_MODEL_LABELS = {
    "tiny": "매우 빠름 (tiny)",
    "base": "기본 (base)",
    "small": "균형 (small)",
    "medium": "정확도 높음 (medium)",
    "large-v3": "최고 정확도 (large-v3)",
    "large-v3-turbo": "고정확도 빠름 (large-v3-turbo)",
}

MEDIA_EXTS = [
    "*.mp4",
    "*.mkv",
    "*.avi",
    "*.mov",
    "*.wmv",
    "*.flv",
    "*.webm",
    "*.m4v",
    "*.mp3",
    "*.wav",
    "*.m4a",
    "*.aac",
    "*.flac",
]
MEDIA_EXT_SET = {ext.lower().removeprefix("*.") for ext in MEDIA_EXTS}

LIVE_SOURCE_SPEAKER = "speaker"
LIVE_SOURCE_URL = "url"

URL_SCHEMES = {"http", "https", "rtmp", "rtmps", "rtsp"}

NLLB_LANGUAGE_MAP = {
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
}

NLLB_FAST_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_ACCURATE_MODEL = "facebook/nllb-200-3.3B"

TRANSLATION_PROFILE_ORDER = ("auto", "fast", "balanced", "accurate")
TRANSLATION_PROFILE_LABELS = {
    "auto": "자동 (NLLB 우선, 실패 시 Argos)",
    "fast": "빠름 (Argos 번역)",
    "balanced": "균형 (NLLB distilled)",
    "accurate": "정확도 우선 (NLLB 3.3B)",
}
TRANSLATION_PROFILE_ORDER_STEPS = {
    "auto": (("nllb", NLLB_FAST_MODEL), ("argos", None)),
    "fast": (("argos", None),),
    "balanced": (("nllb", NLLB_FAST_MODEL),),
    "accurate": (("nllb", NLLB_ACCURATE_MODEL), ("nllb", NLLB_FAST_MODEL)),
}


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


class OperationCancelled(RuntimeError):
    pass


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def windows_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_command(command: list[str], on_line: Callable[[str], None] | None = None, cwd: Path | None = None) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=windows_creation_flags(),
    )

    assert process.stdout is not None
    for line in process.stdout:
        if on_line:
            on_line(line.rstrip())

    if process.wait() != 0:
        raise RuntimeError(f"명령 실행에 실패했습니다: {' '.join(command)}")


def find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path

    if has_module("imageio_ffmpeg"):
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            return get_ffmpeg_exe()
        except Exception:
            return None

    return None


def ensure_cache_dirs() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME / "home"))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def format_srt_time(total_seconds: float) -> str:
    milliseconds = max(0, int(round(total_seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def clean_text_for_srt(text: str) -> str:
    fixed = re.sub(r"\s+", " ", text.strip())
    return fixed


def default_recording_path(output_dir: Path | None = None) -> Path:
    folder = output_dir if output_dir is not None else PAPASTOP_DATA / "recordings"
    return folder / f"screen-{time.strftime('%Y%m%d-%H%M%S')}.mkv"


def build_lossless_screen_capture_command(ffmpeg: str, output_path: Path, fps: int) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "gdigrab",
        "-framerate",
        str(max(1, fps)),
        "-draw_mouse",
        "1",
        "-i",
        "desktop",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "bgra",
        str(output_path),
    ]


def float_audio_to_pcm16_bytes(audio: object) -> tuple[bytes, int, int]:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return b"", 1, 0

    if array.ndim == 1:
        array = array.reshape((-1, 1))
    elif array.ndim != 2:
        raise RuntimeError(f"지원하지 않는 오디오 형식입니다: {array.ndim}차원")

    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767).astype("<i2", copy=False)
    frames = int(pcm.shape[0])
    channels = int(pcm.shape[1])
    return pcm.tobytes(), channels, frames


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        txt = clean_text_for_srt(segment.text)
        if not txt:
            continue
        lines.append(str(index))
        lines.append(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}")
        lines.append(txt)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_float_audio_to_wav(audio: object, sample_rate: int, output_path: Path) -> None:
    pcm_bytes, channels, frames = float_audio_to_pcm16_bytes(audio)
    if not pcm_bytes or frames == 0:
        raise RuntimeError("비어 있는 오디오 조각입니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    if frames == 0:
        raise RuntimeError("WAV 파일에 쓸 오디오가 없습니다.")


def extract_audio(media_path: str, output_wav: Path, on_line: Callable[[str], None] | None = None) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg가 없습니다. 설치한 뒤 다시 시도해 주세요.")

    source = str(media_path)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    run_command(command, on_line=on_line)


def nllb_lang(code: str | None) -> str:
    if not code:
        return "eng_Latn"
    normalized = str(code).strip().lower().split("-")[0]
    return NLLB_LANGUAGE_MAP.get(normalized, "eng_Latn")


def is_http_media_source(source: str) -> bool:
    parsed = urlparse(source.strip())
    return bool(parsed.scheme and parsed.scheme.lower() in URL_SCHEMES and parsed.netloc)


def source_stem_from_url(source: str) -> str:
    parsed = urlparse(source.strip())
    base = Path(parsed.path).stem
    if base:
        return base

    safe_host = re.sub(r"[^a-zA-Z0-9._-]", "_", parsed.netloc or "stream")
    stamp = str(int(time.time()))
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{safe_host}_{stamp}_{digest}"


def source_stem(source: str) -> str:
    if is_http_media_source(source):
        return source_stem_from_url(source)
    path = Path(source)
    if not path.stem:
        return "subtitle"
    return path.stem


def default_srt_path(source: str, output_dir: Path | None = None) -> Path:
    stem_name = source_stem(source)
    if output_dir is None:
        if is_http_media_source(source):
            return Path(f"{stem_name}.srt")
        return Path(source).with_suffix(".srt")

    output_dir.mkdir(parents=True, exist_ok=True)
    if is_http_media_source(source):
        return output_dir / f"{stem_name}.srt"
    return output_dir / Path(source).with_suffix(".srt").name


def build_subtitle_overwrite_prompt(paths: list[Path]) -> str:
    shown = "\n".join(f"- {path.name}" for path in paths[:5])
    hidden_count = max(0, len(paths) - 5)
    hidden_text = f"\n...외 {hidden_count}개" if hidden_count else ""
    return (
        "같은 이름의 자막 파일이 이미 있습니다.\n\n"
        f"{shown}{hidden_text}\n\n"
        "덮어쓸까요?"
    )


def write_pcm16_to_wav(pcm_bytes: bytes, sample_rate: int, output_path: Path) -> None:
    if not pcm_bytes:
        raise RuntimeError("비어 있는 PCM 오디오 조각입니다.")

    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    if not pcm_bytes:
        raise RuntimeError("비어 있는 PCM 오디오 조각입니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = len(pcm_bytes) // 2
    if frames <= 0:
        raise RuntimeError("비어 있는 PCM 오디오 조각입니다.")

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


class ASREngine:
    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL):
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def set_model(self, model_name: str) -> None:
        chosen = model_name if model_name in WHISPER_MODEL_LABELS else DEFAULT_WHISPER_MODEL
        with self._load_lock:
            if chosen == self.model_name:
                return
            self.model_name = chosen
            self._model = None

    def prepare_model(self, model_name: str | None = None) -> None:
        if model_name:
            self.set_model(model_name)
        self._load_model()

    def model_label(self, model_name: str | None = None) -> str:
        target = model_name or self.model_name
        return WHISPER_MODEL_LABELS.get(target, target)

    def _load_model(self):
        if self._model is not None:
            return

        if not has_module("faster_whisper"):
            raise RuntimeError("faster-whisper가 설치되어 있지 않습니다.")

        with self._load_lock:
            if self._model is not None:
                return

            from faster_whisper import WhisperModel
            import torch

            ensure_cache_dirs()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: Path,
        on_line=None,
        should_stop: Callable[[], bool] | None = None,
        on_segment: Callable[[SubtitleSegment], None] | None = None,
    ) -> tuple[list[SubtitleSegment], str]:
        self._load_model()
        assert self._model is not None

        if should_stop is not None and should_stop():
            raise OperationCancelled("취소되었습니다.")

        segments, info = self._model.transcribe(
            str(audio_path),
            language=None,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        detected_language = getattr(info, "language", "auto")
        rows: list[SubtitleSegment] = []
        for segment in segments:
            if should_stop is not None and should_stop():
                raise OperationCancelled("취소되었습니다.")

            text = (segment.text or "").strip()
            if not text:
                continue
            subtitle_segment = SubtitleSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
            )
            rows.append(subtitle_segment)
            if on_segment is not None:
                on_segment(subtitle_segment)

        if on_line:
            on_line(f"감지된 언어: {detected_language}, 자막 줄 수={len(rows)}")

        return rows, detected_language


class TranslationEngine:
    def __init__(self, preferred: str = "auto"):
        self.preferred = preferred
        self._nllb_pipelines: dict[str, object] = {}
        self._pipeline_lock = threading.Lock()
        self._prepare_lock = threading.Lock()

    def set_profile(self, profile: str) -> None:
        if profile not in TRANSLATION_PROFILE_ORDER_STEPS:
            profile = "auto"
        self.preferred = profile

    def _profile_steps(self, profile: str | None = None) -> tuple[tuple[str, str | None], ...]:
        selected = profile or self.preferred
        return TRANSLATION_PROFILE_ORDER_STEPS.get(selected, TRANSLATION_PROFILE_ORDER_STEPS["auto"])

    def prepare_profile(self, profile: str | None = None) -> None:
        chosen = profile or self.preferred
        if chosen not in TRANSLATION_PROFILE_ORDER_STEPS:
            chosen = "auto"

        with self._prepare_lock:
            for engine, model_name in self._profile_steps(chosen):
                if engine == "argos":
                    if not has_module("argostranslate"):
                        raise RuntimeError("Argos 번역 도구가 설치되어 있지 않습니다.")
                    continue
                if engine == "nllb":
                    assert model_name is not None
                    self._load_nllb(model_name)
                else:
                    raise RuntimeError(f"알 수 없는 번역 방식입니다: {engine}")

    def _load_nllb(self, model_name: str):
        if not has_module("transformers"):
            raise RuntimeError("transformers가 설치되어 있지 않습니다.")

        cached = self._nllb_pipelines.get(model_name)
        if cached is not None:
            return cached

        with self._pipeline_lock:
            cached = self._nllb_pipelines.get(model_name)
            if cached is not None:
                return cached

            if not model_name:
                raise RuntimeError("NLLB 모델 이름이 비어 있습니다.")
            from transformers import pipeline
            import torch

            ensure_cache_dirs()
            device = 0 if torch.cuda.is_available() else -1
            pipeline_obj = pipeline(
                "translation",
                model=model_name,
                device=device,
            )
            self._nllb_pipelines[model_name] = pipeline_obj
            return pipeline_obj

    def _translate_nllb(self, texts: list[str], source_code: str, model_name: str) -> list[str]:
        if not texts:
            return []
        pipeline = self._load_nllb(model_name)

        results = pipeline(
            texts,
            src_lang=nllb_lang(source_code),
            tgt_lang="kor_Hang",
            max_length=300,
        )

        translated: list[str] = []
        for item in results:
            if isinstance(item, dict):
                translated.append(item.get("translation_text", "").strip())
            elif isinstance(item, str):
                translated.append(item.strip())
            else:
                translated.append(str(item).strip())

        return translated

    def _translate_argos(self, texts: list[str], source_code: str) -> list[str]:
        if not has_module("argostranslate"):
            raise RuntimeError("Argos 번역 도구가 설치되어 있지 않습니다.")
        if not texts:
            return []

        import argostranslate.translate as argostranslate_translate

        result: list[str] = []
        target = "ko"
        source = (source_code or "en").split("-")[0].lower()
        for text in texts:
            try:
                result.append(argostranslate_translate.translate(text, source, target))
            except Exception:
                result.append(text)

        return result

    def translate_texts(self, texts: list[str], source_code: str) -> list[str]:
        if not texts:
            return []

        for engine, model_name in self._profile_steps():
            try:
                if engine == "nllb":
                    assert model_name is not None
                    return self._translate_nllb(texts, source_code, model_name)
                if engine == "argos":
                    return self._translate_argos(texts, source_code)
                raise RuntimeError(f"알 수 없는 번역 방식입니다: {engine}")
            except Exception:
                if self.preferred == "fast":
                    raise
                continue

        # fallback: no translation but keeps flow alive
        return list(texts)

    def profile_label(self, profile: str | None = None) -> str:
        target = profile or self.preferred
        return TRANSLATION_PROFILE_LABELS.get(target, target)


class FileSubtitleWorkflow:
    def __init__(self, asr: ASREngine, translator: TranslationEngine):
        self.asr = asr
        self.translator = translator

    def run(
        self,
        media_source: str,
        output_srt: Path,
        on_line: Callable[[str], None],
        on_progress: Callable[[float], None],
        should_stop: Callable[[], bool] | None = None,
        on_segment: Callable[[SubtitleSegment], None] | None = None,
    ) -> Path:
        if should_stop is not None and should_stop():
            raise OperationCancelled("취소되었습니다.")

        source = media_source.strip()
        if not source:
            raise RuntimeError("영상 또는 음성 파일을 선택해 주세요.")

        with tempfile.TemporaryDirectory(prefix="papastop_file_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            on_progress(5)
            on_line("오디오를 추출하는 중 ...")
            extract_audio(source, wav_path, on_line=on_line)

            if should_stop is not None and should_stop():
                raise OperationCancelled("취소되었습니다.")

            on_progress(30)
            on_line("음성을 글자로 바꾸는 중 ...")
            segments, source_language = self.asr.transcribe(
                wav_path,
                on_line=on_line,
                should_stop=should_stop,
                on_segment=on_segment,
            )

            if not segments:
                raise RuntimeError("인식할 수 있는 음성을 찾지 못했습니다.")

            on_progress(55)
            on_line("한국어로 번역하는 중 ...")
            translated = self.translator.translate_texts([s.text for s in segments], source_language)
            if len(translated) != len(segments):
                translated = translated + ["" for _ in range(len(segments) - len(translated))]

            translated_segments: list[SubtitleSegment] = []
            for original, text in zip(segments, translated):
                if should_stop is not None and should_stop():
                    raise OperationCancelled("취소되었습니다.")
                subtitle_text = (text or original.text).strip()
                if subtitle_text:
                    translated_segments.append(
                        SubtitleSegment(
                            start=original.start,
                            end=original.end,
                            text=subtitle_text,
                        )
                    )

            on_progress(85)
            on_line(f"SRT 저장: {output_srt}")
            write_srt(translated_segments, output_srt)

        on_progress(100)
        return output_srt


class LiveCaptionService:
    def __init__(
        self,
        asr: ASREngine,
        translator: TranslationEngine,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    ):
        self.asr = asr
        self.translator = translator
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_frames = int(sample_rate * chunk_seconds)
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._chunk_queue: queue.Queue[tuple[int | None, Path | None]] | None = None
        self._temp_dir: Path | None = None
        self._ffmpeg_process: subprocess.Popen | None = None
        self._source_url: str | None = None
        self._loopback_recorder: object | None = None

    def _is_running(self) -> bool:
        return (
            (self._capture_thread is not None and self._capture_thread.is_alive())
            or (self._worker_thread is not None and self._worker_thread.is_alive())
        )

    def start(
        self,
        emit: Callable[[str, object], None],
        stop_event: threading.Event,
        internet_url: str | None = None,
    ) -> None:
        if self._is_running():
            raise RuntimeError("실시간 자막 기능이 이미 실행 중입니다.")

        self._stop_event = stop_event
        stop_event.clear()
        self._chunk_queue = queue.Queue(maxsize=8)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="papastop_live_"))
        self._source_url = internet_url.strip() if internet_url else None

        if self._source_url:
            capture_target = self._capture_from_url
            capture_args = (self._source_url, emit, stop_event)
        else:
            capture_target = self._capture_loop
            capture_args = (emit, stop_event)

        self._capture_thread = threading.Thread(target=capture_target, args=capture_args, daemon=True)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(emit, stop_event),
            daemon=True,
        )

        self._capture_thread.start()
        self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._ffmpeg_process is not None:
            process = self._ffmpeg_process
            self._ffmpeg_process = None
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=1)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        loopback_recorder = self._loopback_recorder
        if loopback_recorder is not None:
            self._loopback_recorder = None
            try:
                loopback_recorder.close()
            except Exception:
                pass

        if self._chunk_queue is not None:
            try:
                self._chunk_queue.put_nowait((None, None))
            except queue.Full:
                try:
                    self._chunk_queue.get_nowait()
                    self._chunk_queue.put_nowait((None, None))
                except Exception:
                    pass

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=timeout)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

        self._capture_thread = None
        self._worker_thread = None
        self._stop_event = None
        if self._chunk_queue is not None:
            while True:
                try:
                    _, leftover = self._chunk_queue.get_nowait()
                    if leftover and leftover.exists():
                        try:
                            leftover.unlink(missing_ok=True)
                        except OSError:
                            pass
                except queue.Empty:
                    break
            self._chunk_queue = None

        self._source_url = None

        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except OSError:
                pass
            self._temp_dir = None

    def _safe_put_chunk(self, item: tuple[int | None, Path | None]) -> None:
        if self._chunk_queue is None:
            return
        try:
            self._chunk_queue.put_nowait(item)
        except queue.Full:
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._chunk_queue.put_nowait(item)
            except queue.Full:
                pass

    def _capture_loop(self, emit: Callable[[str, object], None], stop_event: threading.Event) -> None:
        if os.name != "nt":
            emit("error", "실시간 캡처는 Windows에서만 사용할 수 있습니다.")
            self._safe_put_chunk((None, None))
            return

        if not has_module("soundcard"):
            emit("error", "PC 소리 캡처 도구(soundcard)가 설치되어 있지 않습니다.")
            self._safe_put_chunk((None, None))
            return

        import warnings

        import soundcard as sc

        try:
            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError("기본 스피커를 찾지 못했습니다.")

            emit("log", f"기본 스피커: {speaker.name}")
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with loopback.recorder(samplerate=self.sample_rate) as recorder:
                self._loopback_recorder = recorder
                try:
                    chunk_index = 0
                    while not stop_event.is_set():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="data discontinuity in recording",
                            )
                            audio = recorder.record(numframes=self.chunk_frames)

                        if stop_event.is_set():
                            break
                        if self._temp_dir is None:
                            break
                        if not getattr(audio, "size", 1):
                            continue

                        chunk_path = self._temp_dir / f"chunk-{chunk_index:06d}.wav"
                        save_float_audio_to_wav(audio, self.sample_rate, chunk_path)
                        self._safe_put_chunk((chunk_index, chunk_path))
                        chunk_index += 1
                finally:
                    self._loopback_recorder = None
        except OperationCancelled:
            emit("log", "실시간 캡처가 취소되었습니다.")
        except Exception as exc:
            if stop_event.is_set():
                emit("log", f"실시간 캡처가 중지되었습니다: {exc}")
            else:
                emit("error", f"실시간 캡처 오류: {exc}")
        finally:
            self._safe_put_chunk((None, None))

    def _capture_from_url(
        self,
        url: str,
        emit: Callable[[str, object], None],
        stop_event: threading.Event,
    ) -> None:
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            emit("error", "ffmpeg가 없어 인터넷 실시간 캡처를 사용할 수 없습니다.")
            self._safe_put_chunk((None, None))
            return

        if not is_http_media_source(url):
            emit("error", "실시간 URL은 http, https, rtmp, rtmps, rtsp 중 하나로 시작해야 합니다.")
            self._safe_put_chunk((None, None))
            return

        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-i",
            url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]

        process = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=windows_creation_flags(),
            )
            self._ffmpeg_process = process
            emit("log", f"인터넷 실시간 캡처를 시작했습니다: {url}")

            if process.stdout is None:
                raise RuntimeError("ffmpeg에서 오디오 스트림을 읽지 못했습니다.")

            chunk_bytes = self.chunk_frames * 2
            pending = bytearray()
            chunk_index = 0
            while not stop_event.is_set():
                need = chunk_bytes - len(pending)
                if need > 0:
                    chunk = process.stdout.read(need)
                    if chunk:
                        pending.extend(chunk)
                    elif not pending:
                        break

                if len(pending) < chunk_bytes:
                    continue

                data = bytes(pending[:chunk_bytes])
                del pending[:chunk_bytes]

                if stop_event.is_set():
                    break
                if not self._temp_dir:
                    break

                chunk_path = self._temp_dir / f"chunk-url-{chunk_index:06d}.wav"
                try:
                    write_pcm16_to_wav(data, self.sample_rate, chunk_path)
                except Exception:
                    break
                self._safe_put_chunk((chunk_index, chunk_path))
                chunk_index += 1
        except OperationCancelled:
            emit("log", "실시간 캡처가 취소되었습니다.")
        except Exception as exc:
            emit("error", f"실시간 캡처 오류: {url} - {exc}")
        finally:
            if process is not None:
                try:
                    if process.stdout is not None:
                        process.stdout.close()
                except Exception:
                    pass
                try:
                    process.terminate()
                except Exception:
                    pass
            self._safe_put_chunk((None, None))

    def _worker_loop(self, emit: Callable[[str, object], None], stop_event: threading.Event) -> None:
        while True:
            if self._chunk_queue is None:
                return

            try:
                chunk = self._chunk_queue.get(timeout=0.25)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            chunk_index, chunk_path = chunk
            if chunk_index is None:
                break
            if chunk_path is None:
                continue
            if stop_event.is_set():
                break

            try:
                segments, source_language = self.asr.transcribe(
                    chunk_path,
                    should_stop=stop_event.is_set,
                )
                if segments:
                    translated = self.translator.translate_texts(
                        [s.text for s in segments],
                        source_language,
                    )
                    if len(translated) != len(segments):
                        translated = translated + ["" for _ in range(len(segments) - len(translated))]
                    lines = [t.strip() or s.text.strip() for s, t in zip(segments, translated) if (t or s.text).strip()]
                    if lines:
                        emit("live_caption", "\n".join(lines[-2:]))
            except OperationCancelled:
                emit("log", "실시간 번역이 취소되었습니다.")
            except Exception as exc:
                emit("error", f"실시간 번역 오류: {chunk_path.name} - {exc}")
            finally:
                if chunk_path.exists():
                    try:
                        chunk_path.unlink(missing_ok=True)
                    except OSError:
                        pass


class ScreenRecorderService:
    def __init__(
        self,
        sample_rate: int = DEFAULT_RECORD_SAMPLE_RATE,
        audio_block_seconds: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.audio_block_frames = max(1024, int(sample_rate * audio_block_seconds))
        self._stop_event: threading.Event | None = None
        self._monitor_thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._video_process: subprocess.Popen | None = None
        self._audio_recorder: object | None = None
        self._temp_dir: Path | None = None
        self._video_path: Path | None = None
        self._audio_path: Path | None = None
        self._output_path: Path | None = None
        self._capture_audio = True
        self._audio_has_data = False

    def is_running(self) -> bool:
        process_running = self._video_process is not None and self._video_process.poll() is None
        return process_running or (
            self._monitor_thread is not None and self._monitor_thread.is_alive()
        )

    def start(
        self,
        output_path: Path,
        fps: int,
        capture_audio: bool,
        emit: Callable[[str, object], None],
        stop_event: threading.Event,
    ) -> None:
        if self.is_running():
            raise RuntimeError("화면 녹화가 이미 실행 중입니다.")
        if os.name != "nt":
            raise RuntimeError("화면 녹화는 Windows에서만 사용할 수 있습니다.")

        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("화면 녹화에는 ffmpeg가 필요합니다.")
        if capture_audio and not has_module("soundcard"):
            raise RuntimeError("PC 스피커 소리 녹음에는 soundcard가 필요합니다.")

        output_path = output_path.with_suffix(".mkv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event = stop_event
        stop_event.clear()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="papastop_record_"))
        self._video_path = self._temp_dir / "screen-video.mkv"
        self._audio_path = self._temp_dir / "speaker-audio.wav"
        self._output_path = output_path
        self._capture_audio = capture_audio
        self._audio_has_data = False

        command = build_lossless_screen_capture_command(ffmpeg, self._video_path, fps)
        self._video_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
        )

        if capture_audio:
            self._audio_thread = threading.Thread(
                target=self._record_loopback_audio,
                args=(emit, stop_event),
                daemon=True,
            )
            self._audio_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_recording,
            args=(emit,),
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        self._finish_video_process(timeout=timeout)

    def _record_loopback_audio(
        self,
        emit: Callable[[str, object], None],
        stop_event: threading.Event,
    ) -> None:
        import warnings

        import soundcard as sc

        if self._audio_path is None:
            return

        wav_file: wave.Wave_write | None = None
        expected_channels: int | None = None
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError("기본 스피커를 찾지 못했습니다.")

            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with loopback.recorder(samplerate=self.sample_rate) as recorder:
                self._audio_recorder = recorder
                try:
                    while not stop_event.is_set():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="data discontinuity in recording",
                            )
                            audio = recorder.record(numframes=self.audio_block_frames)

                        pcm_bytes, channels, frame_count = float_audio_to_pcm16_bytes(audio)
                        if frame_count <= 0 or not pcm_bytes:
                            continue

                        if wav_file is None:
                            self._audio_path.parent.mkdir(parents=True, exist_ok=True)
                            wav_file = wave.open(str(self._audio_path), "wb")
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(self.sample_rate)
                            expected_channels = channels
                        elif channels != expected_channels:
                            raise RuntimeError("녹음 중 스피커 소리 채널 수가 바뀌었습니다.")

                        wav_file.writeframes(pcm_bytes)
                        self._audio_has_data = True
                finally:
                    self._audio_recorder = None
        except Exception as exc:
            if not stop_event.is_set():
                emit("log", f"스피커 소리 녹음을 건너뛰었습니다: {exc}")
        finally:
            if wav_file is not None:
                wav_file.close()

    def _monitor_recording(self, emit: Callable[[str, object], None]) -> None:
        try:
            if self._output_path is None or self._video_path is None:
                raise RuntimeError("화면 녹화 준비가 끝나지 않았습니다.")

            emit("record_status", "녹화 중")
            emit("log", f"무손실 화면 녹화를 시작했습니다: {self._output_path}")

            process = self._video_process
            while self._stop_event is not None and not self._stop_event.is_set():
                if process is None or process.poll() is not None:
                    break
                time.sleep(0.3)

            emit("record_status", "녹화를 정리하는 중 ...")
            return_code = self._finish_video_process(timeout=5.0)

            if self._stop_event is not None:
                self._stop_event.set()
            if self._audio_thread is not None:
                self._audio_thread.join(timeout=3.0)

            if return_code not in (0, None):
                error_text = self._read_video_error()
                raise RuntimeError(error_text or f"ffmpeg가 오류 코드 {return_code}로 종료되었습니다.")
            if not self._video_path.exists() or self._video_path.stat().st_size <= 0:
                raise RuntimeError("화면 녹화 파일이 만들어지지 않았습니다.")

            emit("record_status", "저장 중 ...")
            saved_path = self._save_recording(emit)
            emit("record_output_path", str(saved_path))
            emit("record_status", "저장 완료")
            emit("done", f"화면 녹화를 저장했습니다: {saved_path}")
        except Exception as exc:
            emit("error", f"화면 녹화 오류: {exc}")
        finally:
            self._cleanup()
            emit("busy_record", False)
            emit("status", "대기 중")

    def _finish_video_process(self, timeout: float) -> int | None:
        process = self._video_process
        if process is None:
            return None

        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("q\n")
                    process.stdin.flush()
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

        try:
            return process.wait(timeout=timeout)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            try:
                return process.wait(timeout=1)
            except Exception:
                return 1

    def _read_video_error(self) -> str:
        process = self._video_process
        if process is None or process.stderr is None:
            return ""
        try:
            return process.stderr.read().strip()
        except Exception:
            return ""

    def _save_recording(self, emit: Callable[[str, object], None]) -> Path:
        if self._output_path is None or self._video_path is None:
            raise RuntimeError("화면 녹화 저장 위치가 준비되지 않았습니다.")

        audio_ready = (
            self._capture_audio
            and self._audio_has_data
            and self._audio_path is not None
            and self._audio_path.exists()
            and self._audio_path.stat().st_size > 44
        )

        if audio_ready:
            self._mux_video_and_audio(emit)
        else:
            if self._capture_audio:
                emit("log", "스피커 소리가 비어 있어 영상만 저장했습니다.")
            if self._output_path.exists():
                self._output_path.unlink()
            shutil.move(str(self._video_path), str(self._output_path))

        return self._output_path

    def _mux_video_and_audio(self, emit: Callable[[str, object], None]) -> None:
        if self._output_path is None or self._video_path is None or self._audio_path is None:
            raise RuntimeError("영상과 소리를 합칠 준비가 끝나지 않았습니다.")

        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("화면 녹화를 저장하려면 ffmpeg가 필요합니다.")

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self._video_path),
            "-i",
            str(self._audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(self._output_path),
        ]
        run_command(command, on_line=lambda line: emit("log", line))

    def _cleanup(self) -> None:
        self._video_process = None
        self._audio_thread = None
        self._monitor_thread = None
        self._audio_recorder = None
        self._stop_event = None
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except OSError:
                pass
        self._temp_dir = None
        self._video_path = None
        self._audio_path = None
        self._output_path = None
        self._audio_has_data = False


class LiveSubtitleOverlay(tk.Toplevel):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.withdraw()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg="#0f172a")
        self._drag_start_x = 0
        self._drag_start_y = 0

        # 윈도우 상단 바가 없는 상태에서도 마우스로 위치를 이동할 수 있게 드래그 이벤트 등록
        self.bind("<Button-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._on_drag)

        self.label = tk.Label(
            self,
            text="",
            fg="white",
            bg="#0f172a",
            font=("Arial", 26, "bold"),
            justify="center",
            wraplength=1000,
        )
        self.label.bind("<Button-1>", self._begin_drag)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.pack(fill="both", expand=True, padx=16, pady=16)

        screen_w = self.winfo_screenwidth()
        self.geometry(f"1100x220+{max(20, (screen_w - 1100) // 2)}+20")
        self.label.config(text="실시간 자막 준비 완료")
        self.deiconify()

    def _begin_drag(self, event: tk.Event) -> None:
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def _on_drag(self, event: tk.Event) -> None:
        self.geometry(f"+{event.x_root - self._drag_start_x + self.winfo_x()}+{event.y_root - self._drag_start_y + self.winfo_y()}")
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def set_text(self, text: str) -> None:
        if not text:
            return
        self.label.config(text=text)

    def close(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


class PapaStopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("980x700")
        self.minsize(900, 680)

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.asr = ASREngine()
        self.translator = TranslationEngine(preferred="auto")
        self.file_workflow = FileSubtitleWorkflow(self.asr, self.translator)
        self.live_service = LiveCaptionService(self.asr, self.translator)
        self.record_service = ScreenRecorderService()

        self.file_cancel_event = threading.Event()
        self.live_cancel_event = threading.Event()
        self.record_cancel_event = threading.Event()
        self.file_queue: list[str] = []
        self.file_preview_lines: list[str] = []
        self._batch_outputs: list[Path] = []
        self._batch_current = 0
        self.last_output: Path | None = None
        self.overlay: LiveSubtitleOverlay | None = None
        self._is_file_running = False
        self._is_live_running = False
        self._is_record_running = False
        self._is_asr_preparing = False
        self._is_translation_preparing = False

        self._build_ui()
        self._refresh_status()
        self.after(80, self._process_events)

    def _build_ui(self) -> None:
        root = self
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=12, pady=12)

        self.file_tab = ttk.Frame(nb)
        self.live_tab = ttk.Frame(nb)
        self.record_tab = ttk.Frame(nb)
        nb.add(self.file_tab, text="파일 자막")
        nb.add(self.live_tab, text="실시간 자막")
        nb.add(self.record_tab, text="화면 녹화")

        # File tab
        self.file_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.file_status_var = tk.StringVar(value="대기 중")
        self.batch_status_var = tk.StringVar(value="0개 파일")
        self.whisper_model_var = tk.StringVar(value=WHISPER_MODEL_LABELS[DEFAULT_WHISPER_MODEL])
        self.translation_profile_var = tk.StringVar(value=TRANSLATION_PROFILE_LABELS["auto"])
        self.live_url_var = tk.StringVar()
        self.live_source_var = tk.StringVar(value=LIVE_SOURCE_SPEAKER)
        self.record_output_var = tk.StringVar(value=str(default_recording_path()))
        self.record_status_var = tk.StringVar(value="대기 중")
        self.record_audio_var = tk.BooleanVar(value=True)
        self.record_fps_var = tk.StringVar(value=str(DEFAULT_RECORD_FPS))

        file_row0 = ttk.Frame(self.file_tab)
        file_row0.pack(fill="x", pady=6)
        ttk.Label(file_row0, text="원본 영상/음성").pack(side="left")
        ttk.Entry(file_row0, textvariable=self.file_path_var, width=80).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self.select_media_btn = ttk.Button(file_row0, text="선택", command=self._select_media)
        self.select_media_btn.pack(side="left")
        self.add_file_btn = ttk.Button(file_row0, text="목록에 추가", command=self._add_input_to_queue)
        self.add_file_btn.pack(side="left", padx=6)

        file_row1 = ttk.Frame(self.file_tab)
        file_row1.pack(fill="x", pady=6)
        ttk.Label(file_row1, text="저장할 SRT / 폴더").pack(side="left")
        ttk.Entry(file_row1, textvariable=self.output_path_var, width=80).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(file_row1, text="같은 이름으로", command=self._set_default_output).pack(side="left")
        ttk.Button(file_row1, text="폴더 선택", command=self._select_output_dir).pack(side="left", padx=6)

        file_row2 = ttk.Frame(self.file_tab)
        file_row2.pack(fill="x", pady=8)
        self.start_file_btn = ttk.Button(file_row2, text="자막 만들기 시작", command=self._start_file_job)
        self.cancel_file_btn = ttk.Button(file_row2, text="중지", command=self._cancel_file_job, state="disabled")
        self.start_file_btn.pack(side="left")
        self.cancel_file_btn.pack(side="left", padx=6)
        ttk.Button(file_row2, text="결과 열기", command=self._open_output).pack(side="right")

        file_row3 = ttk.Frame(self.file_tab)
        file_row3.pack(fill="x", pady=6)
        ttk.Label(file_row3, text="음성 인식 모델").pack(side="left")
        self.whisper_model_combo = ttk.Combobox(
            file_row3,
            textvariable=self.whisper_model_var,
            values=[WHISPER_MODEL_LABELS[key] for key in WHISPER_MODEL_ORDER],
            state="readonly",
            width=30,
        )
        self.whisper_model_combo.bind("<<ComboboxSelected>>", self._on_whisper_model_selected)
        self.whisper_model_combo.current(WHISPER_MODEL_ORDER.index(DEFAULT_WHISPER_MODEL))
        self.whisper_model_combo.pack(side="left", padx=8)
        self.prepare_asr_btn = ttk.Button(
            file_row3,
            text="위스퍼 모델 다운로드",
            command=self._prepare_asr_model,
        )
        self.prepare_asr_btn.pack(side="left", padx=6)

        file_row4 = ttk.Frame(self.file_tab)
        file_row4.pack(fill="x", pady=6)
        ttk.Label(file_row4, text="번역 모델").pack(side="left")
        self.translation_profile_combo = ttk.Combobox(
            file_row4,
            textvariable=self.translation_profile_var,
            values=[TRANSLATION_PROFILE_LABELS[key] for key in TRANSLATION_PROFILE_ORDER],
            state="readonly",
            width=45,
        )
        self.translation_profile_combo.bind("<<ComboboxSelected>>", self._on_translation_profile_selected)
        self.translation_profile_combo.current(0)
        self.translation_profile_combo.pack(side="left", padx=8)
        self.prepare_translation_btn = ttk.Button(
            file_row4,
            text="번역 모델 다운로드",
            command=self._prepare_translation_model,
        )
        self.prepare_translation_btn.pack(side="left", padx=6)

        queue_frame = ttk.LabelFrame(self.file_tab, text="작업할 파일 목록")
        queue_frame.pack(fill="both", expand=True, pady=(0, 8))

        list_area = ttk.Frame(queue_frame)
        list_area.pack(fill="both", expand=True, padx=6, pady=6)

        listbox_area = ttk.Frame(list_area)
        listbox_area.pack(side="left", fill="both", expand=True)
        self.file_listbox = tk.Listbox(
            listbox_area,
            height=7,
            selectmode="extended",
            exportselection=False,
        )
        list_scroll = ttk.Scrollbar(listbox_area, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=list_scroll.set)
        self.file_listbox.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="left", fill="y")

        queue_button_frame = ttk.Frame(list_area)
        queue_button_frame.pack(side="left", fill="y", padx=(8, 0))
        self.add_queue_btn = ttk.Button(queue_button_frame, text="파일 추가", command=self._add_files_to_queue)
        self.add_queue_btn.pack(fill="x")
        self.add_folder_btn = ttk.Button(queue_button_frame, text="폴더 추가", command=self._add_folder_to_queue)
        self.add_folder_btn.pack(fill="x", pady=4)
        self.remove_queue_btn = ttk.Button(queue_button_frame, text="선택 삭제", command=self._remove_selected_files)
        self.remove_queue_btn.pack(fill="x")
        self.clear_queue_btn = ttk.Button(queue_button_frame, text="전체 비우기", command=self._clear_file_queue)
        self.clear_queue_btn.pack(fill="x", pady=4)
        self.move_up_btn = ttk.Button(queue_button_frame, text="위로", command=self._move_selected_up)
        self.move_up_btn.pack(fill="x")
        self.move_down_btn = ttk.Button(queue_button_frame, text="아래로", command=self._move_selected_down)
        self.move_down_btn.pack(fill="x", pady=4)

        self._batch_controls = [
            self.select_media_btn,
            self.add_file_btn,
            self.add_queue_btn,
            self.add_folder_btn,
            self.remove_queue_btn,
            self.clear_queue_btn,
            self.move_up_btn,
            self.move_down_btn,
            self.start_file_btn,
            self.cancel_file_btn,
            self.whisper_model_combo,
            self.prepare_asr_btn,
            self.translation_profile_combo,
            self.prepare_translation_btn,
        ]

        file_info = ttk.Label(
            self.file_tab,
            text="영상이나 음성에서 한국어 SRT 자막을 만들어 저장합니다.",
        )
        file_info.pack(anchor="w", pady=6)

        ttk.Label(self.file_tab, text="파일 상태").pack(anchor="w")
        ttk.Label(self.file_tab, textvariable=self.file_status_var).pack(anchor="w")
        ttk.Label(self.file_tab, text="전체 진행").pack(anchor="w")
        ttk.Label(self.file_tab, textvariable=self.batch_status_var).pack(anchor="w")

        preview_frame = ttk.LabelFrame(self.file_tab, text="실시간 추출 미리보기")
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        preview_area = ttk.Frame(preview_frame)
        preview_area.pack(fill="both", expand=True, padx=6, pady=6)
        self.file_preview_text = tk.Text(preview_area, height=6, wrap="word")
        preview_scroll = ttk.Scrollbar(preview_area, orient="vertical", command=self.file_preview_text.yview)
        self.file_preview_text.configure(yscrollcommand=preview_scroll.set)
        self.file_preview_text.pack(side="left", fill="both", expand=True)
        preview_scroll.pack(side="left", fill="y")
        self.file_preview_text.insert("end", "자막 만들기를 시작하면 추출된 문장이 여기에 표시됩니다.")
        self.file_preview_text.configure(state="disabled")

        self._refresh_file_queue()

        # Live tab
        self.live_status_var = tk.StringVar(value="대기 중")
        self.live_preview_var = tk.StringVar(value="실시간 한국어 자막이 여기에 표시됩니다.")
        self.show_overlay_var = tk.BooleanVar(value=True)

        source_frame = ttk.LabelFrame(self.live_tab, text="실시간 소리 입력")
        source_frame.pack(fill="x", padx=4, pady=(0, 8))

        live_row0 = ttk.Frame(source_frame)
        live_row0.pack(fill="x", padx=8, pady=6)
        ttk.Label(live_row0, text="입력:").pack(side="left")
        ttk.Radiobutton(
            live_row0,
            text="PC 스피커 소리",
            variable=self.live_source_var,
            value=LIVE_SOURCE_SPEAKER,
        ).pack(side="left", padx=(8, 12))
        ttk.Radiobutton(
            live_row0,
            text="인터넷 URL 스트림",
            variable=self.live_source_var,
            value=LIVE_SOURCE_URL,
        ).pack(side="left")

        live_row_url = ttk.Frame(source_frame)
        live_row_url.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(live_row_url, text="URL").pack(side="left")
        ttk.Entry(live_row_url, textvariable=self.live_url_var, width=90).pack(
            side="left", fill="x", expand=True, padx=8
        )

        live_row1 = ttk.Frame(self.live_tab)
        live_row1.pack(fill="x", pady=6)
        self.start_live_btn = ttk.Button(live_row1, text="시작", command=self._start_live_job)
        self.stop_live_btn = ttk.Button(live_row1, text="중지", command=self._stop_live_job, state="disabled")
        self.start_live_btn.pack(side="left", padx=(12, 6))
        self.stop_live_btn.pack(side="left")

        live_row2 = ttk.Frame(self.live_tab)
        live_row2.pack(fill="x", pady=6)
        ttk.Checkbutton(
            live_row2,
            text="자막 창을 항상 위에 표시",
            variable=self.show_overlay_var,
            onvalue=True,
            offvalue=False,
        ).pack(side="left")

        ttk.Label(self.live_tab, text="실시간 자막 미리보기:").pack(anchor="w", pady=(12, 2))
        self.live_preview = tk.Message(
            self.live_tab,
            textvariable=self.live_preview_var,
            width=900,
            font=("Arial", 14),
            bg="#f8fafc",
            relief="sunken",
            padx=8,
            pady=6,
        )
        self.live_preview.pack(fill="x")
        ttk.Label(self.live_tab, textvariable=self.live_status_var).pack(anchor="w", pady=6)

        # Screen record tab
        record_source_frame = ttk.LabelFrame(self.record_tab, text="녹화 설정")
        record_source_frame.pack(fill="x", padx=4, pady=(0, 8))

        record_row0 = ttk.Frame(record_source_frame)
        record_row0.pack(fill="x", padx=8, pady=6)
        ttk.Label(record_row0, text="저장할 MKV").pack(side="left")
        self.record_output_entry = ttk.Entry(record_row0, textvariable=self.record_output_var, width=80)
        self.record_output_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.browse_record_btn = ttk.Button(record_row0, text="찾아보기", command=self._select_record_output)
        self.browse_record_btn.pack(side="left")

        record_row1 = ttk.Frame(record_source_frame)
        record_row1.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(record_row1, text="초당 화면 수").pack(side="left")
        self.record_fps_combo = ttk.Combobox(
            record_row1,
            textvariable=self.record_fps_var,
            values=("24", "30", "60"),
            state="readonly",
            width=6,
        )
        self.record_fps_combo.pack(side="left", padx=(8, 18))
        self.record_audio_check = ttk.Checkbutton(
            record_row1,
            text="PC 스피커 소리도 녹음",
            variable=self.record_audio_var,
            onvalue=True,
            offvalue=False,
        )
        self.record_audio_check.pack(side="left")

        record_row2 = ttk.Frame(self.record_tab)
        record_row2.pack(fill="x", pady=6)
        self.start_record_btn = ttk.Button(record_row2, text="녹화 시작", command=self._start_record_job)
        self.stop_record_btn = ttk.Button(
            record_row2,
            text="중지",
            command=self._stop_record_job,
            state="disabled",
        )
        self.start_record_btn.pack(side="left", padx=(12, 6))
        self.stop_record_btn.pack(side="left")
        ttk.Button(record_row2, text="결과 열기", command=self._open_output).pack(side="right", padx=(0, 12))

        ttk.Label(self.record_tab, textvariable=self.record_status_var).pack(anchor="w", pady=6)
        self._record_controls = [
            self.record_output_entry,
            self.browse_record_btn,
            self.record_fps_combo,
            self.record_audio_check,
            self.start_record_btn,
            self.stop_record_btn,
        ]

        # Bottom common
        common = ttk.LabelFrame(root, text="진행 상황과 작업 기록")
        common.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="대기 중")

        ttk.Label(common, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(6, 0))
        self.progress = ttk.Progressbar(common, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", padx=8, pady=6)

        self.log_text = tk.Text(common, height=14, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text.configure(state="disabled")

        self._append_log("앱을 시작했습니다.")
    def _append_log(self, msg: object) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_file_preview(self) -> None:
        self.file_preview_lines.clear()
        self.file_preview_text.configure(state="normal")
        self.file_preview_text.delete("1.0", tk.END)
        self.file_preview_text.configure(state="disabled")

    def _append_file_preview(self, line: object) -> None:
        text = str(line).strip()
        if not text:
            return
        self.file_preview_lines.append(text)
        if len(self.file_preview_lines) > 300:
            self.file_preview_lines = self.file_preview_lines[-300:]

        self.file_preview_text.configure(state="normal")
        self.file_preview_text.delete("1.0", tk.END)
        self.file_preview_text.insert("end", "\n".join(self.file_preview_lines))
        self.file_preview_text.see("end")
        self.file_preview_text.configure(state="disabled")

    def _emit(self, kind: str, payload: object) -> None:
        self.event_queue.put((kind, payload))

    def _process_events(self) -> None:
        drained = 0
        while drained < 80:
            drained += 1
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(payload)
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "file_status":
                self.file_status_var.set(str(payload))
            elif kind == "batch_status":
                self.batch_status_var.set(str(payload))
            elif kind == "file_preview":
                self._append_file_preview(payload)
            elif kind == "file_preview_clear":
                self._clear_file_preview()
            elif kind == "live_status":
                self.live_status_var.set(str(payload))
            elif kind == "record_status":
                self.record_status_var.set(str(payload))
            elif kind == "progress":
                self.progress_var.set(float(payload))
            elif kind == "output_path":
                self.last_output = Path(str(payload))
                self.output_path_var.set(str(self.last_output))
            elif kind == "record_output_path":
                self.last_output = Path(str(payload))
                self.record_output_var.set(str(self.last_output))
            elif kind == "live_caption":
                text = str(payload)
                self.live_preview_var.set(text)
                if self.overlay is not None:
                    self.overlay.set_text(text)
            elif kind == "busy_file":
                self._set_file_busy(bool(payload))
            elif kind == "busy_live":
                self._set_live_busy(bool(payload))
            elif kind == "busy_record":
                self._set_record_busy(bool(payload))
            elif kind == "error":
                self._append_log(payload)
                if self._is_live_running or self._is_file_running or self._is_record_running:
                    messagebox.showerror(APP_NAME, str(payload))
            elif kind == "done":
                self._append_log(payload)
                messagebox.showinfo(APP_NAME, str(payload))
            elif kind == "subtitle_output":
                path = Path(str(payload))
                self.last_output = path
                self.output_path_var.set(str(path))
            elif kind == "asr_prepare_done":
                payload_data = payload
                self._set_asr_preparing(False)
                if isinstance(payload_data, dict) and payload_data.get("message"):
                    self._append_log(payload_data["message"])
            elif kind == "translation_prepare_done":
                payload_data = payload
                self._set_translation_preparing(False)
                if isinstance(payload_data, dict) and payload_data.get("message"):
                    self._append_log(payload_data["message"])

            if kind == "error":
                if self._is_file_running:
                    self._set_file_busy(False)
                if self._is_live_running:
                    self._set_live_busy(False)
                if self._is_record_running:
                    self._set_record_busy(False)

        self.after(50, self._process_events)

    def _set_file_busy(self, busy: bool) -> None:
        self._is_file_running = busy
        for widget in self._batch_controls:
            state = "disabled" if busy else "normal"
            if widget is self.cancel_file_btn:
                state = "normal" if busy else "disabled"
            if widget is self.start_file_btn and busy:
                state = "disabled"
            if widget in (self.whisper_model_combo, self.translation_profile_combo) and not busy:
                state = "readonly"
            widget.configure(state=state)
        if self._is_asr_preparing:
            self.prepare_asr_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
        if self._is_translation_preparing:
            self.prepare_translation_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")

    def _set_asr_preparing(self, preparing: bool) -> None:
        self._is_asr_preparing = preparing
        self.prepare_asr_btn.configure(state="disabled" if preparing else "normal")
        self.prepare_translation_btn.configure(state="disabled" if preparing else "normal")
        self.start_file_btn.configure(state="disabled" if preparing else "normal")
        self.start_live_btn.configure(state="disabled" if preparing else "normal")
        self.start_record_btn.configure(state="disabled" if preparing else "normal")
        if self._is_translation_preparing:
            self.prepare_asr_btn.configure(state="disabled")
            self.prepare_translation_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
            self.start_live_btn.configure(state="disabled")
            self.start_record_btn.configure(state="disabled")
        if self._is_file_running or self._is_live_running or self._is_record_running:
            self.start_file_btn.configure(state="disabled")
            self.start_live_btn.configure(state="disabled")
            self.start_record_btn.configure(state="disabled")

    def _set_translation_preparing(self, preparing: bool) -> None:
        self._is_translation_preparing = preparing
        if preparing:
            self.prepare_translation_btn.configure(state="disabled")
        else:
            self.prepare_translation_btn.configure(state="normal")
        self.prepare_asr_btn.configure(state="disabled" if preparing else "normal")
        self.start_file_btn.configure(state="disabled" if preparing else "normal")
        self.start_live_btn.configure(state="disabled" if preparing else "normal")
        self.start_record_btn.configure(state="disabled" if preparing else "normal")
        if self._is_asr_preparing:
            self.prepare_asr_btn.configure(state="disabled")
            self.prepare_translation_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
            self.start_live_btn.configure(state="disabled")
            self.start_record_btn.configure(state="disabled")

    def _set_live_busy(self, busy: bool) -> None:
        self._is_live_running = busy
        self.start_live_btn.configure(state="disabled" if busy else "normal")
        self.stop_live_btn.configure(state="normal" if busy else "disabled")
        if not busy and self.overlay is not None:
            self.overlay.close()
            self.overlay = None

    def _set_record_busy(self, busy: bool) -> None:
        self._is_record_running = busy
        for widget in self._record_controls:
            state = "disabled" if busy else "normal"
            if widget is self.stop_record_btn:
                state = "normal" if busy else "disabled"
            if widget is self.record_fps_combo and not busy:
                state = "readonly"
            widget.configure(state=state)
        if self._is_asr_preparing or self._is_translation_preparing:
            self.start_record_btn.configure(state="disabled")

    def _select_record_output(self) -> None:
        initial_path = Path(self.record_output_var.get().strip() or str(default_recording_path()))
        path = filedialog.asksaveasfilename(
            defaultextension=".mkv",
            filetypes=[("MKV 영상", "*.mkv"), ("모든 파일", "*.*")],
            initialdir=str(initial_path.parent),
            initialfile=initial_path.name,
        )
        if not path:
            return
        self.record_output_var.set(str(Path(path).with_suffix(".mkv")))

    def _select_media(self) -> None:
        patterns = [("영상/음성 파일", " ".join(MEDIA_EXTS)), ("모든 파일", "*.*")]
        path = filedialog.askopenfilename(filetypes=patterns)
        if not path:
            return
        p = Path(path)
        self.file_path_var.set(str(p))
        self._set_default_output()

    def _set_default_output(self) -> None:
        media_text = self.file_path_var.get().strip()
        if not media_text:
            return
        if is_http_media_source(media_text):
            self.output_path_var.set(str(default_srt_path(media_text)))
        else:
            media_path = Path(media_text)
            if not media_path.exists():
                return
            self.output_path_var.set(str(media_path.with_suffix(".srt")))

    def _refresh_status(self) -> None:
        tools = []
        tools.append("faster-whisper: " + ("정상" if has_module("faster_whisper") else "없음"))
        tools.append("transformers: " + ("정상" if has_module("transformers") else "없음"))
        tools.append("argostranslate: " + ("정상" if has_module("argostranslate") else "없음"))
        tools.append("soundcard: " + ("정상" if has_module("soundcard") else "없음"))
        tools.append("ffmpeg: " + ("정상" if find_ffmpeg() else "없음"))
        tools.append("음성 인식 모델: " + self.asr.model_label())
        tools.append("번역 방식: " + self.translator.profile_label())
        self._append_log(" | ".join(tools))

    def _whisper_model_id_from_label(self, label: str) -> str:
        for key, value in WHISPER_MODEL_LABELS.items():
            if value == label:
                return key
        return DEFAULT_WHISPER_MODEL

    def _whisper_model_from_selection(self) -> str:
        return self._whisper_model_id_from_label(self.whisper_model_var.get())

    def _on_whisper_model_selected(self, _event: object | None = None) -> None:
        if self._is_file_running or self._is_live_running or self._is_asr_preparing:
            self.whisper_model_var.set(self.asr.model_label())
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 모델을 바꿔 주세요.")
            return
        model_name = self._whisper_model_from_selection()
        self.asr.set_model(model_name)
        self._append_log(f"음성 인식 모델을 바꿨습니다: {self.asr.model_label(model_name)}")

    def _prepare_asr_model(self) -> None:
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        model_name = self._whisper_model_from_selection()
        self.asr.set_model(model_name)
        self._set_asr_preparing(True)
        self._emit("log", f"위스퍼 모델을 미리 다운로드하는 중입니다: {self.asr.model_label(model_name)}")

        def worker() -> None:
            try:
                self.asr.prepare_model(model_name)
                self._emit(
                    "asr_prepare_done",
                    {"ok": True, "message": f"위스퍼 모델 준비 완료: {self.asr.model_label(model_name)}"},
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "asr_prepare_done",
                    {"ok": False, "message": f"위스퍼 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _translation_profile_id_from_label(self, label: str) -> str:
        for key, value in TRANSLATION_PROFILE_LABELS.items():
            if value == label:
                return key
        return "auto"

    def _translation_profile_from_selection(self) -> str:
        return self._translation_profile_id_from_label(self.translation_profile_var.get())

    def _on_translation_profile_selected(self, _event: object | None = None) -> None:
        if self._is_file_running or self._is_live_running or self._is_translation_preparing:
            self.translation_profile_var.set(self.translator.profile_label())
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 번역 모델을 바꿔 주세요.")
            return
        profile = self._translation_profile_from_selection()
        self.translator.set_profile(profile)
        self._append_log(f"번역 방식을 바꿨습니다: {self.translator.profile_label()}")

    def _prepare_translation_model(self) -> None:
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        profile = self._translation_profile_from_selection()
        self.translator.set_profile(profile)
        self._set_translation_preparing(True)
        self._emit("log", f"번역 모델을 준비하는 중입니다: {self.translator.profile_label()}")

        def worker() -> None:
            try:
                self.translator.prepare_profile(profile)
                self._emit(
                    "translation_prepare_done",
                    {"ok": True, "message": f"번역 모델 준비 완료: {self.translator.profile_label(profile)}"},
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "translation_prepare_done",
                    {"ok": False, "message": f"번역 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _select_output_dir(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.output_path_var.set(folder)

    def _append_to_queue(self, sources: list[str]) -> None:
        added = 0
        for source in sources:
            source_text = str(source).strip()
            if not source_text:
                continue

            if source_text in self.file_queue:
                continue

            if not is_http_media_source(source_text):
                path = Path(source_text)
                if not path.exists() or not path.is_file():
                    self._append_log(f"잘못된 파일이라 건너뜁니다: {source_text}")
                    continue
                source_text = str(path.resolve())

            self.file_queue.append(source_text)
            added += 1

        if added:
            self._refresh_file_queue()
            if self.file_path_var.get().strip():
                self._set_default_output()

    def _add_input_to_queue(self) -> None:
        source = self.file_path_var.get().strip()
        if not source:
            messagebox.showwarning(APP_NAME, "먼저 영상 또는 음성 파일을 선택해 주세요.")
            return
        self._append_to_queue([source])

    def _add_files_to_queue(self) -> None:
        patterns = [("영상/음성 파일", " ".join(MEDIA_EXTS)), ("모든 파일", "*.*")]
        paths = filedialog.askopenfilenames(filetypes=patterns)
        if not paths:
            return
        self._append_to_queue(list(paths))

    def _add_folder_to_queue(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            messagebox.showwarning(APP_NAME, "올바른 폴더가 아닙니다.")
            return

        found: list[str] = []
        for item in sorted(folder_path.rglob("*")):
            if not item.is_file():
                continue
            if item.suffix.lower().lstrip(".") not in MEDIA_EXT_SET:
                continue
            found.append(str(item))

        if not found:
            messagebox.showinfo(APP_NAME, "폴더 안에서 영상/음성 파일을 찾지 못했습니다.")
            return

        self._append_to_queue(found)

    def _remove_selected_files(self) -> None:
        selected = list(self.file_listbox.curselection())
        if not selected:
            return
        for index in reversed(sorted(selected)):
            del self.file_queue[index]
        self._refresh_file_queue()

    def _clear_file_queue(self) -> None:
        self.file_queue.clear()
        self._refresh_file_queue()

    def _move_selected_up(self) -> None:
        selected = list(self.file_listbox.curselection())
        if len(selected) != 1:
            return
        index = selected[0]
        if index <= 0:
            return
        self.file_queue[index - 1], self.file_queue[index] = self.file_queue[index], self.file_queue[index - 1]
        self._refresh_file_queue()
        self.file_listbox.selection_set(index - 1)

    def _move_selected_down(self) -> None:
        selected = list(self.file_listbox.curselection())
        if len(selected) != 1:
            return
        index = selected[0]
        if index >= len(self.file_queue) - 1:
            return
        self.file_queue[index + 1], self.file_queue[index] = self.file_queue[index], self.file_queue[index + 1]
        self._refresh_file_queue()
        self.file_listbox.selection_set(index + 1)

    def _refresh_file_queue(self) -> None:
        self.file_listbox.delete(0, tk.END)
        for source in self.file_queue:
            self.file_listbox.insert(tk.END, source)
        self.batch_status_var.set(f"{len(self.file_queue)}개 파일 대기 중")

    def _collect_batch_sources(self) -> list[str]:
        if self.file_queue:
            selected = list(self.file_listbox.curselection())
            if selected:
                return [self.file_queue[idx] for idx in selected]
            return list(self.file_queue)

        source = self.file_path_var.get().strip()
        return [source] if source else []

    def _start_file_job(self) -> None:
        if self._is_asr_preparing or self._is_translation_preparing:
            messagebox.showwarning(APP_NAME, "모델 준비가 끝날 때까지 기다려 주세요.")
            return

        if self._is_file_running or self._is_live_running or self._is_record_running:
            return

        self.asr.set_model(self._whisper_model_from_selection())

        media_sources = self._collect_batch_sources()
        if not media_sources:
            messagebox.showwarning(APP_NAME, "영상 또는 음성 파일을 선택하고 목록에 추가해 주세요.")
            return

        output_text = self.output_path_var.get().strip()
        total = len(media_sources)
        output_paths: list[Path] = []
        try:
            for source in media_sources:
                output_path = self._resolve_output_for_source(source, output_text, total)
                output_paths.append(output_path)

            for output_path in output_paths:
                output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showwarning(APP_NAME, str(exc))
            return

        existing_outputs = [path for path in output_paths if path.exists()]
        if existing_outputs:
            if not messagebox.askyesno(
                APP_NAME,
                build_subtitle_overwrite_prompt(existing_outputs),
            ):
                return

        for path in media_sources:
            if is_http_media_source(path):
                continue
            source_path = Path(path)
            if not source_path.exists() or not source_path.is_file():
                messagebox.showwarning(APP_NAME, f"올바른 영상/음성 파일이 아닙니다: {path}")
                return

        self.file_cancel_event.clear()
        self._set_file_busy(True)
        self.progress_var.set(0)
        self._clear_file_preview()
        self.batch_status_var.set(f"1/{total} 시작 중")
        self._batch_outputs = output_paths
        self._batch_current = 0
        self._emit("file_status", "처리 중")
        self._emit("status", f"파일 자막 작업 ({total}개)")

        def update_for_overall_progress(file_index: int, file_progress: float) -> None:
            base = ((file_index) / total) * 100
            overall = base + (max(0.0, min(100.0, file_progress)) / total)
            self._emit("progress", overall)

        def worker() -> None:
            saved_count = 0
            failed_count = 0
            try:
                for index, media_source in enumerate(media_sources, start=1):
                    self._batch_current = index
                    if self.file_cancel_event.is_set():
                        break

                    target_output = output_paths[index - 1]
                    display_name = Path(media_source).name if not is_http_media_source(media_source) else media_source
                    self._emit("file_status", f"처리 중 {index}/{total}: {display_name}")
                    self._emit("batch_status", f"{index}/{total}개 파일")
                    self._emit("file_preview", f"--- {index}/{total} {display_name} ---")
                    self._emit("progress", ((index - 1) / total) * 100)

                    try:
                        def preview_segment(segment: SubtitleSegment) -> None:
                            start_text = format_srt_time(segment.start)
                            self._emit("file_preview", f"{start_text}  {segment.text}")

                        saved = self.file_workflow.run(
                            media_source=media_source,
                            output_srt=target_output,
                            on_line=lambda m, s=media_source: self._emit("log", f"[{index}/{total}] {Path(s).name if not is_http_media_source(s) else s} - {m}"),
                            on_progress=lambda p, i=index: update_for_overall_progress(i - 1, p),
                            should_stop=self.file_cancel_event.is_set,
                            on_segment=preview_segment,
                        )
                        saved_count += 1
                        self._emit("subtitle_output", str(saved))
                        self._emit("output_path", str(saved))
                        self._emit("log", f"[{index}/{total}] 저장 완료: {saved}")
                    except OperationCancelled:
                        break
                    except Exception as exc:
                        failed_count += 1
                        self._emit("log", f"[{index}/{total}] 실패: {exc}")
                        continue

                    if self.file_cancel_event.is_set():
                        break

                if self.file_cancel_event.is_set():
                    self._emit("progress", (saved_count / total) * 100)
                    self._emit("file_status", f"취소됨 ({saved_count}/{total})")
                    self._emit("status", "취소됨")
                else:
                    self._emit("progress", (saved_count / total) * 100 if saved_count < total else 100)
                    self._emit("file_status", f"완료 ({saved_count}/{total})")
                    self._emit("done", f"작업이 끝났습니다. 성공: {saved_count}개, 실패: {failed_count}개")
            except Exception as exc:
                self._emit("error", str(exc))
            finally:
                self._emit("busy_file", False)
                self._emit("status", "대기 중")
                if total > 0:
                    self._emit("batch_status", f"{saved_count}/{total}개 완료, {failed_count}개 실패")
                else:
                    self._emit("batch_status", "0개 파일")

        self._emit("busy_file", True)
        threading.Thread(target=worker, daemon=True).start()

    def _resolve_output_for_source(self, media_source: str, output_text: str, total_files: int) -> Path:
        if total_files <= 0:
            return default_srt_path(media_source)

        if output_text:
            output_path = Path(output_text)
            if output_path.exists():
                if output_path.is_file():
                    if total_files > 1:
                        raise RuntimeError("여러 파일을 처리할 때는 파일 하나가 아니라 저장 폴더를 선택해 주세요.")
                    return output_path

                return default_srt_path(media_source, output_path)

            if total_files > 1:
                if output_path.suffix:
                    raise RuntimeError("여러 파일을 처리할 때는 파일 하나가 아니라 저장 폴더를 선택해 주세요.")
                return default_srt_path(media_source, output_path)

            return output_path

        return default_srt_path(media_source)

    def _cancel_file_job(self) -> None:
        if self._is_file_running:
            self.file_cancel_event.set()
            self._append_log("파일 자막 작업 중지를 요청했습니다.")

    def _start_record_job(self) -> None:
        if self._is_asr_preparing or self._is_translation_preparing:
            messagebox.showwarning(APP_NAME, "모델 준비가 끝날 때까지 기다려 주세요.")
            return
        if self._is_file_running or self._is_live_running:
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 다시 시도해 주세요.")
            return
        if self._is_record_running:
            return
        if os.name != "nt":
            messagebox.showwarning(APP_NAME, "화면 녹화는 Windows에서만 사용할 수 있습니다.")
            return
        if not find_ffmpeg():
            messagebox.showwarning(APP_NAME, "화면 녹화에는 ffmpeg가 필요합니다.")
            return
        if self.record_audio_var.get() and not has_module("soundcard"):
            messagebox.showwarning(APP_NAME, "PC 스피커 소리를 녹음하려면 soundcard가 필요합니다.")
            return

        output_text = self.record_output_var.get().strip()
        output_path = Path(output_text).with_suffix(".mkv") if output_text else default_recording_path()
        self.record_output_var.set(str(output_path))

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showwarning(APP_NAME, f"저장 폴더를 만들 수 없습니다: {exc}")
            return

        if output_path.exists() and not messagebox.askyesno(
            APP_NAME,
            "이미 같은 이름의 파일이 있습니다.\n덮어쓰고 계속할까요?",
        ):
            return

        try:
            fps = int(self.record_fps_var.get())
        except ValueError:
            fps = DEFAULT_RECORD_FPS

        self.record_cancel_event = threading.Event()
        self.record_cancel_event.clear()
        self._emit("busy_record", True)
        self._emit("record_status", "시작 중 ...")
        self._emit("status", "화면 녹화 중")

        def worker() -> None:
            try:
                self.record_service.start(
                    output_path=output_path,
                    fps=fps,
                    capture_audio=bool(self.record_audio_var.get()),
                    emit=self._emit,
                    stop_event=self.record_cancel_event,
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit("busy_record", False)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_record_job(self) -> None:
        if not self._is_record_running:
            return

        self._emit("record_status", "중지하는 중 ...")
        try:
            self.record_service.stop()
        except Exception as exc:
            self._append_log(f"화면 녹화 중지 오류: {exc}")

    def _start_live_job(self) -> None:
        if self._is_asr_preparing or self._is_translation_preparing:
            messagebox.showwarning(APP_NAME, "모델 준비가 끝날 때까지 기다려 주세요.")
            return

        if self._is_live_running or self._is_file_running or self._is_record_running:
            return

        self.asr.set_model(self._whisper_model_from_selection())

        live_source = self.live_source_var.get()
        live_url = self.live_url_var.get().strip() if live_source == LIVE_SOURCE_URL else ""

        if live_source == LIVE_SOURCE_URL:
            if not live_url:
                messagebox.showwarning(APP_NAME, "인터넷 URL을 입력해 주세요.")
                return
            if not is_http_media_source(live_url):
                messagebox.showwarning(APP_NAME, "URL은 http, https, rtmp, rtmps, rtsp 중 하나로 시작해야 합니다.")
                return
            if not find_ffmpeg():
                messagebox.showwarning(APP_NAME, "인터넷 실시간 캡처에는 ffmpeg가 필요합니다.")
                return
        else:
            if not has_module("soundcard"):
                messagebox.showwarning(APP_NAME, "PC 스피커 실시간 자막에는 soundcard가 필요합니다.")
                return

        self.live_cancel_event = threading.Event()
        self.live_cancel_event.clear()
        self._emit("busy_live", True)
        self._emit("live_status", "연결 중 ...")
        self._emit("status", "실시간 자막")

        if self.show_overlay_var.get():
            self.overlay = LiveSubtitleOverlay(self)
            self.overlay.lift()

        def worker() -> None:
            try:
                self.live_service.start(
                    self._emit,
                    self.live_cancel_event,
                    internet_url=live_url if live_source == LIVE_SOURCE_URL else None,
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit("busy_live", False)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_live_job(self) -> None:
        if not self._is_live_running:
            return

        self._emit("live_status", "중지하는 중 ...")
        try:
            self.live_service.stop()
        except Exception as exc:
            self._append_log(f"실시간 자막 중지 오류: {exc}")
        self._emit("busy_live", False)
        self._emit("live_status", "대기 중")
        self._emit("status", "대기 중")

    def _open_output(self) -> None:
        if self.last_output is None or not self.last_output.exists():
            messagebox.showwarning(APP_NAME, "아직 열 수 있는 결과 파일이 없습니다.")
            return
        if os.name == "nt":
            os.startfile(str(self.last_output))
        else:
            self._emit("log", f"이 운영체제에서는 자동 열기를 사용할 수 없습니다: {self.last_output}")
def main() -> None:
    if not APP_ROOT.exists():
        raise RuntimeError(f"앱 폴더를 확인할 수 없습니다: {APP_ROOT}")

    app = PapaStopApp()
    app.mainloop()


if __name__ == "__main__":
    main()



