from __future__ import annotations

import functools
import importlib.util
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
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
from tkinter import filedialog, messagebox

import customtkinter as ctk


APP_NAME = "PapaStop"
# 배포(PyInstaller EXE)에서 __file__은 임시 추출 폴더를 가리킨다.
# EXE 옆에 동봉한 .papastop_data(모델·CUDA 라이브러리)를 찾으려면 실행 파일 기준 경로를 써야 한다.
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent
PAPASTOP_DATA = APP_ROOT / ".papastop_data"
HF_HOME = PAPASTOP_DATA / "huggingface"

DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_RECORD_SAMPLE_RATE = 48_000
DEFAULT_RECORD_FPS = 30

# 화면 녹화 영상 형식: 무손실(FFV1)과 고화질 압축(H.264) 중 선택.
RECORD_CODEC_ORDER = ("ffv1", "h264")
RECORD_CODEC_LABELS = {
    "ffv1": "무손실 (FFV1·매우 큼)",
    "h264": "고화질 압축 (H.264)",
}
DEFAULT_RECORD_CODEC = "ffv1"

# 캡처 대상 드롭다운 고정 라벨(나머지 "모니터 N ..."은 실행 시 동적 생성).
CAPTURE_FULL_LABEL = "전체 화면"
CAPTURE_REGION_LABEL = "영역 지정 (드래그)"

# 실시간 자막: 시계 기준 고정 분할 대신 silero VAD로 발화(말소리) 단위로 자른다.
LIVE_FRAME_SECONDS = 0.5            # 캡처 프레임 크기(이 단위로 모아 VAD 검사)
LIVE_MIN_SILENCE_MS = 400          # 이만큼 침묵하면 문장 끝으로 간주(빠른 반응 우선)
LIVE_MAX_UTTERANCE_SECONDS = 8.0   # 말이 계속 이어져도 이 길이에서 강제 분할
LIVE_SPEECH_PAD_MS = 200           # 발화 앞뒤 여유 패딩
# 실시간은 이미 발화 단위로 잘려 들어오므로 디코딩 폭(beam)을 좁혀 속도를 높인다.
# 파일 자막은 정확도 우선이라 영향받지 않는다. 더 빠르게 하려면 1(greedy)로 낮출 수 있다.
LIVE_BEAM_SIZE = 2
# 자동 감지 모드에서 같은 언어가 이만큼 연속 감지되면 그 언어로 잠시 고정한다.
# 짧은 발화마다 언어 감지가 흔들려 엉뚱한 언어로 번역되는 것을 막는다.
LIVE_STICKY_LANGUAGE_HITS = 3

WHISPER_MODEL_ORDER = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3")
WHISPER_MODEL_LABELS = {
    "tiny": "매우 빠름 (tiny)",
    "base": "기본 (base)",
    "small": "균형 (small)",
    "medium": "정확도 높음 (medium)",
    "large-v3": "최고 정확도 (large-v3)",
    "large-v3-turbo": "고정확도 빠름 (large-v3-turbo)",
    "distil-large-v3": "영어 전용 고속 (distil-large-v3)",
}

SOURCE_LANG_AUTO = "auto"
SOURCE_LANG_ORDER = ("auto", "en", "ja", "zh")
SOURCE_LANG_LABELS = {
    "auto": "자동 감지",
    "en": "영어",
    "ja": "일본어",
    "zh": "중국어",
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

NLLB_LIVE_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_FAST_MODEL = "facebook/nllb-200-distilled-1.3B"
NLLB_ACCURATE_MODEL = "facebook/nllb-200-3.3B"
NLLB_TOKENIZER_MODEL = NLLB_ACCURATE_MODEL
NLLB_CT2_MODEL_DIRS = {
    NLLB_LIVE_MODEL: PAPASTOP_DATA / "nllb-200-distilled-600M-ct2",
    NLLB_FAST_MODEL: PAPASTOP_DATA / "nllb-200-distilled-1.3B-ct2",
    NLLB_ACCURATE_MODEL: PAPASTOP_DATA / "nllb-200-3.3B-ct2",
}

# llama.cpp(GGUF) 기반 LLM 번역 모델 — 파일 자막 품질 우선용.
# 모두 단일 파일 GGUF. 기본은 전 레이어 GPU 적재(n_gpu_layers=-1)이며,
# 16GB VRAM을 넘는 대형 모델만 아래 옵션으로 조정한다.
#   - n_gpu_layers : GPU에 올릴 레이어 수(나머지는 CPU+RAM, 일부 CPU 분담 = 느려짐)
#   - exclusive_vram : 번역 직전 음성 인식 모델을 내려 VRAM을 비운다(대형 모델 OOM 방지)
#   - no_think : Qwen3 등 추론(thinking) 모델의 사고 과정을 끄는 지시를 덧붙인다
LLM_GGUF_DIR = PAPASTOP_DATA / "gguf"
LLM_TRANSLATION_CHUNK_SIZE = 8
LLM_MODEL_ORDER = (
    "exaone-7.8b",
    "qwen3-14b",
    "gemma3-12b",
    "gemma4-12b",
    "translategemma-12b",
    "aya-expanse-8b",
    "mistral-small-24b",
    "qwen3-30b-a3b",
    "gemma3-27b",
    "exaone4-32b",
)
LLM_MODELS = {
    "exaone-7.8b": {
        "repo": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF",
        "file": "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf",
        "label": "EXAONE 3.5 7.8B (한국어 특화)",
        "vram_gb": 5,
    },
    # ── 2025년 세대 모델(전 레이어 GPU 적재) ──────────────────────
    "qwen3-14b": {
        "repo": "bartowski/Qwen_Qwen3-14B-GGUF",
        "file": "Qwen_Qwen3-14B-Q4_K_M.gguf",
        "label": "Qwen3 14B (다국어 최신·고품질)",
        "vram_gb": 9,
        "no_think": True,
    },
    "gemma3-12b": {
        "repo": "bartowski/google_gemma-3-12b-it-GGUF",
        "file": "google_gemma-3-12b-it-Q4_K_M.gguf",
        "label": "Gemma 3 12B (다국어·한국어 강함)",
        "vram_gb": 8,
    },
    "gemma4-12b": {
        "repo": "ggml-org/gemma-4-12B-it-GGUF",
        "file": "gemma-4-12B-it-Q4_K_M.gguf",
        "label": "Gemma 4 12B (최신·고효율)",
        "vram_gb": 8,
    },
    "translategemma-12b": {
        "repo": "mradermacher/translategemma-12b-it-GGUF",
        "file": "translategemma-12b-it.Q4_K_M.gguf",
        "label": "TranslateGemma 12B (번역 특화)",
        "vram_gb": 9,
    },
    "aya-expanse-8b": {
        "repo": "bartowski/aya-expanse-8b-GGUF",
        "file": "aya-expanse-8b-Q4_K_M.gguf",
        "label": "Aya Expanse 8B (번역 특화 다국어)",
        "vram_gb": 6,
    },
    # ── 대형 최고 품질 모델(번역 직전 위스퍼 해제로 VRAM 확보) ────
    "mistral-small-24b": {
        "repo": "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "file": "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        "label": "Mistral Small 24B (효율 고품질)",
        "vram_gb": 14,
        "exclusive_vram": True,
    },
    # MoE: 30B지만 토큰당 3B만 계산 → 일부 CPU 분담이어도 대형치고 빠르다.
    "qwen3-30b-a3b": {
        "repo": "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        "file": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        "label": "Qwen3 30B-A3B (MoE·빠른 대형)",
        "vram_gb": 17,
        "n_gpu_layers": 36,
        "exclusive_vram": True,
        "no_think": True,
    },
    "gemma3-27b": {
        "repo": "bartowski/google_gemma-3-27b-it-GGUF",
        "file": "google_gemma-3-27b-it-Q3_K_M.gguf",
        "label": "Gemma 3 27B (최고 품질)",
        "vram_gb": 13,
        "exclusive_vram": True,
    },
    "exaone4-32b": {
        "repo": "LGAI-EXAONE/EXAONE-4.0-32B-GGUF",
        "file": "EXAONE-4.0-32B-IQ4_XS.gguf",
        "label": "EXAONE 4.0 32B (한국어 특화·일부 CPU)",
        "vram_gb": 16,
        "n_gpu_layers": 48,
        "exclusive_vram": True,
    },
}

# OPUS-MT(Marian) — 실시간용 경량 1차 번역(원본→영어).
# 한국어 직결 모델이 마땅치 않아 영어를 거친 뒤 2차 번역기로 한국어화한다.
OPUS_MT_MODELS = {
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
}

# LLM 번역 프롬프트에 원본 언어를 명시해 정확도를 높인다.
LLM_LANG_NAMES = {
    "ja": "Japanese",
    "zh": "Chinese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}

# OPUS 1차 번역 후 영어→한국어 2차 번역에 쓸 엔진 선택지.
LIVE_SECONDARY_ORDER = ("nllb600", "nllb1.3", "argos")
LIVE_SECONDARY_LABELS = {
    "nllb600": "NLLB 600M (빠름)",
    "nllb1.3": "NLLB 1.3B (정확)",
    "argos": "Argos (초경량)",
}
LIVE_SECONDARY_STEPS = {
    "nllb600": ("nllb", NLLB_LIVE_MODEL),
    "nllb1.3": ("nllb", NLLB_FAST_MODEL),
    "argos": ("argos", None),
}

TRANSLATION_PROFILE_ORDER = ("auto", "fast", "nllb600", "balanced", "accurate", "opus") + tuple(
    f"llm-{key}" for key in LLM_MODEL_ORDER
)
TRANSLATION_PROFILE_LABELS = {
    "auto": "자동 (NLLB 우선, 실패 시 Argos)",
    "fast": "빠름 (Argos 번역)",
    "nllb600": "경량 (NLLB 600M)",
    "balanced": "균형 (NLLB 1.3B)",
    "accurate": "정확도 우선 (NLLB 3.3B)",
    "live": "빠름 (NLLB 600M)",
}
TRANSLATION_PROFILE_ORDER_STEPS = {
    "auto": (("nllb", NLLB_FAST_MODEL), ("argos", None)),
    "fast": (("argos", None),),
    "nllb600": (("nllb", NLLB_LIVE_MODEL), ("nllb", NLLB_FAST_MODEL)),
    "balanced": (("nllb", NLLB_FAST_MODEL),),
    "accurate": (("nllb", NLLB_ACCURATE_MODEL), ("nllb", NLLB_FAST_MODEL)),
    # 실시간 전용: 600M 우선, 없으면 1.3B로, 그래도 안 되면 Argos로 내려간다.
    "live": (("nllb", NLLB_LIVE_MODEL), ("nllb", NLLB_FAST_MODEL), ("argos", None)),
}
# 파일 자막 LLM 프로필을 자동 생성한다. LLM 실패 시 NLLB 1.3B로 안전하게 내려간다.
for _llm_key in LLM_MODEL_ORDER:
    TRANSLATION_PROFILE_LABELS[f"llm-{_llm_key}"] = "고품질 LLM · " + LLM_MODELS[_llm_key]["label"]
    TRANSLATION_PROFILE_ORDER_STEPS[f"llm-{_llm_key}"] = (
        ("llama", _llm_key),
        ("nllb", NLLB_FAST_MODEL),
    )
# OPUS 2단계 프로필 진입점(set_profile 검증 통과용). 실제 2단계 처리는 엔진에서 분기한다.
TRANSLATION_PROFILE_ORDER_STEPS["opus"] = (("opus", None),)
TRANSLATION_PROFILE_LABELS["opus"] = "OPUS-MT (영어 1차→2차)"

# 실시간 탭 번역 모델은 파일 탭과 동일한 TRANSLATION_PROFILE_ORDER/LABELS를 그대로 사용한다.

DEFAULT_RESOURCE_MODE = "max"
RESOURCE_MODE_ORDER = ("max", "balanced")
RESOURCE_MODE_LABELS = {
    "max": "최대 성능 (CPU/GPU 적극 사용)",
    "balanced": "균형 (PC 여유 남김)",
}


# ──────────────────────────────────────────────────────────────
# 모던 라이트 UI 테마 색상 (Tailwind slate + indigo 계열)
# ──────────────────────────────────────────────────────────────
UI_BG = "#e6eaf2"            # 앱 전체 배경 (카드와 대비를 위해 약간 진한 회색)
UI_SURFACE = "#ffffff"       # 입력/목록/로그 등 콘텐츠 표면
UI_SURFACE_HOVER = "#eef2ff"  # 버튼 호버 등 옅은 강조 배경
UI_BORDER = "#d6dbe6"        # 옅은 경계선
UI_BORDER_STRONG = "#9aa6ba"  # 카드/입력 테두리 (버튼 옆에서도 또렷하게)
UI_CARD_BORDER = "#b0bac9"   # 카드 구분용 테두리
UI_TEXT = "#0f172a"          # 본문 텍스트
UI_TEXT_MUTED = "#64748b"    # 보조 텍스트
UI_ACCENT = "#6366f1"        # 옅은 인디고 강조색 (자극적이지 않게)
UI_ACCENT_HOVER = "#5457e6"
UI_ACCENT_ACTIVE = "#4548c9"
UI_ACCENT_SOFT = "#e0e7ff"   # 선택된 탭/세그먼트 배경 (옅은 인디고 틴트)
UI_ACCENT_SOFT_HOVER = "#c7d2fe"
UI_DANGER = "#dc2626"        # 중지/삭제/실패
UI_DANGER_HOVER = "#b91c1c"
UI_OK = "#16a34a"            # 모델 준비 완료 표시 (완료 상태에만 제한적 사용)
UI_WARN = "#d97706"          # 모델 다운로드 필요 표시
OVERLAY_BG = "#0f172a"       # 자막 오버레이 배경 (어두운 톤 유지)
OVERLAY_FG = "#f8fafc"
OVERLAY_IDLE_CLEAR_MS = 6000  # 새 자막이 이 시간(ms) 동안 없으면 오버레이의 오래된 자막을 지운다.

_UI_FONT_FAMILY_CACHE: str | None = None


def resolve_ui_font_family() -> str:
    """현재 시스템에서 쓸 수 있는 가장 모던한 글꼴을 한 번만 골라 둡니다."""
    global _UI_FONT_FAMILY_CACHE
    if _UI_FONT_FAMILY_CACHE is not None:
        return _UI_FONT_FAMILY_CACHE

    family = "Segoe UI"
    try:
        import tkinter.font as tkfont

        available = set(tkfont.families())
        for candidate in ("Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI", "Malgun Gothic"):
            if candidate in available:
                family = candidate
                break
    except Exception:
        family = "Segoe UI"

    _UI_FONT_FAMILY_CACHE = family
    return family


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ResourceSettings:
    mode: str
    cpu_count: int
    cpu_threads: int
    num_workers: int
    asr_device: str
    asr_compute_type: str
    asr_batch_size: int
    asr_beam_size: int
    translation_ct2_device: str
    translation_ct2_compute_type: str
    translation_device: int
    translation_batch_size: int
    translation_dtype: str | None
    summary: str


class OperationCancelled(RuntimeError):
    pass


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def windows_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _initialize_windows_audio_thread() -> Callable[[], None]:
    """Windows audio APIs need COM prepared separately in each worker thread."""
    if os.name != "nt":
        return lambda: None

    try:
        import ctypes

        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoUninitialize.argtypes = ()
        ole32.CoUninitialize.restype = None

        # COINIT_MULTITHREADED is appropriate for these background capture threads.
        hr = int(ole32.CoInitializeEx(None, 0))
    except AttributeError:
        return lambda: None
    except Exception as exc:
        raise RuntimeError(f"윈도우 소리 장치 준비 실패: {exc}") from exc

    normalized_hr = hr & 0xFFFFFFFF
    if normalized_hr in (0x00000000, 0x00000001):  # S_OK, S_FALSE
        return ole32.CoUninitialize
    if normalized_hr == 0x80010106:  # RPC_E_CHANGED_MODE: already initialized differently.
        return lambda: None
    if normalized_hr & 0x80000000:
        raise RuntimeError(f"윈도우 소리 장치 준비 실패: HRESULT 0x{normalized_hr:08x}")
    return lambda: None


def _close_windows_audio_recorder(recorder: object) -> None:
    release_audio_thread: Callable[[], None] = lambda: None
    try:
        release_audio_thread = _initialize_windows_audio_thread()
    except Exception:
        pass
    try:
        close = getattr(recorder, "close", None)
        if close is not None:
            close()
    finally:
        try:
            release_audio_thread()
        except Exception:
            pass


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


def format_duration_short(seconds: float | int | None) -> str:
    if seconds is None:
        return "계산 중"

    seconds = int(max(0, round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {seconds}초"
    return f"{seconds}초"


def build_eta_text(progress: float, started_at: float | None, now: float | None = None) -> str:
    if started_at is None:
        return "예상 시간: 대기 중"

    progress = max(0.0, min(100.0, float(progress)))
    if progress >= 100:
        return "예상 시간: 완료"
    if progress < 3:
        return "예상 시간: 계산 중"

    now = time.time() if now is None else now
    elapsed = max(0.0, now - started_at)
    if elapsed <= 0:
        return "예상 시간: 계산 중"

    remaining = elapsed * (100.0 - progress) / progress
    finish_time = time.strftime("%H:%M", time.localtime(now + remaining))
    return f"예상 시간: 약 {format_duration_short(remaining)} 남음 · 완료 예상 {finish_time}"


def normalize_resource_mode(mode: str | None) -> str:
    return mode if mode in RESOURCE_MODE_LABELS else DEFAULT_RESOURCE_MODE


_CUDA_DLL_CONFIGURED = False
_CUDA_DLL_HANDLES: list[object] = []


def configure_cuda_dll_paths() -> None:
    global _CUDA_DLL_CONFIGURED
    if _CUDA_DLL_CONFIGURED or os.name != "nt":
        return

    candidates: list[Path] = []

    # 배포(EXE)에서는 .papastop_data/nvidia에 CUDA 라이브러리를 동봉한다(개발 환경엔 없으므로 무시됨).
    bundled_root = PAPASTOP_DATA / "nvidia"
    candidates.extend(
        [
            bundled_root / "cublas" / "bin",
            bundled_root / "cudnn" / "bin",
            bundled_root / "cuda_runtime" / "bin",
            bundled_root / "cuda_nvrtc" / "bin",
        ]
    )

    try:
        import nvidia

        nvidia_root = Path(nvidia.__file__).resolve().parent
        candidates.extend(
            [
                nvidia_root / "cublas" / "bin",
                nvidia_root / "cudnn" / "bin",
                nvidia_root / "cuda_runtime" / "bin",
                nvidia_root / "cuda_nvrtc" / "bin",
            ]
        )
    except Exception:
        pass

    site_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    candidates.extend(
        [
            site_root / "cublas" / "bin",
            site_root / "cudnn" / "bin",
            site_root / "cuda_runtime" / "bin",
            site_root / "cuda_nvrtc" / "bin",
        ]
    )

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for path in candidates:
        if not path.exists():
            continue
        path_text = str(path)
        if hasattr(os, "add_dll_directory"):
            try:
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(path_text))
            except Exception:
                pass
        if path_text not in path_parts:
            path_parts.insert(0, path_text)

    os.environ["PATH"] = os.pathsep.join(path_parts)
    _CUDA_DLL_CONFIGURED = True


def get_ctranslate2_cuda_available() -> bool:
    configure_cuda_dll_paths()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def get_ctranslate2_compute_types(device: str) -> set[str]:
    configure_cuda_dll_paths()
    try:
        import ctranslate2

        return set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return set()


def get_torch_cuda_available() -> bool:
    if not has_module("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def get_primary_cuda_vram_bytes() -> int | None:
    # 총 VRAM은 실행 중 바뀌지 않는다. torch 없으면 nvidia-smi 서브프로세스(약 110ms)를 타는데,
    # build_resource_settings가 발화·번역마다 호출하므로 캐시해 매번 재조회하는 낭비를 없앤다.
    if has_module("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass

    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return None

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        memory_mib = _optional_int(line.strip())
        if memory_mib is not None:
            return memory_mib * 1024 * 1024
    return None


def get_nvidia_memory_used_mib() -> int | None:
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return None

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        memory_mib = _optional_int(line.strip())
        if memory_mib is not None:
            return memory_mib
    return None


def configure_runtime_threads(cpu_threads: int) -> None:
    cpu_threads = max(1, int(cpu_threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)

    if not has_module("torch"):
        return

    try:
        import torch

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except Exception:
            pass
    except Exception:
        pass


def build_resource_settings(mode: str | None = None) -> ResourceSettings:
    selected_mode = normalize_resource_mode(mode)
    cpu_count = max(1, os.cpu_count() or 1)
    if selected_mode == "max":
        cpu_threads = cpu_count
        num_workers = max(1, min(4, cpu_count // 2 or 1))
        asr_beam_size = 8
    else:
        cpu_threads = max(1, cpu_count - 2)
        num_workers = 1
        asr_beam_size = 5

    vram_bytes = get_primary_cuda_vram_bytes()
    asr_cuda = get_ctranslate2_cuda_available()
    torch_cuda = get_torch_cuda_available()

    asr_device = "cuda" if asr_cuda else "cpu"
    if asr_device == "cuda":
        cuda_types = get_ctranslate2_compute_types("cuda")
        if "float16" in cuda_types and (vram_bytes is None or vram_bytes >= 6 * 1024**3):
            asr_compute_type = "float16"
        elif "int8_float16" in cuda_types:
            asr_compute_type = "int8_float16"
        elif cuda_types:
            asr_compute_type = sorted(cuda_types)[0]
        else:
            asr_compute_type = "default"
    else:
        cpu_types = get_ctranslate2_compute_types("cpu")
        asr_compute_type = "int8" if "int8" in cpu_types else "default"

    if asr_device == "cuda":
        if vram_bytes is not None and vram_bytes >= 12 * 1024**3:
            asr_batch_size = 16 if selected_mode == "max" else 8
        else:
            asr_batch_size = 8 if selected_mode == "max" else 4
    else:
        asr_batch_size = 4 if selected_mode == "max" else 2

    translation_ct2_device = "cuda" if asr_cuda else "cpu"
    if translation_ct2_device == "cuda":
        cuda_types = get_ctranslate2_compute_types("cuda")
        if "int8_float16" in cuda_types:
            translation_ct2_compute_type = "int8_float16"
        elif "float16" in cuda_types:
            translation_ct2_compute_type = "float16"
        else:
            translation_ct2_compute_type = "default"
    else:
        cpu_types = get_ctranslate2_compute_types("cpu")
        translation_ct2_compute_type = "int8" if "int8" in cpu_types else "default"

    translation_device = 0 if torch_cuda else -1
    if translation_device >= 0:
        translation_dtype = "float16"
        translation_batch_size = 16 if selected_mode == "max" and vram_bytes and vram_bytes >= 12 * 1024**3 else 8
    else:
        translation_dtype = None
        translation_batch_size = 4 if selected_mode == "max" else 2
    if translation_ct2_device == "cuda":
        translation_batch_size = 16 if selected_mode == "max" and vram_bytes and vram_bytes >= 12 * 1024**3 else 8

    summary_parts = [
        f"CPU {cpu_threads}/{cpu_count}스레드",
        f"자막 {asr_device.upper()} {asr_compute_type}, 배치 {asr_batch_size}",
        f"번역 CT2 {translation_ct2_device.upper()} {translation_ct2_compute_type}, 배치 {translation_batch_size}",
    ]
    if asr_device == "cuda" and translation_device < 0 and has_module("torch"):
        summary_parts.append("PyTorch CUDA 없음")

    return ResourceSettings(
        mode=selected_mode,
        cpu_count=cpu_count,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_batch_size=asr_batch_size,
        asr_beam_size=asr_beam_size,
        translation_ct2_device=translation_ct2_device,
        translation_ct2_compute_type=translation_ct2_compute_type,
        translation_device=translation_device,
        translation_batch_size=translation_batch_size,
        translation_dtype=translation_dtype,
        summary=" | ".join(summary_parts),
    )


def format_bytes_as_gib(total_bytes: int | None) -> str:
    if total_bytes is None or total_bytes <= 0:
        return "알 수 없음"
    return f"{total_bytes / (1024 ** 3):.1f} GB"


def get_cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if str(value).strip():
                    return str(value).strip()
        except Exception:
            pass

    for candidate in (platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")):
        if candidate and candidate.strip():
            return candidate.strip()
    return "알 수 없음"


def get_total_ram_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return None
    return None


def _optional_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return number if number > 0 else None


def find_nvidia_smi() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None and os.name == "nt":
        candidate = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if candidate.exists():
            nvidia_smi = str(candidate)
    return nvidia_smi


def parse_nvidia_smi_csv(output: str) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue

        name = parts[0]
        memory_mib = _optional_int(parts[1])
        driver_version = parts[2]
        if not name or memory_mib is None:
            continue
        rows.append((name, memory_mib * 1024 * 1024, driver_version))
    return rows


def format_vga_spec(name: object, adapter_ram: object = None, driver_version: object = None) -> str:
    display_name = str(name or "").strip() or "알 수 없음"
    details: list[str] = []
    ram_bytes = _optional_int(adapter_ram)
    if ram_bytes is not None:
        details.append(f"VRAM {format_bytes_as_gib(ram_bytes)}")
    if driver_version:
        details.append(f"드라이버 {str(driver_version).strip()}")
    if details:
        return f"{display_name} ({', '.join(details)})"
    return display_name


def _run_powershell_json(script: str, timeout: float = 8.0) -> object | None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return None

    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-Command", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=timeout,
        )
    except Exception:
        return None

    output = completed.stdout.strip()
    if completed.returncode != 0 or not output:
        return None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def get_cim_video_controllers() -> list[dict[str, object]]:
    if os.name != "nt":
        return []

    data = _run_powershell_json(
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    if data is None:
        return []

    items = data if isinstance(data, list) else [data]
    return [item for item in items if isinstance(item, dict)]


def _driver_versions_by_gpu_name() -> dict[str, object]:
    result: dict[str, object] = {}
    for item in get_cim_video_controllers():
        name = str(item.get("Name") or "").strip().lower()
        if name and item.get("DriverVersion"):
            result[name] = item.get("DriverVersion")
    return result


def _matching_driver_version(name: str, driver_versions: dict[str, object]) -> object | None:
    normalized_name = name.strip().lower()
    for driver_name, version in driver_versions.items():
        if normalized_name == driver_name or normalized_name in driver_name or driver_name in normalized_name:
            return version
    return None


def get_nvidia_smi_vga_specs() -> list[str]:
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return []

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=5,
        )
    except Exception:
        return []

    if completed.returncode != 0:
        return []

    return [
        format_vga_spec(name, memory_bytes, driver_version)
        for name, memory_bytes, driver_version in parse_nvidia_smi_csv(completed.stdout)
    ]


def get_dxgi_vga_specs() -> list[str]:
    if os.name != "nt":
        return []

    try:
        import ctypes
        import uuid
        from ctypes import wintypes
    except Exception:
        return []

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DXGIAdapterDesc1(ctypes.Structure):
        _fields_ = [
            ("Description", wintypes.WCHAR * 128),
            ("VendorId", wintypes.UINT),
            ("DeviceId", wintypes.UINT),
            ("SubSysId", wintypes.UINT),
            ("Revision", wintypes.UINT),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", LUID),
            ("Flags", wintypes.UINT),
        ]

    def make_guid(value: str) -> GUID:
        parsed = uuid.UUID(value)
        return GUID(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:]),
        )

    factory = ctypes.c_void_p()
    iid_factory_1 = make_guid("770aae78-f26f-4dba-a829-253c83d1b387")
    try:
        hr = ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory_1), ctypes.byref(factory))
    except Exception:
        return []
    if hr != 0 or not factory.value:
        return []

    specs: list[str] = []
    driver_versions = _driver_versions_by_gpu_name()
    try:
        factory_vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        enum_adapters_1 = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(factory_vtable[12])

        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_adapters_1(factory, index, ctypes.byref(adapter))
            if (hr & 0xFFFFFFFF) == 0x887A0002:
                break
            if hr != 0 or not adapter.value:
                break

            adapter_vtable = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            try:
                get_desc_1 = ctypes.WINFUNCTYPE(
                    ctypes.c_long,
                    ctypes.c_void_p,
                    ctypes.POINTER(DXGIAdapterDesc1),
                )(adapter_vtable[10])
                desc = DXGIAdapterDesc1()
                desc_hr = get_desc_1(adapter, ctypes.byref(desc))
                if desc_hr == 0 and not (desc.Flags & 2):
                    specs.append(
                        format_vga_spec(
                            desc.Description,
                            int(desc.DedicatedVideoMemory),
                            _matching_driver_version(desc.Description, driver_versions),
                        )
                    )
            finally:
                release_adapter = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(adapter_vtable[2])
                release_adapter(adapter)

            index += 1
    finally:
        factory_vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        release_factory = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(factory_vtable[2])
        release_factory(factory)

    return specs


def get_cim_vga_specs() -> list[str]:
    specs: list[str] = []
    for item in get_cim_video_controllers():
        spec = format_vga_spec(
            item.get("Name"),
            item.get("AdapterRAM"),
            item.get("DriverVersion"),
        )
        if spec != "알 수 없음":
            specs.append(spec)
    return specs


def _gpu_key(spec: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", spec.split("(", 1)[0].lower())


def merge_vga_specs(primary: list[str], secondary: list[str]) -> list[str]:
    merged = list(primary)
    seen = {_gpu_key(spec) for spec in merged}
    for spec in secondary:
        key = _gpu_key(spec)
        if not key or key in seen:
            continue
        merged.append(spec)
        seen.add(key)
    return merged


def get_vga_specs() -> list[str]:
    nvidia_specs = get_nvidia_smi_vga_specs()
    dxgi_specs = get_dxgi_vga_specs()
    if nvidia_specs:
        return merge_vga_specs(nvidia_specs, dxgi_specs)
    if dxgi_specs:
        return dxgi_specs

    return get_cim_vga_specs()


def collect_system_specs() -> dict[str, str]:
    vga_specs = get_vga_specs()
    return {
        "cpu": get_cpu_name(),
        "ram": format_bytes_as_gib(get_total_ram_bytes()),
        "vga": "\n".join(vga_specs) if vga_specs else "알 수 없음",
    }


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


SETTINGS_FILE = PAPASTOP_DATA / "settings.json"


def load_user_settings() -> dict:
    """마지막에 저장한 사용자 선택(모델·번역·자원 방식 등)을 읽어온다. 없으면 빈 dict."""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_settings(data: dict) -> None:
    """사용자 선택을 settings.json에 저장한다. 실패해도 앱 종료를 막지 않는다."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
    except Exception:
        pass


def ensure_cache_dirs() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME / "home"))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def hf_file_cached(repo_id: str, filename: str) -> bool:
    """HF 허브 캐시에 해당 파일이 실제로 내려받아져 있으면 True(네트워크 접근 없음)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        ensure_cache_dirs()
        result = try_to_load_from_cache(repo_id, filename, cache_dir=str(HF_HOME / "hub"))
        return isinstance(result, str) and os.path.exists(result)
    except Exception:
        return False


def hf_repo_cache_key(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def whisper_model_repo(model_name: str) -> str:
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model_name, f"Systran/faster-whisper-{model_name}")
    except Exception:
        return f"Systran/faster-whisper-{model_name}"


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _remove_path(path: Path) -> int:
    removed_bytes = _path_size(path)
    if not path.exists():
        return 0

    def clear_readonly(func, target, _exc_info):
        try:
            os.chmod(target, 0o700)
        except OSError:
            pass
        func(target)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onerror=clear_readonly)
    else:
        path.unlink()
    return removed_bytes


def delete_hf_repo_files(repo_id: str) -> tuple[int, list[Path]]:
    ensure_cache_dirs()
    hub_root = HF_HOME / "hub"
    cache_key = hf_repo_cache_key(repo_id)
    candidates = (
        hub_root / cache_key,
        hub_root / ".locks" / cache_key,
    )
    removed_bytes = 0
    removed_paths: list[Path] = []
    for path in candidates:
        if path.exists() and _path_within(path, hub_root):
            removed_bytes += _remove_path(path)
            removed_paths.append(path)
    return removed_bytes, removed_paths


def delete_whisper_model_files(model_name: str) -> tuple[int, list[Path]]:
    return delete_hf_repo_files(whisper_model_repo(model_name))


def delete_llm_model_files(key: str) -> tuple[int, list[Path]]:
    spec = LLM_MODELS.get(key)
    if spec is None:
        return 0, []
    return delete_hf_repo_files(spec["repo"])


def delete_nllb_model_files(model_name: str) -> tuple[int, list[Path]]:
    removed_bytes = 0
    removed_paths: list[Path] = []
    ct2_path = NLLB_CT2_MODEL_DIRS.get(model_name)
    if ct2_path is not None and ct2_path.exists() and _path_within(ct2_path, PAPASTOP_DATA):
        removed_bytes += _remove_path(ct2_path)
        removed_paths.append(ct2_path)
    hf_bytes, hf_paths = delete_hf_repo_files(model_name)
    removed_bytes += hf_bytes
    removed_paths.extend(hf_paths)
    return removed_bytes, removed_paths


def delete_opus_model_files() -> tuple[int, list[Path]]:
    removed_bytes = 0
    removed_paths: list[Path] = []
    for repo_id in OPUS_MT_MODELS.values():
        hf_bytes, hf_paths = delete_hf_repo_files(repo_id)
        removed_bytes += hf_bytes
        removed_paths.extend(hf_paths)
    return removed_bytes, removed_paths


def _is_argos_package_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "argos-translate" in parts and "packages" in parts


def delete_argos_package_files() -> tuple[int, list[Path]]:
    if not has_module("argostranslate"):
        return 0, []
    removed_bytes = 0
    removed_paths: list[Path] = []
    try:
        import argostranslate.package as argostranslate_package

        packages = argostranslate_package.get_installed_packages()
    except Exception:
        return 0, []

    for package in packages:
        package_path = Path(getattr(package, "package_path", ""))
        if package_path.exists() and _is_argos_package_path(package_path):
            removed_bytes += _remove_path(package_path)
            removed_paths.append(package_path)

    try:
        import argostranslate.translate as argostranslate_translate

        argostranslate_translate.load_installed_languages()
    except Exception:
        pass
    return removed_bytes, removed_paths


# 모델 다운로드 진행률을 GUI로 흘려보내기 위한 전역 후크.
# faster-whisper·transformers·llama_cpp는 모두 huggingface_hub로 모델을 내려받고,
# huggingface_hub는 내부에서 tqdm 진행바를 쓴다. 그 tqdm을 가로채 진행/속도/남은시간을 전달한다.
_download_progress_callback: "Callable[[dict], None] | None" = None
_hf_tqdm_hooked = False
_last_download_emit = 0.0


def set_download_progress_callback(callback: "Callable[[dict], None] | None") -> None:
    global _download_progress_callback
    _download_progress_callback = callback


def _report_tqdm_progress(bar: object, closing: bool = False) -> None:
    global _last_download_emit
    callback = _download_progress_callback
    if callback is None:
        return
    fmt = getattr(bar, "format_dict", None) or {}
    total = fmt.get("total")
    unit = getattr(bar, "unit", "") or ""
    # 바이트 단위 다운로드만 진행률로 표시한다("Fetching N files" 같은 개수 카운터는 무시).
    if not total or "B" not in unit:
        return
    now = time.monotonic()
    # 이벤트 큐 폭주를 막기 위해 0.15초 간격으로만 보낸다(완료 시점은 항상 전달).
    if not closing and (now - _last_download_emit) < 0.15:
        return
    _last_download_emit = now
    n = fmt.get("n", 0)
    rate = fmt.get("rate")
    percent = min(100.0, n / total * 100.0) if total else 0.0
    eta_seconds = ((total - n) / rate) if (rate and rate > 0) else None
    try:
        callback(
            {
                "desc": getattr(bar, "desc", "") or "",
                "percent": percent,
                "n": n,
                "total": total,
                "eta_seconds": eta_seconds,
                "closing": closing,
            }
        )
    except Exception:
        pass


def _has_real_std_stream() -> bool:
    for stream in (sys.stderr, sys.stdout):
        if stream is not None and hasattr(stream, "write"):
            return True
    return False


def install_hf_download_hook() -> None:
    """huggingface_hub가 쓰는 tqdm의 init/update/close를 한 번만 감싸 진행 콜백을 붙인다."""
    global _hf_tqdm_hooked
    if _hf_tqdm_hooked:
        return
    try:
        from huggingface_hub.utils import tqdm as HfTqdm
    except Exception:
        return

    # 진행바가 꺼져 있으면 tqdm이 n을 갱신하지 않아 진행률을 받을 수 없으므로 켠다(표시는 GUI로 한다).
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
    except Exception:
        pass

    orig_init = HfTqdm.__init__
    orig_update = HfTqdm.update
    orig_close = HfTqdm.close

    def init(self, *args, **kwargs):
        # windowed(EXE)에서는 표준 출력 스트림이 없어 tqdm 출력이 예외를 던질 수 있다.
        # 진행은 GUI로 보여주므로, 스트림이 없을 때만 안전한 더미 버퍼로 출력을 돌린다.
        if kwargs.get("file") is None and not _has_real_std_stream():
            import io

            kwargs["file"] = io.StringIO()
        orig_init(self, *args, **kwargs)

    def update(self, n=1):
        result = orig_update(self, n)
        _report_tqdm_progress(self)
        return result

    def close(self):
        _report_tqdm_progress(self, closing=True)
        return orig_close(self)

    HfTqdm.__init__ = init
    HfTqdm.update = update
    HfTqdm.close = close
    _hf_tqdm_hooked = True


def format_hms(seconds: float) -> str:
    total = int(max(0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_file_size(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_srt_time(total_seconds: float) -> str:
    milliseconds = max(0, int(round(total_seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def clean_text_for_srt(text: str) -> str:
    fixed = re.sub(r"\s+", " ", text.strip())
    return fixed


_HANGUL_RE = re.compile(r"[가-힣]")
_SOURCE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[\.\)\]:：、-]\s*")
_TRANSLATION_LABEL_RE = re.compile(r"^(?:번역문?|한국어|korean|translation|target)\s*[:：]\s*", re.IGNORECASE)
_TRANSLATION_LABEL_ANY_RE = re.compile(r"(?:번역문?|한국어|korean|translation|target)\s*[:：]\s*", re.IGNORECASE)
_SOURCE_LABEL_RE = re.compile(
    r"^(?:원문|일본어|중국어|영어|source|original|japanese|chinese|english|日本語|中文)\s*[:：]\s*",
    re.IGNORECASE,
)


def clean_translation_for_subtitle(translated: str, original: str) -> str:
    """번역 모델 답변에서 원문·라벨을 걷어내고 SRT에 넣을 한국어 자막만 남긴다."""
    raw = str(translated or "").replace("\r\n", "\n").replace("\r", "\n").replace("```", "")
    original_clean = clean_text_for_srt(str(original or ""))
    pieces: list[str] = []

    for raw_line in raw.split("\n"):
        line = raw_line.strip().strip("\"'“”‘’")
        if not line:
            continue
        line = _NUMBER_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        if _SOURCE_LABEL_RE.match(line):
            parts = _TRANSLATION_LABEL_ANY_RE.split(_SOURCE_LABEL_RE.sub("", line, count=1), maxsplit=1)
            if len(parts) != 2:
                continue
            line = parts[1].strip()
        line = _TRANSLATION_LABEL_RE.sub("", line, count=1).strip()
        if not line:
            continue

        if re.search(r"\s(?:=>|->|→|=)\s", line):
            arrow_parts = [part.strip() for part in re.split(r"\s*(?:=>|->|→|=)\s*", line) if part.strip()]
            korean_parts = [part for part in arrow_parts if _HANGUL_RE.search(part)]
            if korean_parts:
                line = korean_parts[-1]

        line_clean = clean_text_for_srt(line)
        if original_clean and line_clean == original_clean and _SOURCE_SCRIPT_RE.search(original_clean):
            continue
        if original_clean and original_clean in line and _HANGUL_RE.search(line):
            line = line.replace(original_clean, " ").strip()
        if line:
            pieces.append(line)

    if not pieces:
        return ""
    if any(_HANGUL_RE.search(piece) for piece in pieces):
        pieces = [piece for piece in pieces if _HANGUL_RE.search(piece)]
    cleaned = clean_text_for_srt(" ".join(pieces))
    if _SOURCE_SCRIPT_RE.search(cleaned) and not _HANGUL_RE.search(cleaned):
        return ""
    return cleaned


# 실시간 자막 후처리용 패턴. 위스퍼·번역기 출력에 흔한 문장부호 앞 공백, 괄호/따옴표
# 주변 공백, 과도한 반복 부호, 문두의 홀로 남은 부호를 정리한다.
_TIDY_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:…、。！？])")
_TIDY_OPEN_WRAP_RE = re.compile(r"([(\[\"'“‘])\s+")
_TIDY_CLOSE_WRAP_RE = re.compile(r"\s+([)\]\"'”’])")
_TIDY_REPEAT_MARK_RE = re.compile(r"([!?！？])\1{2,}")
_TIDY_LONG_DOTS_RE = re.compile(r"\.{4,}")
_TIDY_LEADING_PUNCT_RE = re.compile(r"^[\s,.;:、。·]+")


def tidy_korean_subtitle(text: str) -> str:
    """실시간 한국어 자막의 문장부호·띄어쓰기를 가볍게 정리한다(뜻은 건드리지 않음)."""
    cleaned = clean_text_for_srt(text)  # 연속 공백 -> 하나
    if not cleaned:
        return ""
    # "안녕 ." -> "안녕.", "정말 ?" -> "정말?" 처럼 부호 앞의 불필요한 공백 제거.
    cleaned = _TIDY_SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    # 여는 괄호·따옴표 뒤, 닫는 괄호·따옴표 앞의 공백 정리.
    cleaned = _TIDY_OPEN_WRAP_RE.sub(r"\1", cleaned)
    cleaned = _TIDY_CLOSE_WRAP_RE.sub(r"\1", cleaned)
    # 같은 종결부호가 3개 이상 반복되면 하나로 줄인다("!!"는 강조로 남겨 둔다).
    cleaned = _TIDY_REPEAT_MARK_RE.sub(r"\1", cleaned)
    # 마침표 4개 이상은 말줄임표로 정리(".."·"..."는 그대로 둔다).
    cleaned = _TIDY_LONG_DOTS_RE.sub("…", cleaned)
    # 문장 맨 앞에 홀로 남은 쉼표·마침표 등을 제거.
    cleaned = _TIDY_LEADING_PUNCT_RE.sub("", cleaned)
    return cleaned.strip()


def default_recording_path(output_dir: Path | None = None) -> Path:
    folder = output_dir if output_dir is not None else PAPASTOP_DATA / "recordings"
    return folder / f"screen-{time.strftime('%Y%m%d-%H%M%S')}.mkv"


def default_live_srt_path(output_dir: Path | None = None) -> Path:
    folder = output_dir if output_dir is not None else PAPASTOP_DATA / "subtitles"
    return folder / f"live-{time.strftime('%Y%m%d-%H%M%S')}.ko.srt"


def enumerate_monitors() -> list[dict]:
    """가상 데스크톱 내 각 모니터의 위치·크기를 반환한다(Windows 전용, 실패 시 빈 리스트).

    좌표는 gdigrab의 -offset_x/-offset_y와 같은 가상 데스크톱 절대 좌표(보조 모니터는 음수 가능).
    """
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        results: list[dict] = []

        def _callback(hmon, hdc, lprc, lparam):
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                rect = info.rcMonitor
                results.append(
                    {
                        "x": int(rect.left),
                        "y": int(rect.top),
                        "w": int(rect.right - rect.left),
                        "h": int(rect.bottom - rect.top),
                        "primary": bool(info.dwFlags & 1),  # MONITORINFOF_PRIMARY
                    }
                )
            return 1

        proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
        )
        user32.EnumDisplayMonitors(0, 0, proc_type(_callback), 0)
        # 주 모니터를 맨 앞에, 나머지는 좌->우·상->하 순서로 정렬한다.
        results.sort(key=lambda m: (not m["primary"], m["x"], m["y"]))
        return results
    except Exception:
        return []


def build_screen_capture_command(
    ffmpeg: str,
    output_path: Path,
    fps: int,
    region: tuple[int, int, int, int] | None = None,
    codec: str = DEFAULT_RECORD_CODEC,
) -> list[str]:
    command = [
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
    ]
    if region is not None:
        x, y, w, h = region
        # H.264(yuv420p)는 가로·세로가 짝수여야 하므로 영역도 짝수로 맞춘다.
        w -= w % 2
        h -= h % 2
        command += ["-offset_x", str(int(x)), "-offset_y", str(int(y)), "-video_size", f"{int(w)}x{int(h)}"]
    command += ["-i", "desktop", "-an"]
    if codec == "h264":
        # crop으로 홀수 해상도를 짝수로 보정(전체 화면 캡처 시 대비).
        command += [
            "-vf",
            "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        command += ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgra"]
    command += [str(output_path)]
    return command


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


# 위스퍼가 무음 구간에서 통째로 지어내는 대표적 인사·구독 유도 문구(유튜브 자막 학습 흔적).
# 세그먼트 전체가 이 문구와 정확히 일치할 때만 걸러내 실제 문장 오탐을 최소화한다.
_HALLUCINATION_FILLER_PHRASES = frozenset({
    # 한국어
    "감사합니다", "고맙습니다", "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다",
    "구독과 좋아요", "구독 좋아요 알림설정", "구독과 좋아요 부탁드립니다", "다음 영상에서 만나요",
    # 영어
    "thank you", "thank you very much", "thanks for watching", "thank you for watching",
    "please subscribe", "subscribe to my channel", "see you next time",
    # 일본어
    "ありがとうございました", "ご視聴ありがとうございました",
    # 중국어
    "谢谢观看", "谢谢大家", "请订阅", "字幕",
})
# 양끝 문장부호·따옴표·괄호와 공백을 걷어내고 소문자화해 필러 문구와 비교하기 위한 패턴.
_FILLER_TRIM_RE = re.compile(r"^[\s\"'“”‘’()\[\].!?…,、。！？·\-~]+|[\s\"'“”‘’()\[\].!?…,、。！？·\-~]+$")


def _normalize_for_filler_match(text: str) -> str:
    cleaned = _FILLER_TRIM_RE.sub("", text.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def is_probable_hallucination(
    text: str,
    no_speech_prob: float,
    avg_logprob: float,
    compression_ratio: float,
) -> bool:
    """무음·잡음 구간에서 위스퍼가 지어내는 '환각' 자막인지 추정한다(실시간 전용 필터).

    실시간은 침묵·박수·배경음까지 발화로 잘려 들어와, 위스퍼가 "감사합니다", "Thank you"
    같은 문구나 같은 단어 반복을 지어내 이상한 번역의 주원인이 된다. 실제 발화를 지우지
    않도록 신호가 뚜렷할 때만 True를 돌려주는 보수적 판단이다.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    # 무음 확률이 매우 높으면(위스퍼가 소리가 없다고 강하게 판단) 지어낸 자막으로 본다.
    if no_speech_prob >= 0.85:
        return True
    # 같은 토큰이 반복되는 환각은 압축률이 비정상적으로 높다(정상 문장은 대개 2.4 미만).
    if compression_ratio >= 2.6 and avg_logprob <= -0.4:
        return True
    # 세그먼트 전체가 알려진 환각 문구와 정확히 일치하면 걸러낸다(부분 일치는 건드리지 않음).
    if _normalize_for_filler_match(stripped) in _HALLUCINATION_FILLER_PHRASES:
        return True
    return False


class ASREngine:
    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, resource_mode: str = DEFAULT_RESOURCE_MODE):
        self.model_name = model_name
        self.resource_mode = normalize_resource_mode(resource_mode)
        self.source_language = SOURCE_LANG_AUTO
        self._model = None
        self._batched_model = None
        self._load_lock = threading.Lock()
        self._resource_summary = build_resource_settings(self.resource_mode).summary

    def set_source_language(self, code: str | None) -> None:
        self.source_language = code if code in SOURCE_LANG_LABELS else SOURCE_LANG_AUTO

    def set_model(self, model_name: str) -> None:
        chosen = model_name if model_name in WHISPER_MODEL_LABELS else DEFAULT_WHISPER_MODEL
        with self._load_lock:
            if chosen == self.model_name:
                return
            self.model_name = chosen
            self._model = None
            self._batched_model = None

    def set_resource_mode(self, mode: str | None) -> None:
        chosen = normalize_resource_mode(mode)
        with self._load_lock:
            if chosen == self.resource_mode:
                return
            self.resource_mode = chosen
            self._model = None
            self._batched_model = None
            self._resource_summary = build_resource_settings(self.resource_mode).summary

    def prepare_model(self, model_name: str | None = None) -> None:
        if model_name:
            self.set_model(model_name)
        self._load_model()

    def unload(self) -> None:
        """모델을 메모리에서 내려 VRAM을 비운다(대형 LLM 번역 직전에 호출).
        다음 사용 시 _load_model()이 자동으로 다시 올린다."""
        with self._load_lock:
            self._model = None
            self._batched_model = None
        import gc

        gc.collect()

    def model_label(self, model_name: str | None = None) -> str:
        target = model_name or self.model_name
        return WHISPER_MODEL_LABELS.get(target, target)

    def is_model_ready(self, model_name: str | None = None) -> bool:
        """선택한 음성 인식 모델이 이미 내려받아져 바로 쓸 수 있으면 True."""
        size = model_name or self.model_name
        return hf_file_cached(whisper_model_repo(size), "model.bin")

    def resource_summary(self) -> str:
        return self._resource_summary

    def _load_model(self, on_line: Callable[[str], None] | None = None):
        if self._model is not None:
            if on_line:
                on_line(f"위스퍼 모델 사용 중: {self.resource_summary()}")
            return

        if not has_module("faster_whisper"):
            raise RuntimeError("faster-whisper가 설치되어 있지 않습니다.")

        with self._load_lock:
            if self._model is not None:
                return

            from faster_whisper import WhisperModel

            ensure_cache_dirs()
            configure_cuda_dll_paths()
            settings = build_resource_settings(self.resource_mode)
            configure_runtime_threads(settings.cpu_threads)
            gpu_memory_before = get_nvidia_memory_used_mib() if settings.asr_device == "cuda" else None
            if on_line:
                on_line(
                    f"위스퍼 모델을 {settings.asr_device.upper()}에 올리는 중입니다 "
                    f"({settings.asr_compute_type}, 배치 {settings.asr_batch_size}) ..."
                )

            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=settings.asr_device,
                    device_index=0,
                    compute_type=settings.asr_compute_type,
                    cpu_threads=settings.cpu_threads,
                    num_workers=settings.num_workers,
                    download_root=str(HF_HOME / "hub"),
                )
                if settings.asr_device == "cuda":
                    from faster_whisper import BatchedInferencePipeline

                    self._batched_model = BatchedInferencePipeline(self._model)
                else:
                    self._batched_model = None
                self._resource_summary = settings.summary
                if on_line:
                    gpu_memory_after = get_nvidia_memory_used_mib() if settings.asr_device == "cuda" else None
                    if gpu_memory_before is not None and gpu_memory_after is not None:
                        on_line(
                            "GPU 메모리 변화: "
                            f"{gpu_memory_before}MiB -> {gpu_memory_after}MiB "
                            f"(+{gpu_memory_after - gpu_memory_before}MiB)"
                        )
                    on_line(f"위스퍼 모델 로딩 완료: {self.resource_summary()}")
            except Exception as exc:
                if settings.asr_device != "cuda":
                    raise
                if settings.mode == "max":
                    raise RuntimeError(
                        "GPU 위스퍼 모델을 시작하지 못했습니다. 최대 성능 모드에서는 CPU로 몰래 전환하지 않습니다. "
                        f"CUDA 드라이버, ctranslate2, 모델 파일 상태를 확인해 주세요. 원인: {exc}"
                    )
                cpu_threads = settings.cpu_threads
                self._model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                    num_workers=settings.num_workers,
                    download_root=str(HF_HOME / "hub"),
                )
                self._batched_model = None
                self._resource_summary = (
                    f"GPU 시작 실패로 CPU 전환 | CPU {cpu_threads}/{settings.cpu_count}스레드 | "
                    "자막 CPU int8"
                )
                if on_line:
                    on_line(self._resource_summary)

    def transcribe(
        self,
        audio,
        on_line=None,
        should_stop: Callable[[], bool] | None = None,
        on_segment: Callable[[SubtitleSegment], None] | None = None,
        vad_filter: bool = True,
        beam_size: int | None = None,
        word_timestamps: bool = False,
        language_override: str | None = None,
        drop_hallucinations: bool = False,
    ) -> tuple[list[SubtitleSegment], str]:
        self._load_model(on_line=on_line)
        assert self._model is not None

        if should_stop is not None and should_stop():
            raise OperationCancelled("취소되었습니다.")

        settings = build_resource_settings(self.resource_mode)
        # language_override가 주어지면(실시간 sticky 고정) 그 언어를 우선 강제하고,
        # 없으면 사용자가 고른 소스 언어(자동이면 감지)를 따른다.
        if language_override is not None:
            forced_language = language_override
        else:
            forced_language = None if self.source_language == SOURCE_LANG_AUTO else self.source_language
        # 파일 경로면 문자열로, 실시간 numpy 파형이면 그대로 넘긴다(faster-whisper가 둘 다 지원).
        audio_input = str(audio) if isinstance(audio, (str, Path)) else audio
        transcribe_kwargs = {
            "language": forced_language,
            "beam_size": beam_size if beam_size is not None else settings.asr_beam_size,
            "vad_filter": vad_filter,
            "condition_on_previous_text": False,
            "word_timestamps": word_timestamps,
        }
        if self._batched_model is not None:
            transcribe_kwargs["batch_size"] = settings.asr_batch_size
            transcribe_kwargs["without_timestamps"] = False
            segments, info = self._batched_model.transcribe(audio_input, **transcribe_kwargs)
        else:
            segments, info = self._model.transcribe(audio_input, **transcribe_kwargs)

        detected_language = getattr(info, "language", "auto")
        rows: list[SubtitleSegment] = []
        for segment in segments:
            if should_stop is not None and should_stop():
                raise OperationCancelled("취소되었습니다.")

            text = (segment.text or "").strip()
            if not text:
                continue
            # 실시간 전용: 무음·잡음 구간에서 지어낸 환각 자막을 걸러 이상한 번역을 막는다.
            # 속성이 없는(테스트용 가짜) 세그먼트는 기본값 덕분에 걸러지지 않는다.
            if drop_hallucinations and is_probable_hallucination(
                text,
                float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                float(getattr(segment, "compression_ratio", 0.0) or 0.0),
            ):
                continue
            # 단어 타임스탬프가 있으면 실제 첫/마지막 단어 시각으로 자막 구간을 좁혀 싱크를 맞춘다.
            words = getattr(segment, "words", None) if word_timestamps else None
            if words and words[0].start is not None and words[-1].end is not None:
                start = float(words[0].start)
                end = float(words[-1].end)
            else:
                start = float(segment.start)
                end = float(segment.end)
            subtitle_segment = SubtitleSegment(
                start=start,
                end=end,
                text=text,
            )
            rows.append(subtitle_segment)
            if on_segment is not None:
                on_segment(subtitle_segment)

        if on_line:
            on_line(f"자원 사용: {self.resource_summary()}")
            on_line(f"감지된 언어: {detected_language}, 자막 줄 수={len(rows)}")

        return rows, detected_language


class TranslationEngine:
    def __init__(self, preferred: str = "auto", resource_mode: str = DEFAULT_RESOURCE_MODE, realtime: bool = False):
        self.preferred = preferred
        self.resource_mode = normalize_resource_mode(resource_mode)
        # 실시간 인스턴스는 지연 예산이 빠듯해, 체감 품질 손실이 거의 없는 선에서 NLLB beam을 낮춘다.
        self.realtime = realtime
        self._nllb_pipelines: dict[str, object] = {}
        self._nllb_ct2_translators: dict[str, object] = {}
        self._nllb_tokenizer = None
        self._llama_models: dict[str, object] = {}
        self._active_llm_no_think = False
        self._opus_pipelines: dict[str, object] = {}
        # OPUS 2단계(영어 경유) 설정: 2차 번역기와 사용 여부.
        self.opus_secondary = "nllb600"
        self.opus_use_secondary = True
        self._pipeline_lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._resource_summary = build_resource_settings(self.resource_mode).summary

    def set_opus_secondary(self, key: str | None) -> None:
        if key in LIVE_SECONDARY_STEPS:
            self.opus_secondary = key

    def set_opus_use_secondary(self, enabled: bool) -> None:
        self.opus_use_secondary = bool(enabled)

    def set_profile(self, profile: str) -> None:
        if profile not in TRANSLATION_PROFILE_ORDER_STEPS:
            profile = "auto"
        self.preferred = profile

    def set_resource_mode(self, mode: str | None) -> None:
        chosen = normalize_resource_mode(mode)
        with self._pipeline_lock:
            if chosen == self.resource_mode:
                return
            self.resource_mode = chosen
            self._nllb_pipelines.clear()
            self._nllb_ct2_translators.clear()
            self._resource_summary = build_resource_settings(self.resource_mode).summary

    def resource_summary(self) -> str:
        return self._resource_summary

    def _profile_steps(self, profile: str | None = None) -> tuple[tuple[str, str | None], ...]:
        selected = profile or self.preferred
        return TRANSLATION_PROFILE_ORDER_STEPS.get(selected, TRANSLATION_PROFILE_ORDER_STEPS["auto"])

    def needs_exclusive_vram(self, profile: str | None = None) -> bool:
        """현재 프로필의 주 모델이 VRAM을 독점해야 하는 대형 LLM이면 True.
        파일 자막에서 번역 직전에 음성 인식 모델을 내려 OOM을 막는 데 쓴다."""
        steps = self._profile_steps(profile)
        if not steps:
            return False
        engine, model = steps[0]
        if engine != "llama" or model is None:
            return False
        return bool((LLM_MODELS.get(model) or {}).get("exclusive_vram"))

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
                elif engine == "llama":
                    assert model_name is not None
                    self._load_llama(model_name)
                    break  # LLM 주 모델만 미리 받고, 보조 NLLB는 실패 시 그때 내려받는다.
                elif engine == "opus":
                    self._prepare_opus()
                    break
                else:
                    raise RuntimeError(f"알 수 없는 번역 방식입니다: {engine}")

    def _nllb_ct2_path(self, model_name: str) -> Path | None:
        path = NLLB_CT2_MODEL_DIRS.get(model_name)
        if path is None:
            return None
        required = ("config.json", "model.bin", "shared_vocabulary.json")
        return path if all((path / name).exists() for name in required) else None

    def _load_nllb_tokenizer(self):
        if self._nllb_tokenizer is not None:
            return self._nllb_tokenizer

        if not has_module("transformers"):
            raise RuntimeError("transformers가 설치되어 있지 않습니다.")

        from transformers import AutoTokenizer

        ensure_cache_dirs()
        self._nllb_tokenizer = AutoTokenizer.from_pretrained(
            NLLB_TOKENIZER_MODEL,
            cache_dir=str(HF_HOME / "hub"),
        )
        return self._nllb_tokenizer

    def _load_nllb_ct2(self, model_name: str):
        model_path = self._nllb_ct2_path(model_name)
        if model_path is None:
            return None

        cached = self._nllb_ct2_translators.get(model_name)
        if cached is not None:
            return cached

        configure_cuda_dll_paths()
        if not has_module("ctranslate2"):
            return None

        import ctranslate2

        settings = build_resource_settings(self.resource_mode)
        configure_runtime_threads(settings.cpu_threads)
        translator = ctranslate2.Translator(
            str(model_path),
            device=settings.translation_ct2_device,
            compute_type=settings.translation_ct2_compute_type,
            intra_threads=settings.cpu_threads,
            inter_threads=settings.num_workers,
        )
        self._nllb_ct2_translators[model_name] = translator
        self._resource_summary = settings.summary
        return translator

    def _load_nllb_pipeline(self, model_name: str):
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
            settings = build_resource_settings(self.resource_mode)
            configure_runtime_threads(settings.cpu_threads)
            pipeline_kwargs = {
                "model": model_name,
                "device": settings.translation_device,
            }
            if settings.translation_dtype == "float16":
                pipeline_kwargs["torch_dtype"] = torch.float16
            try:
                pipeline_obj = pipeline("translation", **pipeline_kwargs)
            except TypeError:
                pipeline_kwargs.pop("torch_dtype", None)
                pipeline_obj = pipeline("translation", **pipeline_kwargs)
            self._nllb_pipelines[model_name] = pipeline_obj
            self._resource_summary = settings.summary
            return pipeline_obj

    def _load_nllb(self, model_name: str) -> tuple[str, object]:
        ct2_translator = self._load_nllb_ct2(model_name)
        if ct2_translator is not None:
            self._load_nllb_tokenizer()
            return "ct2", ct2_translator
        return "pipeline", self._load_nllb_pipeline(model_name)

    def _translate_nllb(self, texts: list[str], source_code: str, model_name: str) -> list[str]:
        if not texts:
            return []
        backend, loaded = self._load_nllb(model_name)

        if backend == "ct2":
            tokenizer = self._load_nllb_tokenizer()
            tokenizer.src_lang = nllb_lang(source_code)
            source_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts]
            settings = build_resource_settings(self.resource_mode)
            results = loaded.translate_batch(  # type: ignore[attr-defined]
                source_tokens,
                target_prefix=[["kor_Hang"] for _ in source_tokens],
                beam_size=(3 if self.realtime else (5 if self.preferred == "accurate" else 4)),
                max_decoding_length=300,
                max_batch_size=settings.translation_batch_size,
            )

            translated: list[str] = []
            for item in results:
                tokens = list(item.hypotheses[0]) if item.hypotheses else []
                if tokens and tokens[0] == "kor_Hang":
                    tokens = tokens[1:]
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                translated.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
            return translated

        pipeline = loaded

        results = pipeline(
            texts,
            src_lang=nllb_lang(source_code),
            tgt_lang="kor_Hang",
            max_length=300,
            batch_size=build_resource_settings(self.resource_mode).translation_batch_size,
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

    # ── llama.cpp(GGUF) LLM 번역 ─────────────────────────────────
    def _llama_model_path(self, key: str, download: bool = False) -> str:
        spec = LLM_MODELS.get(key)
        if spec is None:
            raise RuntimeError(f"알 수 없는 LLM 모델입니다: {key}")
        ensure_cache_dirs()
        from huggingface_hub import hf_hub_download

        kwargs = {
            "repo_id": spec["repo"],
            "filename": spec["file"],
            "cache_dir": str(HF_HOME / "hub"),
        }
        if not download:
            kwargs["local_files_only"] = True
        return hf_hub_download(**kwargs)

    def _load_llama(self, key: str):
        spec = LLM_MODELS.get(key) or {}
        # 추론(thinking) 모델이면 채팅 시 사고 과정을 끄는 지시를 덧붙인다.
        self._active_llm_no_think = bool(spec.get("no_think"))
        cached = self._llama_models.get(key)
        if cached is not None:
            return cached

        with self._pipeline_lock:
            cached = self._llama_models.get(key)
            if cached is not None:
                return cached

            if not has_module("llama_cpp"):
                raise RuntimeError("llama-cpp-python이 설치되어 있지 않습니다.")

            # llama.cpp의 ggml-cuda.dll이 CUDA 런타임 DLL을 찾으려면 import 전에 PATH 등록이 필요하다.
            configure_cuda_dll_paths()
            try:
                model_path = self._llama_model_path(key, download=False)
            except Exception:
                model_path = self._llama_model_path(key, download=True)

            # 한 번에 하나의 LLM만 VRAM에 유지한다(16GB에 여러 개 동시 적재 방지).
            for other in list(self._llama_models):
                if other != key:
                    self._llama_models.pop(other, None)

            from llama_cpp import Llama

            settings = build_resource_settings(self.resource_mode)
            # 기본은 전 레이어 GPU(-1). 16GB에 다 안 올라가는 대형 모델은
            # spec의 n_gpu_layers만큼만 GPU에 올리고 나머지는 CPU+RAM이 맡는다.
            if get_ctranslate2_cuda_available():
                n_gpu = int(spec.get("n_gpu_layers", -1))
            else:
                n_gpu = 0
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu,
                n_ctx=4096,
                n_threads=settings.cpu_threads,
                verbose=False,
            )
            self._llama_models[key] = llm
            self._resource_summary = settings.summary
            return llm

    def _llm_system_message(self, source_code: str) -> str:
        src = (source_code or "").split("-")[0].lower()
        lang = LLM_LANG_NAMES.get(src, "the source")
        return (
            f"You are a professional {lang}-to-Korean subtitle translator. "
            "Translate meaning and tone naturally for Korean viewers. "
            "Use concise Korean subtitle style. "
            "Never include the source text, labels, romanization, explanations, markdown, or notes."
        )

    def _llm_max_tokens(self, texts: list[str]) -> int:
        budget = sum(max(24, len(text) * 3) for text in texts) + 160
        return min(2048, max(256, budget))

    def _llama_chat(self, llm, messages: list[dict[str, str]], max_tokens: int) -> str:
        if self._active_llm_no_think and messages:
            # Qwen3 등은 마지막 사용자 메시지에 "/no_think"를 붙이면 사고 과정을 건너뛴다.
            messages = [dict(message) for message in messages]
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["content"] = f"{message['content']} /no_think"
                    break
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
        )
        try:
            content = str(out["choices"][0]["message"]["content"])
        except Exception as exc:
            raise RuntimeError("LLM 번역 결과를 읽지 못했습니다.") from exc
        # 추론(thinking) 모델이 남기는 <think>...</think> 사고 과정은 자막에서 제거한다.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if "</think>" in content:
            content = content.split("</think>")[-1]
        return content.strip()

    def _parse_llm_numbered_translations(self, raw: str, originals: list[str]) -> list[str] | None:
        expected = len(originals)
        if expected <= 0:
            return []

        slots: dict[int, str] = {}
        current_index: int | None = None
        current_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_index, current_lines
            if current_index is not None:
                slots[current_index] = "\n".join(current_lines).strip()
            current_index = None
            current_lines = []

        for raw_line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(\d{1,3})\s*[\.\)\]:：、-]\s*(.*)$", line)
            if match:
                number = int(match.group(1))
                if 1 <= number <= expected:
                    flush_current()
                    current_index = number
                    current_lines = [match.group(2).strip()]
                    continue
            if current_index is not None:
                current_lines.append(line)
        flush_current()

        if all(index in slots for index in range(1, expected + 1)):
            return [
                clean_translation_for_subtitle(slots[index], originals[index - 1])
                for index in range(1, expected + 1)
            ]

        plain_lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(plain_lines) == expected:
            return [
                clean_translation_for_subtitle(line, original)
                for line, original in zip(plain_lines, originals)
            ]
        return None

    def _translate_llama_one(self, llm, text: str, source_code: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        messages = [
            {"role": "system", "content": self._llm_system_message(source_code)},
            {
                "role": "user",
                "content": (
                    "Translate this subtitle line into Korean.\n"
                    "Return only the Korean subtitle text. Do not copy the source text.\n\n"
                    f"Source:\n{stripped}"
                ),
            },
        ]
        raw = self._llama_chat(llm, messages, self._llm_max_tokens([stripped]))
        return clean_translation_for_subtitle(raw, stripped)

    def _translate_llama_batch(self, llm, texts: list[str], source_code: str) -> list[str]:
        if not texts:
            return []
        src = (source_code or "").split("-")[0].lower()
        lang = LLM_LANG_NAMES.get(src, "the source")
        numbered_lines = "\n".join(f"{index}. {text.strip()}" for index, text in enumerate(texts, start=1))
        messages = [
            {"role": "system", "content": self._llm_system_message(source_code)},
            {
                "role": "user",
                "content": (
                    f"Translate these {lang} subtitle lines into Korean.\n"
                    "Use surrounding lines as context, but keep one Korean output line for each input line.\n"
                    f"Return exactly {len(texts)} lines. Each line must start with the same number and a period.\n"
                    "Do not copy any source-language text.\n\n"
                    f"{numbered_lines}"
                ),
            },
        ]
        raw = self._llama_chat(llm, messages, self._llm_max_tokens(texts))
        parsed = self._parse_llm_numbered_translations(raw, texts)
        if parsed is None:
            return [self._translate_llama_one(llm, text, source_code) for text in texts]
        return [
            parsed_text or self._translate_llama_one(llm, original, source_code)
            for parsed_text, original in zip(parsed, texts)
        ]

    def _translate_llama(self, texts: list[str], source_code: str, key: str) -> list[str]:
        if not texts:
            return []
        llm = self._load_llama(key)
        settings = build_resource_settings(self.resource_mode)
        batch_size = max(1, min(LLM_TRANSLATION_CHUNK_SIZE, settings.translation_batch_size))

        results: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunk_results = ["" for _ in chunk]
            nonempty_positions: list[int] = []
            nonempty_texts: list[str] = []
            for index, text in enumerate(chunk):
                stripped = text.strip()
                if stripped:
                    nonempty_positions.append(index)
                    nonempty_texts.append(stripped)
            if not nonempty_texts:
                results.extend(chunk_results)
                continue
            translated = self._translate_llama_batch(llm, nonempty_texts, source_code)
            for position, translated_text in zip(nonempty_positions, translated):
                chunk_results[position] = translated_text
            results.extend(chunk_results)
        return results

    # ── OPUS-MT 2단계(원본→영어→한국어) ─────────────────────────
    def _load_opus(self, source: str):
        # 이 환경의 transformers에는 "translation" 파이프라인이 없어 MarianMT 모델을 직접 로드한다.
        model_name = OPUS_MT_MODELS.get(source)
        if model_name is None:
            return None
        cached = self._opus_pipelines.get(source)
        if cached is not None:
            return cached
        if not has_module("transformers"):
            raise RuntimeError("transformers가 설치되어 있지 않습니다.")

        with self._pipeline_lock:
            cached = self._opus_pipelines.get(source)
            if cached is not None:
                return cached
            from transformers import MarianMTModel, MarianTokenizer

            ensure_cache_dirs()
            cache = str(HF_HOME / "hub")
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache)
            model = MarianMTModel.from_pretrained(model_name, cache_dir=cache)
            model.eval()
            self._opus_pipelines[source] = (tokenizer, model)
            return self._opus_pipelines[source]

    def _prepare_opus(self) -> None:
        for src in OPUS_MT_MODELS:
            self._load_opus(src)
        if self.opus_use_secondary:
            engine, model = LIVE_SECONDARY_STEPS[self.opus_secondary]
            if engine == "nllb":
                assert model is not None
                self._load_nllb(model)

    def _opus_to_english(self, texts: list[str], source: str) -> list[str]:
        import torch

        tokenizer, model = self._load_opus(source)
        settings = build_resource_settings(self.resource_mode)
        batch_size = max(1, settings.translation_batch_size)
        english: list[str] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=400)
                generated = model.generate(**encoded, max_length=400, num_beams=4)
                english.extend(t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True))
        return english

    def _translate_opus(self, texts: list[str], source_code: str) -> list[str]:
        if not texts:
            return []
        src = (source_code or "en").split("-")[0].lower()

        # 1차: 원본 → 영어 (영어 입력이면 그대로, OPUS 미지원 언어는 NLLB로 바로 한국어 번역)
        if src == "en":
            english = [t.strip() for t in texts]
        elif src in OPUS_MT_MODELS:
            english = self._opus_to_english(texts, src)
        else:
            return self._translate_nllb(texts, source_code, NLLB_LIVE_MODEL)

        # 2차: 영어 → 한국어 (체크 해제 시 영어 자막을 그대로 표시)
        if not self.opus_use_secondary:
            return english
        engine, model = LIVE_SECONDARY_STEPS[self.opus_secondary]
        if engine == "argos":
            return self._translate_argos(english, "en")
        assert model is not None
        return self._translate_nllb(english, "en", model)

    def translate_texts(
        self, texts: list[str], source_code: str, allow_original_fallback: bool = True
    ) -> list[str]:
        if not texts:
            return []

        for engine, model_name in self._profile_steps():
            try:
                if engine == "nllb":
                    assert model_name is not None
                    return self._translate_nllb(texts, source_code, model_name)
                if engine == "argos":
                    return self._translate_argos(texts, source_code)
                if engine == "llama":
                    assert model_name is not None
                    return self._translate_llama(texts, source_code, model_name)
                if engine == "opus":
                    return self._translate_opus(texts, source_code)
                raise RuntimeError(f"알 수 없는 번역 방식입니다: {engine}")
            except Exception:
                if self.preferred == "fast":
                    raise
                continue

        # 모든 엔진이 실패한 경우. 파일 자막은 원문이라도 남기지만(allow_original_fallback),
        # 실시간은 원문(외국어)이 자막으로 새는 것을 막으려 예외를 던져 해당 구간을 건너뛴다.
        if allow_original_fallback:
            return list(texts)
        raise RuntimeError("번역 엔진을 사용할 수 없어 이 구간을 건너뜁니다.")

    def profile_label(self, profile: str | None = None) -> str:
        target = profile or self.preferred
        return TRANSLATION_PROFILE_LABELS.get(target, target)

    # ── 모델 다운로드 상태 확인 ──────────────────────────────────
    def is_profile_ready(self, profile: str | None = None) -> bool:
        """해당 번역 프로필의 주 엔진/모델이 이미 내려받아져 바로 쓸 수 있으면 True."""
        steps = self._profile_steps(profile)
        if not steps:
            return False
        engine, model = steps[0]
        return self._engine_ready(engine, model)

    def _engine_ready(self, engine: str, model: str | None) -> bool:
        if engine == "argos":
            return self._argos_ready()
        if engine == "nllb":
            return model is not None and self._nllb_ready(model)
        if engine == "llama":
            return model is not None and self._llama_ready(model)
        if engine == "opus":
            return self._opus_ready()
        return False

    def _argos_ready(self) -> bool:
        if not has_module("argostranslate"):
            return False
        try:
            import argostranslate.translate as argostranslate_translate

            codes = {lang.code for lang in argostranslate_translate.get_installed_languages()}
            return "ko" in codes and len(codes) >= 2
        except Exception:
            return False

    def _nllb_ready(self, model_name: str) -> bool:
        # 사전 변환된 CT2 모델이 있으면 바로 사용 가능. 없으면 transformers 캐시 여부로 판단.
        if self._nllb_ct2_path(model_name) is not None:
            return True
        return hf_file_cached(model_name, "config.json")

    def _llama_ready(self, key: str) -> bool:
        spec = LLM_MODELS.get(key)
        if spec is None:
            return False
        return hf_file_cached(spec["repo"], spec["file"])

    def _opus_ready(self) -> bool:
        # ja/zh OPUS 1차 모델이 모두 받아져 있어야 사용 가능.
        return all(hf_file_cached(repo, "pytorch_model.bin") for repo in OPUS_MT_MODELS.values())

    def release_runtime_cache(self, engine: str, model: str | None) -> None:
        """삭제 직전 메모리에 올려둔 번역 모델을 내려 파일 잠금을 피한다."""
        with self._pipeline_lock:
            if engine == "llama" and model is not None:
                loaded = self._llama_models.pop(model, None)
                close = getattr(loaded, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            elif engine == "nllb" and model is not None:
                self._nllb_ct2_translators.pop(model, None)
                self._nllb_pipelines.pop(model, None)
                self._nllb_tokenizer = None
            elif engine == "opus":
                self._opus_pipelines.clear()


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
                word_timestamps=True,  # 자막 시각을 실제 단어 경계로 정밀하게 맞춘다.
            )

            if not segments:
                raise RuntimeError("인식할 수 있는 음성을 찾지 못했습니다.")

            # 대형 LLM 번역은 16GB VRAM을 거의 다 쓰므로, 번역 전에 위스퍼를 내려 자리를 비운다.
            if self.translator.needs_exclusive_vram():
                on_line("대형 번역 모델을 위해 음성 인식 모델을 메모리에서 내립니다 ...")
                self.asr.unload()

            on_progress(55)
            on_line("한국어로 번역하는 중 ...")
            on_line(f"번역 자원 사용: {self.translator.resource_summary()}")
            translated = self.translator.translate_texts([s.text for s in segments], source_language)
            if len(translated) != len(segments):
                translated = translated + ["" for _ in range(len(segments) - len(translated))]

            translated_segments: list[SubtitleSegment] = []
            skipped_translations = 0
            for original, text in zip(segments, translated):
                if should_stop is not None and should_stop():
                    raise OperationCancelled("취소되었습니다.")
                subtitle_text = clean_translation_for_subtitle(text, original.text)
                if subtitle_text:
                    translated_segments.append(
                        SubtitleSegment(
                            start=original.start,
                            end=original.end,
                            text=subtitle_text,
                        )
                    )
                else:
                    skipped_translations += 1

            if skipped_translations:
                on_line(f"번역 결과가 비어 있거나 원문만 나온 {skipped_translations}줄은 저장하지 않았습니다.")

            on_progress(85)
            on_line(f"SRT 저장: {output_srt}")
            write_srt(translated_segments, output_srt)

        on_progress(100)
        return output_srt


def _default_speech_timestamps(audio, sample_rate: int, min_silence_ms: int, max_utterance_s: float, speech_pad_ms: int):
    # faster-whisper 내장 silero VAD를 그대로 재활용한다(추가 설치 없음).
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_silence_duration_ms=min_silence_ms,
        max_speech_duration_s=max_utterance_s,
        speech_pad_ms=speech_pad_ms,
    )
    return get_speech_timestamps(audio, options, sampling_rate=sample_rate)


class _UtteranceSegmenter:
    """작은 오디오 프레임을 모아 침묵 지점에서 '발화' 단위로 잘라 내보낸다.

    on_utterance(start_seconds, clip)로 확정된 발화를 전달한다. start_seconds는 캡처
    시작 기준 절대 시각, clip은 mono float32 1D numpy 배열이다. VAD 함수는 테스트에서
    교체할 수 있도록 speech_timestamps_fn으로 주입한다(기본은 silero VAD).
    """

    def __init__(
        self,
        sample_rate: int,
        on_utterance: Callable[[float, object], None],
        min_silence_ms: int = LIVE_MIN_SILENCE_MS,
        max_utterance_s: float = LIVE_MAX_UTTERANCE_SECONDS,
        speech_pad_ms: int = LIVE_SPEECH_PAD_MS,
        speech_timestamps_fn: Callable[[object, int], list[dict]] | None = None,
    ):
        import numpy as np

        self._np = np
        self.sample_rate = sample_rate
        self.on_utterance = on_utterance
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_s)
        if speech_timestamps_fn is not None:
            self._speech_timestamps = speech_timestamps_fn
        else:
            self._speech_timestamps = lambda audio, sr: _default_speech_timestamps(
                audio, sr, min_silence_ms, max_utterance_s, speech_pad_ms
            )
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed_seconds = 0.0

    def _to_mono(self, frame):
        array = self._np.asarray(frame, dtype=self._np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)
        return array.reshape(-1)

    def add_frame(self, frame) -> None:
        mono = self._to_mono(frame)
        if mono.size:
            self._buffer = self._np.concatenate((self._buffer, mono))
        self._drain(final=False)

    def flush_remaining(self) -> None:
        # 정지/스트림 종료 시 버퍼에 남은 음성을 모두 내보낸다.
        self._drain(final=True)
        self._buffer = self._np.zeros(0, dtype=self._np.float32)
        self._consumed_seconds = 0.0

    def _drain(self, final: bool) -> None:
        if self._buffer.size == 0:
            return

        timestamps = self._speech_timestamps(self._buffer, self.sample_rate)
        if not timestamps:
            # 음성이 없는데 버퍼만 길어지면(잡음/무음) 앞부분을 버려 메모리 상한을 지킨다.
            if not final and self._buffer.size > self.max_utterance_samples:
                drop = self._buffer.size - self.max_utterance_samples
                self._consumed_seconds += drop / self.sample_rate
                self._buffer = self._buffer[drop:]
            return

        if final:
            closed = timestamps
        else:
            trailing = self._buffer.size - timestamps[-1]["end"]
            last_open = trailing < self.min_silence_samples
            force_split = self._buffer.size >= self.max_utterance_samples
            if last_open and not force_split:
                closed = timestamps[:-1]
            else:
                closed = timestamps

        if not closed:
            return

        # 절대 시각은 트림 이전 버퍼 기준으로 계산한 뒤, 마지막에 한 번 트림한다.
        for segment in closed:
            start = int(segment["start"])
            end = int(segment["end"])
            clip = self._buffer[start:end]
            if clip.size == 0:
                continue
            start_seconds = self._consumed_seconds + start / self.sample_rate
            self.on_utterance(start_seconds, clip.copy())

        cut_at = int(closed[-1]["end"])
        self._consumed_seconds += cut_at / self.sample_rate
        self._buffer = self._buffer[cut_at:]


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
        # 캡처는 작은 프레임으로 받아 _UtteranceSegmenter가 발화 단위로 모아 자른다.
        self.frame_frames = int(sample_rate * LIVE_FRAME_SECONDS)
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._chunk_queue: queue.Queue[tuple[float | None, object | None]] | None = None
        self._segmenter: _UtteranceSegmenter | None = None
        self._ffmpeg_process: subprocess.Popen | None = None
        self._source_url: str | None = None
        self._loopback_recorder: object | None = None
        self._srt_path: Path | None = None
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """매 실행마다 0으로 되돌리는 누적·상태 값(__init__과 start()가 함께 쓴다)."""
        self._srt_rows: list[SubtitleSegment] = []
        self._srt_emitted = False
        self._dropped_chunks = 0
        self._lag_warned_at = 0.0
        # 자동 언어 감지 흔들림 방지용 sticky 상태.
        self._sticky_language: str | None = None
        self._sticky_candidate: str | None = None
        self._sticky_candidate_hits = 0
        # 발화 처리 중 발생하는 일시적 오류(치명적 아님)의 상태·rate limit용.
        self._worker_error_warned_at = 0.0
        self._worker_error_count = 0

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
        save_srt_path: Path | None = None,
    ) -> None:
        if self._is_running():
            raise RuntimeError("실시간 자막 기능이 이미 실행 중입니다.")

        self._stop_event = stop_event
        stop_event.clear()
        self._chunk_queue = queue.Queue(maxsize=8)
        self._source_url = internet_url.strip() if internet_url else None
        self._srt_path = save_srt_path
        self._reset_run_state()
        source_text = f"인터넷 URL: {self._source_url}" if self._source_url else "PC 스피커 소리"
        emit("log", f"실시간 자막 작업을 준비합니다. 입력: {source_text}")
        if self._srt_path is not None:
            self._srt_path.parent.mkdir(parents=True, exist_ok=True)
            emit("log", f"실시간 자막 SRT 저장을 준비했습니다: {self._srt_path}")

        # 첫 청크에서 모델을 로딩하면 10초 가까이 걸려 그 사이 들어온 앞 구간이 버려진다.
        # 캡처를 시작하기 전에 음성 인식·번역 모델을 미리 올려 두어 첫 구간부터 따라잡게 한다.
        emit("live_status", "음성 인식·번역 모델을 준비하는 중입니다 ...")
        emit("log", "실시간 자막 모델을 준비합니다.")
        try:
            self.asr.prepare_model()
            emit("log", f"음성 인식 모델 준비 완료: {self.asr.model_label()}")
            self.translator.prepare_profile()
            emit("log", f"번역 모델 준비 완료: {self.translator.profile_label()}")
            self._warmup(emit)
        except Exception as exc:
            emit("log", f"실시간 모델 사전 준비 중 경고: {exc}")
        if stop_event.is_set():
            emit("log", "실시간 자막 시작이 취소되었습니다.")
            return
        emit("live_status", "실시간 자막을 시작합니다 ...")
        emit("log", "실시간 자막 캡처와 처리 스레드를 시작합니다.")

        # 캡처 두 경로(PC 소리/URL)가 같은 분할기를 공유한다. 확정된 발화를 워커 큐로 넘긴다.
        self._segmenter = _UtteranceSegmenter(
            self.sample_rate,
            lambda start_seconds, clip: self._safe_put_chunk((start_seconds, clip)),
        )

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
        emit("log", "실시간 자막 작업이 실행 중입니다.")

    def _warmup(self, emit: Callable[[str, object], None]) -> None:
        # 첫 추론은 CUDA 커널 준비로 유독 느리다. 시작할 때 짧은 더미 입력으로 인식·번역을 한 번씩
        # 미리 돌려, 그 준비 비용을 첫 실제 발화가 떠안지 않게 한다.
        try:
            import numpy as np

            dummy = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
            self.asr.transcribe(dummy, vad_filter=False, beam_size=LIVE_BEAM_SIZE)
            self.translator.translate_texts(["warm up"], "en")
            emit("log", "실시간 자막 준비가 완료되었습니다.")
        except Exception as exc:
            emit("log", f"실시간 자막 준비 중 경고: {exc}")

    def stop(self, timeout: float = 2.0, emit: Callable[[str, object], None] | None = None) -> None:
        def log(message: str) -> None:
            if emit is not None:
                emit("log", message)

        log("실시간 자막 내부 작업에 중지 신호를 보냈습니다.")
        if self._stop_event is not None:
            self._stop_event.set()
        if self._ffmpeg_process is not None:
            process = self._ffmpeg_process
            self._ffmpeg_process = None
            log("인터넷 실시간 캡처 프로세스를 종료하는 중입니다.")
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
            log("PC 스피커 캡처 장치를 닫는 중입니다.")
            try:
                _close_windows_audio_recorder(loopback_recorder)
                log("PC 스피커 캡처 장치를 닫았습니다.")
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
            log("실시간 캡처 스레드 종료를 기다리는 중입니다.")
            self._capture_thread.join(timeout=timeout)
            if self._capture_thread.is_alive():
                log("실시간 캡처 스레드가 아직 정리 중입니다.")
            else:
                log("실시간 캡처 스레드가 종료되었습니다.")
        if self._worker_thread is not None:
            log("실시간 자막 처리 스레드 종료를 기다리는 중입니다.")
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                log("실시간 자막 처리 스레드가 아직 정리 중입니다.")
            else:
                log("실시간 자막 처리 스레드가 종료되었습니다.")

        self._capture_thread = None
        self._worker_thread = None
        self._stop_event = None
        self._segmenter = None
        if self._chunk_queue is not None:
            # 남은 발화는 메모리상 numpy 배열이라 별도 파일 정리가 필요 없다.
            while True:
                try:
                    self._chunk_queue.get_nowait()
                except queue.Empty:
                    break
            self._chunk_queue = None

        self._source_url = None
        log("실시간 자막 내부 작업 정리가 완료되었습니다.")

    def _safe_put_chunk(self, item: tuple[float | None, object | None]) -> None:
        if self._chunk_queue is None:
            return
        try:
            self._chunk_queue.put_nowait(item)
        except queue.Full:
            try:
                self._chunk_queue.get_nowait()
                if item[0] is not None:
                    self._dropped_chunks += 1
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

        release_audio_thread: Callable[[], None] = lambda: None
        try:
            emit("log", "PC 스피커 캡처를 준비합니다.")
            import soundcard as sc
            release_audio_thread = _initialize_windows_audio_thread()

            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError("기본 스피커를 찾지 못했습니다.")

            emit("log", f"기본 스피커: {speaker.name}")
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with loopback.recorder(samplerate=self.sample_rate) as recorder:
                self._loopback_recorder = recorder
                emit("log", "PC 스피커 캡처를 시작했습니다.")
                try:
                    while not stop_event.is_set():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="data discontinuity in recording",
                            )
                            audio = recorder.record(numframes=self.frame_frames)

                        if stop_event.is_set():
                            break
                        if self._segmenter is None:
                            break
                        if not getattr(audio, "size", 1):
                            continue

                        self._segmenter.add_frame(audio)
                finally:
                    self._loopback_recorder = None
                    emit("log", "PC 스피커 캡처 장치를 정리했습니다.")
        except OperationCancelled:
            emit("log", "실시간 캡처가 취소되었습니다.")
        except Exception as exc:
            if stop_event.is_set():
                emit("log", f"실시간 캡처가 중지되었습니다: {exc}")
            else:
                emit("error", f"실시간 캡처 오류: {exc}")
        finally:
            if self._segmenter is not None:
                try:
                    self._segmenter.flush_remaining()
                except Exception:
                    pass
            self._safe_put_chunk((None, None))
            try:
                release_audio_thread()
            except Exception:
                pass
            emit("log", "PC 스피커 캡처 스레드가 종료되었습니다.")

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

            import numpy as np

            # ffmpeg가 16kHz mono s16le로 내보내므로 작은 프레임으로 모아 float32로 변환해 분할기에 넘긴다.
            frame_bytes = self.frame_frames * 2
            pending = bytearray()
            while not stop_event.is_set():
                need = frame_bytes - len(pending)
                if need > 0:
                    chunk = process.stdout.read(need)
                    if chunk:
                        pending.extend(chunk)
                    elif not pending:
                        break

                if len(pending) < frame_bytes:
                    continue

                data = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]

                if stop_event.is_set():
                    break
                if self._segmenter is None:
                    break

                frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._segmenter.add_frame(frame)
        except OperationCancelled:
            emit("log", "실시간 캡처가 취소되었습니다.")
        except Exception as exc:
            emit("error", f"실시간 캡처 오류: {url} - {exc}")
        finally:
            if process is not None:
                emit("log", "인터넷 실시간 캡처 프로세스를 정리합니다.")
                try:
                    if process.stdout is not None:
                        process.stdout.close()
                except Exception:
                    pass
                try:
                    process.terminate()
                except Exception:
                    pass
            if self._segmenter is not None:
                try:
                    self._segmenter.flush_remaining()
                except Exception:
                    pass
            self._safe_put_chunk((None, None))
            emit("log", "인터넷 실시간 캡처 스레드가 종료되었습니다.")

    def _worker_loop(self, emit: Callable[[str, object], None], stop_event: threading.Event) -> None:
        emit("log", "실시간 자막 처리 스레드가 시작되었습니다.")
        try:
            while True:
                if self._chunk_queue is None:
                    return

                try:
                    chunk = self._chunk_queue.get(timeout=0.25)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                start_seconds, audio = chunk
                if start_seconds is None:
                    break
                if audio is None:
                    continue
                if stop_event.is_set():
                    break

                work_started_at = time.monotonic()
                try:
                    # 이미 발화 단위로 잘라 침묵이 제거됐으므로 위스퍼 쪽 VAD는 끄고(이중 처리 방지),
                    # 디코딩 폭(beam)도 좁혀 실시간 속도를 높인다.
                    asr_started_at = time.monotonic()
                    lang_override = self._live_language_override()
                    segments, source_language = self.asr.transcribe(
                        audio,
                        should_stop=stop_event.is_set,
                        vad_filter=False,
                        beam_size=LIVE_BEAM_SIZE,
                        language_override=lang_override,
                        drop_hallucinations=True,
                    )
                    self._update_sticky_language(source_language, lang_override, emit)
                    asr_elapsed = time.monotonic() - asr_started_at
                    translate_elapsed = 0.0
                    if segments:
                        translate_started_at = time.monotonic()
                        translated = self.translator.translate_texts(
                            [s.text for s in segments],
                            source_language,
                            allow_original_fallback=False,
                        )
                        translate_elapsed = time.monotonic() - translate_started_at
                        if len(translated) != len(segments):
                            translated = translated + ["" for _ in range(len(segments) - len(translated))]
                        pairs = []
                        for segment, text in zip(segments, translated):
                            subtitle_text = clean_translation_for_subtitle(text, segment.text)
                            subtitle_text = tidy_korean_subtitle(subtitle_text)
                            if subtitle_text:
                                pairs.append((segment, subtitle_text))
                        lines = [text for _, text in pairs]
                        if lines:
                            emit("live_caption", "\n".join(lines[-2:]))
                        if self._srt_path is not None and pairs:
                            self._append_live_srt(start_seconds, pairs, emit)
                    utterance_seconds = len(audio) / self.sample_rate
                    self._report_processing_lag(
                        time.monotonic() - work_started_at,
                        utterance_seconds,
                        asr_elapsed,
                        translate_elapsed,
                        emit,
                    )
                except OperationCancelled:
                    emit("log", "실시간 번역이 취소되었습니다.")
                except Exception as exc:
                    # 발화 하나의 실패로 실시간 전체를 멈추지 않는다. 치명적 오류(모달 팝업+정지)
                    # 대신 로그·상태줄에만 남기고 다음 발화를 계속 처리한다.
                    self._report_worker_error(exc, emit)
        finally:
            emit("log", "실시간 자막 처리 스레드가 종료되었습니다.")

    def _source_is_auto(self) -> bool:
        # 사용자가 소스 언어를 직접 고르면 transcribe가 이미 그 언어를 강제하므로 sticky는 관여하지 않는다.
        return self.asr.source_language == SOURCE_LANG_AUTO

    def _live_language_override(self) -> str | None:
        # 자동 감지 모드에서만, 언어가 고정된 뒤에는 그 언어를 넘겨 감지 흔들림을 없앤다.
        if not self._source_is_auto():
            return None
        return self._sticky_language

    def _update_sticky_language(
        self,
        detected: str | None,
        lang_override: str | None,
        emit: Callable[[str, object], None],
    ) -> None:
        # 자동 감지 모드에서 아직 고정 전이라면, 같은 언어가 연속으로 감지될 때 그 언어로 고정한다.
        if not self._source_is_auto():
            return
        if lang_override is not None or not detected or detected == "auto":
            return
        if detected == self._sticky_candidate:
            self._sticky_candidate_hits += 1
        else:
            self._sticky_candidate = detected
            self._sticky_candidate_hits = 1
        if self._sticky_candidate_hits >= LIVE_STICKY_LANGUAGE_HITS:
            self._sticky_language = self._sticky_candidate
            emit(
                "log",
                f"실시간 감지 언어를 '{self._sticky_language}'로 고정합니다"
                "(짧은 발화마다 언어가 흔들려 오역되는 것을 방지).",
            )

    def _report_worker_error(self, exc: Exception, emit: Callable[[str, object], None]) -> None:
        # 일시적 오류는 로그에 모두 남기되, 상태줄 알림은 10초에 한 번으로 묶어 오류 폭주를 막는다.
        emit("log", f"실시간 처리 오류(계속 진행): {exc}")
        self._worker_error_count += 1
        now = time.monotonic()
        if now - self._worker_error_warned_at < 10.0:
            return
        self._worker_error_warned_at = now
        message = f"⚠ 실시간 처리 중 오류가 발생해 일부 구간을 건너뜁니다: {exc}"
        if self._worker_error_count > 1:
            message += f" (누적 {self._worker_error_count}건)"
        emit("live_worker_error", message)

    def _report_processing_lag(
        self,
        elapsed: float,
        utterance_seconds: float,
        asr_elapsed: float,
        translate_elapsed: float,
        emit: Callable[[str, object], None],
    ) -> None:
        # 한 발화 분석 시간이 그 발화 길이보다 길면 실시간을 따라가지 못해 일부 구간을 건너뛴다.
        # 짧은 발화는 모델 고정 오버헤드 탓에 비율만 보면 오경고가 잦아 1.5초 바닥값을 둔다.
        threshold = max(utterance_seconds * 1.15, 1.5)
        if elapsed <= threshold:
            return
        now = time.monotonic()
        if now - self._lag_warned_at < 10.0:
            return
        self._lag_warned_at = now
        message = (
            f"⏱ 실시간 분석이 느립니다: 분석 {elapsed:.1f}초"
            f"(인식 {asr_elapsed:.1f}초 + 번역 {translate_elapsed:.1f}초)"
            f" > 발화 길이 {utterance_seconds:.1f}초. "
            "따라가지 못해 일부 구간을 건너뜁니다"
        )
        if self._dropped_chunks:
            message += f" (지금까지 건너뜀 {self._dropped_chunks}구간)"
        message += ". 더 빠른 모델(예: tiny/base)이나 균형 모드를 권장합니다."
        emit("live_status", message)
        emit("log", message)

    def _append_live_srt(
        self,
        start_seconds: float,
        pairs: list[tuple[SubtitleSegment, str]],
        emit: Callable[[str, object], None],
    ) -> None:
        if self._srt_path is None:
            return
        offset = start_seconds
        for segment, text in pairs:
            self._srt_rows.append(
                SubtitleSegment(
                    start=segment.start + offset,
                    end=segment.end + offset,
                    text=text,
                )
            )
        try:
            write_srt(self._srt_rows, self._srt_path)
        except OSError as exc:
            emit("error", f"실시간 자막 저장 오류: {exc}")
            return
        if not self._srt_emitted:
            self._srt_emitted = True
            emit("live_subtitle_output", str(self._srt_path))


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
        self._codec = DEFAULT_RECORD_CODEC
        self._record_started_at: float | None = None

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
        region: tuple[int, int, int, int] | None = None,
        codec: str = DEFAULT_RECORD_CODEC,
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
        self._codec = codec if codec in RECORD_CODEC_LABELS else DEFAULT_RECORD_CODEC
        self._record_started_at = None

        command = build_screen_capture_command(ffmpeg, self._video_path, fps, region, self._codec)
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

        if self._audio_path is None:
            return

        wav_file: wave.Wave_write | None = None
        expected_channels: int | None = None
        release_audio_thread: Callable[[], None] = lambda: None
        try:
            import soundcard as sc
            release_audio_thread = _initialize_windows_audio_thread()

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
            try:
                release_audio_thread()
            except Exception:
                pass

    def _monitor_recording(self, emit: Callable[[str, object], None]) -> None:
        try:
            if self._output_path is None or self._video_path is None:
                raise RuntimeError("화면 녹화 준비가 끝나지 않았습니다.")

            codec_text = "MKV · " + ("H.264" if self._codec == "h264" else "FFV1 무손실")
            self._record_started_at = time.monotonic()
            emit("record_status", f"녹화 중 · 00:00:00 · 0 B · {codec_text}")
            emit("log", f"화면 녹화를 시작했습니다({codec_text}): {self._output_path}")

            process = self._video_process
            last_report = 0.0
            while self._stop_event is not None and not self._stop_event.is_set():
                if process is None or process.poll() is not None:
                    break
                now = time.monotonic()
                if now - last_report >= 1.0:
                    last_report = now
                    elapsed = now - (self._record_started_at or now)
                    try:
                        size = self._video_path.stat().st_size if self._video_path.exists() else 0
                    except OSError:
                        size = 0
                    emit(
                        "record_status",
                        f"녹화 중 · {format_hms(elapsed)} · {format_file_size(size)} · {codec_text}",
                    )
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
            emit("eta_status", "대기 중")

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
        try:
            self.attributes("-alpha", 0.95)
        except Exception:
            pass
        # 바깥 테두리를 인디고 액센트로 얇게 깔아 카드 느낌을 줍니다.
        self.configure(
            bg=UI_ACCENT,
            cursor="fleur",
            highlightbackground=UI_ACCENT_ACTIVE,
            highlightcolor=UI_ACCENT_ACTIVE,
            highlightthickness=2,
        )
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._visible = False
        self._clear_after_id: str | None = None

        # 윈도우 상단 바가 없는 상태에서도 마우스로 위치를 이동할 수 있게 드래그 이벤트 등록
        self.bind("<Button-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._on_drag)
        # 창 폭이 바뀌면 자막 줄바꿈 폭을 폭에 맞춰 갱신한다.
        self.bind("<Configure>", self._on_overlay_configure)

        self.label = tk.Label(
            self,
            text="",
            fg=OVERLAY_FG,
            bg=OVERLAY_BG,
            font=(resolve_ui_font_family(), 26, "bold"),
            justify="center",
            wraplength=1000,
            padx=28,
            pady=20,
            cursor="fleur",
        )
        self.label.bind("<Button-1>", self._begin_drag)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.pack(fill="both", expand=True, padx=6, pady=6)

        screen_w = self.winfo_screenwidth()
        self.geometry(f"1100x220+{max(20, (screen_w - 1100) // 2)}+20")
        self.label.config(text="실시간 자막 준비 완료")
        self.show()

    def _begin_drag(self, event: tk.Event) -> None:
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def _on_drag(self, event: tk.Event) -> None:
        self.geometry(f"+{event.x_root - self._drag_start_x + self.winfo_x()}+{event.y_root - self._drag_start_y + self.winfo_y()}")
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def _on_overlay_configure(self, event: tk.Event) -> None:
        label = getattr(self, "label", None)
        if event.widget is self and label is not None:
            label.config(wraplength=max(300, int(event.width) - 80))

    def set_text(self, text: str) -> None:
        if not text:
            return
        self.label.config(text=text)
        self._schedule_idle_clear()

    def _schedule_idle_clear(self) -> None:
        # 새 자막이 한동안 오지 않으면(무음/환각 필터로 걸러짐) 오래된 자막을 지워
        # 화면에 멈춘 것처럼 남지 않게 한다.
        if self._clear_after_id is not None:
            try:
                self.after_cancel(self._clear_after_id)
            except Exception:
                pass
        self._clear_after_id = self.after(OVERLAY_IDLE_CLEAR_MS, self._clear_idle_text)

    def _clear_idle_text(self) -> None:
        self._clear_after_id = None
        self.label.config(text="")

    def show(self) -> None:
        self.deiconify()
        self.lift()
        self._visible = True

    def hide(self) -> None:
        self.withdraw()
        self._visible = False

    def is_visible(self) -> bool:
        return self._visible

    def close(self) -> None:
        try:
            self._visible = False
            if self._clear_after_id is not None:
                try:
                    self.after_cancel(self._clear_after_id)
                except Exception:
                    pass
                self._clear_after_id = None
            self.destroy()
        except Exception:
            pass


class PapaStopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # 제목 표시줄에 프로그램명을 표시하고, 사이드바에는 탭 메뉴만 둔다.
        self.title(APP_NAME)
        self._configure_main_window()
        self._apply_theme()
        self.configure(fg_color=UI_BG)

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # 모델 다운로드(huggingface_hub) 진행률을 GUI 공통 진행 영역에 실시간 표시한다.
        install_hf_download_hook()
        set_download_progress_callback(lambda data: self._emit("download_progress", data))

        self.asr = ASREngine()
        # 실시간은 반응속도, 파일은 정확도(large-v3-turbo)로 모델을 따로 쓰도록 엔진을 분리한다.
        # 실시간 기본은 medium: small보다 인식 정확도가 뚜렷이 높고, GPU(예: RTX 5070 Ti 16GB)면
        # 실시간을 따라간다. GPU 여유가 적은 PC는 실시간 탭에서 small/base로 낮출 수 있다.
        self.live_asr = ASREngine(model_name="medium")
        self.translator = TranslationEngine(preferred="auto")
        # 실시간 번역도 파일과 따로 둔다. 기본은 품질 우선(1.3B), 실시간 탭 콤보에서 600M(빠름)로 바꿀 수 있다.
        # realtime=True: 실시간 지연 예산에 맞춰 NLLB beam을 낮춰(3) 번역 속도를 확보한다.
        self.live_translator = TranslationEngine(preferred="balanced", realtime=True)
        self.file_workflow = FileSubtitleWorkflow(self.asr, self.translator)
        self.live_service = LiveCaptionService(self.live_asr, self.live_translator)
        self.record_service = ScreenRecorderService()

        self.file_cancel_event = threading.Event()
        self.live_cancel_event = threading.Event()
        self.record_cancel_event = threading.Event()
        self.file_queue: list[str] = []
        self._batch_outputs: list[Path] = []
        self._batch_current = 0
        self._failed_sources: list[str] = []
        self.last_output: Path | None = None
        self.overlay: LiveSubtitleOverlay | None = None
        self._is_file_running = False
        self._is_live_running = False
        self._is_record_running = False
        self._is_asr_preparing = False
        self._is_translation_preparing = False
        self.progress_started_at: float | None = None

        self._build_ui()
        self._apply_saved_settings(load_user_settings())
        self._refresh_status()
        self._refresh_system_specs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._process_events)

    def _apply_saved_settings(self, data: dict) -> None:
        """지난 실행에서 저장한 모델·번역·자원 방식 선택을 UI와 엔진에 되살린다."""
        if not data:
            return

        mode = data.get("resource_mode")
        if mode in RESOURCE_MODE_LABELS:
            self.resource_mode_var.set(RESOURCE_MODE_LABELS[mode])
            self._apply_resource_mode_from_selection()

        file_whisper = data.get("file_whisper_model")
        if file_whisper in WHISPER_MODEL_LABELS:
            self.whisper_model_var.set(WHISPER_MODEL_LABELS[file_whisper])
            self.asr.set_model(file_whisper)

        file_profile = data.get("file_translation_profile")
        if file_profile in TRANSLATION_PROFILE_LABELS:
            self.translation_profile_var.set(TRANSLATION_PROFILE_LABELS[file_profile])
            self.translator.set_profile(file_profile)

        source_lang = data.get("source_lang")
        if source_lang in SOURCE_LANG_LABELS:
            self.source_lang_var.set(SOURCE_LANG_LABELS[source_lang])
            self.asr.set_source_language(source_lang)
            self.live_asr.set_source_language(source_lang)

        live_whisper = data.get("live_whisper_model")
        if live_whisper in WHISPER_MODEL_LABELS:
            self.live_whisper_model_var.set(WHISPER_MODEL_LABELS[live_whisper])
            self.live_asr.set_model(live_whisper)

        live_profile = data.get("live_translation_profile")
        if live_profile in TRANSLATION_PROFILE_LABELS:
            self.live_translation_var.set(TRANSLATION_PROFILE_LABELS[live_profile])
            self.live_translator.set_profile(live_profile)
            self._update_live_secondary_state()

        live_source = data.get("live_source")
        if live_source in (LIVE_SOURCE_SPEAKER, LIVE_SOURCE_URL):
            self.live_source_var.set(live_source)
            self._on_live_source_selected()

        self._refresh_model_status()

    def _collect_current_settings(self) -> dict:
        return {
            "resource_mode": self._resource_mode_from_selection(),
            "file_whisper_model": self._whisper_model_from_selection(),
            "file_translation_profile": self._translation_profile_from_selection(),
            "source_lang": self._source_lang_id_from_label(self.source_lang_var.get()),
            "live_whisper_model": self._live_whisper_model_from_selection(),
            "live_translation_profile": self._live_translation_profile_from_selection(),
            "live_source": self.live_source_var.get(),
        }

    def _on_close(self) -> None:
        try:
            save_user_settings(self._collect_current_settings())
        except Exception:
            pass
        self.destroy()

    def _apply_theme(self) -> None:
        """customtkinter 라이트 테마 + 인디고 액센트를 전역으로 설정합니다."""
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        indigo = [UI_ACCENT, UI_ACCENT]
        indigo_hover = [UI_ACCENT_HOVER, UI_ACCENT_HOVER]
        indigo_active = [UI_ACCENT_ACTIVE, UI_ACCENT_ACTIVE]
        scroll_color = [UI_BORDER_STRONG, UI_BORDER_STRONG]
        scroll_hover = [UI_TEXT_MUTED, UI_TEXT_MUTED]

        theme = ctk.ThemeManager.theme

        def assign(widget: str, key: str, value: object) -> None:
            if widget in theme and key in theme[widget]:
                theme[widget][key] = value

        assign("CTkButton", "fg_color", indigo)
        assign("CTkButton", "hover_color", indigo_hover)
        assign("CTkProgressBar", "progress_color", indigo)
        # 드롭다운은 입력칸처럼 흰 배경 + 본문 글자색으로 두고, 오른쪽 화살표 칸에만 옅은 인디고를 쓴다.
        assign("CTkOptionMenu", "fg_color", [UI_SURFACE, UI_SURFACE])
        assign("CTkOptionMenu", "text_color", [UI_TEXT, UI_TEXT])
        assign("CTkOptionMenu", "button_color", indigo)
        assign("CTkOptionMenu", "button_hover_color", indigo_hover)
        assign("CTkOptionMenu", "dropdown_fg_color", [UI_SURFACE, UI_SURFACE])
        assign("CTkOptionMenu", "dropdown_hover_color", [UI_SURFACE_HOVER, UI_SURFACE_HOVER])
        assign("CTkOptionMenu", "dropdown_text_color", [UI_TEXT, UI_TEXT])
        assign("CTkCheckBox", "fg_color", indigo)
        assign("CTkCheckBox", "hover_color", indigo_hover)
        assign("CTkRadioButton", "fg_color", indigo)
        assign("CTkRadioButton", "hover_color", indigo_hover)
        assign("CTkSegmentedButton", "selected_color", [UI_ACCENT_SOFT, UI_ACCENT_SOFT])
        assign("CTkSegmentedButton", "selected_hover_color", [UI_ACCENT_SOFT_HOVER, UI_ACCENT_SOFT_HOVER])
        assign("CTkScrollbar", "button_color", scroll_color)
        assign("CTkScrollbar", "button_hover_color", scroll_hover)

        # customtkinter 기본 글꼴(Roboto)은 한글이 깨지므로 한글 지원 글꼴로 강제한다.
        self._ui_font_family = resolve_ui_font_family()
        try:
            if isinstance(theme.get("CTkFont"), dict):
                theme["CTkFont"]["family"] = self._ui_font_family
        except Exception:
            pass

    def _configure_main_window(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1120, max(980, screen_w - 80))
        height = min(860, max(680, screen_h - 100))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(900, width), min(620, height))

    def _make_card(self, parent, title: str):
        """제목이 달린 흰색 카드 프레임을 만들고 (카드, 내용 프레임)을 돌려줍니다."""
        card = ctk.CTkFrame(
            parent, fg_color=UI_SURFACE, corner_radius=6, border_width=1, border_color=UI_CARD_BORDER
        )
        ctk.CTkLabel(
            card, text=title, font=self.font_section, text_color=UI_TEXT, anchor="w"
        ).pack(fill="x", padx=14, pady=(10, 2))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        return card, body

    def _make_model_status_row(self, parent):
        """음성 인식·번역 모델 준비 상태를 한 줄로 보여주는 행을 만든다.
        공백 정렬 대신 고정 라벨로 맞춰 탭마다 위치가 어긋나지 않게 한다.
        (행 프레임, 음성 인식 상태 라벨, 번역 상태 라벨)을 돌려준다."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(row, text="모델 상태", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        ctk.CTkLabel(row, text="음성 인식", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left", padx=(14, 4))
        asr_lbl = ctk.CTkLabel(row, text="확인 중", font=self.font_base, text_color=UI_TEXT_MUTED)
        asr_lbl.pack(side="left")
        ctk.CTkLabel(row, text="번역", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left", padx=(18, 4))
        trans_lbl = ctk.CTkLabel(row, text="확인 중", font=self.font_base, text_color=UI_TEXT_MUTED)
        trans_lbl.pack(side="left")
        return row, asr_lbl, trans_lbl

    def _live_label(self, parent, variable: tk.StringVar, **kwargs):
        """StringVar 값이 바뀌면 자동으로 글자를 갱신하는 CTkLabel을 만듭니다."""
        label = ctk.CTkLabel(parent, text=variable.get(), **kwargs)
        variable.trace_add("write", lambda *_a, v=variable, w=label: w.configure(text=v.get()))
        return label

    def _make_option_menu(self, parent, *, width: int, **kwargs):
        """CTkOptionMenu는 테두리 인자를 지원하지 않으므로 1px 띠 프레임으로 감싸 입력칸과
        같은 테두리를 입힌다. (holder, option_menu)를 돌려준다.
        배치는 holder에 하고, 상태 제어(state)는 option_menu에 한다."""
        holder = ctk.CTkFrame(
            parent, fg_color=UI_BORDER_STRONG, corner_radius=5, width=width + 2, height=36
        )
        holder.pack_propagate(False)
        menu = ctk.CTkOptionMenu(holder, **{**self._option_kw, **kwargs})
        menu.pack(fill="both", expand=True, padx=1, pady=1)
        return holder, menu

    def _make_form_frame(self, parent):
        """탭마다 입력 줄의 시작 위치가 같도록 쓰는 공통 폼 프레임."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, minsize=self._label_w)
        frame.grid_columnconfigure(1, weight=1)
        return frame

    def _form_label(self, parent, text: str, *, width: int | None = None, **kwargs):
        label_width = self._label_w if width is None else width
        return ctk.CTkLabel(
            parent,
            text=text,
            width=label_width,
            font=self.font_base,
            text_color=UI_TEXT_MUTED,
            anchor="w",
            **kwargs,
        )

    def _hint_label(self, parent, text: str, *, wraplength: int = 360, **kwargs):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=self.font_base,
            text_color=UI_TEXT_MUTED,
            justify="left",
            anchor="w",
            wraplength=wraplength,
            **kwargs,
        )

    def _make_status_row(self, parent, items: list[tuple[str, tk.StringVar]]):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        for index, (label, variable) in enumerate(items):
            ctk.CTkLabel(
                row,
                text=label,
                font=self.font_base,
                text_color=UI_TEXT_MUTED,
            ).pack(side="left", padx=((20, 0) if index else (0, 0)))
            self._live_label(
                row,
                variable,
                font=self.font_base,
                text_color=UI_TEXT,
                anchor="w",
            ).pack(side="left", padx=(8, 0))
        return row

    def _build_help_tab(self) -> None:
        help_scroll = ctk.CTkScrollableFrame(
            self.help_tab,
            fg_color="transparent",
            scrollbar_button_color=UI_BORDER_STRONG,
            scrollbar_button_hover_color=UI_TEXT_MUTED,
        )
        help_scroll.pack(fill="both", expand=True)

        def add_section(title: str, lines: list[str]) -> None:
            card, body = self._make_card(help_scroll, title)
            card.pack(fill="x", padx=2, pady=(0, 10))
            for line in lines:
                ctk.CTkLabel(
                    body,
                    text=line,
                    font=self.font_base,
                    text_color=UI_TEXT,
                    justify="left",
                    anchor="w",
                    wraplength=820,
                ).pack(fill="x", pady=2)

        add_section(
            "제작자",
            [
                "제작자 : 황재훈",
                "연락처 : hunmonkey@hanmail.net",
                "허가받은 사용자 외의 사용을 금합니다.",
                "외부 유출, 유포, 공유를 금지합니다.",
            ],
        )
        add_section(
            "사이드 메뉴",
            [
                "파일 자막: 영상이나 음성 파일을 SRT 자막으로 만들고 필요하면 번역합니다.",
                "실시간 자막: PC 스피커 소리나 인터넷 URL 스트림을 실시간으로 듣고 한국어 자막을 표시합니다.",
                "화면 녹화: 전체 화면, 모니터, 선택 영역을 MKV 파일로 녹화합니다.",
                "시스템 사양: 현재 PC의 CPU, RAM, 그래픽카드와 자원 사용 방식을 확인합니다.",
                "도움말: 각 메뉴와 버튼 사용법, 주의사항을 확인합니다.",
            ],
        )
        add_section(
            "파일 자막 메뉴",
            [
                "원본 영상/음성: 자막으로 만들 파일 경로입니다. 직접 입력하거나 선택 버튼으로 고릅니다.",
                "선택: 영상이나 음성 파일 하나를 고릅니다.",
                "목록에 추가: 현재 원본 경로를 작업할 파일 목록에 넣습니다.",
                "저장할 SRT / 폴더: 결과 자막 파일이나 저장 폴더를 지정합니다.",
                "같은 이름으로: 원본 파일과 같은 위치에 같은 이름의 .srt 경로를 자동으로 넣습니다.",
                "폴더 선택: 여러 파일을 처리할 때 결과를 저장할 폴더를 고릅니다.",
                "자막 만들기 시작: 목록의 파일을 순서대로 처리합니다. 작업 중에는 설정을 바꾸지 않는 것이 안전합니다.",
                "중지: 현재 파일 작업을 멈추도록 요청합니다. 처리 중인 단계에 따라 몇 초 걸릴 수 있습니다.",
                "결과 열기: 마지막으로 만들어진 결과 파일이나 폴더를 엽니다.",
                "음성 인식 모델: 빠를수록 처리 속도가 좋고, 큰 모델일수록 정확도가 좋아질 수 있습니다.",
                "위스퍼 모델 다운로드: 선택한 음성 인식 모델을 미리 내려받습니다.",
                "원본 언어: 자동 감지가 기본입니다. 언어를 알고 있으면 직접 고르면 오인식이 줄 수 있습니다.",
                "번역 모델: 속도와 품질의 균형을 고릅니다. 큰 모델은 시간이 오래 걸리고 저장 공간을 많이 씁니다.",
                "번역 모델 다운로드: 선택한 번역 모델을 미리 준비합니다.",
                "파일 추가/폴더 추가: 작업 목록에 파일 또는 폴더 안의 미디어 파일을 넣습니다.",
                "선택 삭제/전체 비우기: 작업 목록을 정리합니다. 원본 파일은 삭제하지 않습니다.",
                "위로/아래로: 여러 파일 처리 순서를 바꿉니다.",
                "주의: 처음 쓰는 모델은 자동 다운로드가 필요할 수 있어 시간이 오래 걸립니다.",
            ],
        )
        add_section(
            "실시간 자막 메뉴",
            [
                "PC 스피커 소리: 현재 PC에서 재생되는 소리를 듣고 자막을 만듭니다.",
                "인터넷 URL 스트림: URL을 통해 들어오는 실시간 방송이나 오디오 스트림을 처리합니다.",
                "URL: 인터넷 URL 스트림을 선택했을 때만 사용합니다. http, https, rtmp, rtmps, rtsp 주소를 입력합니다.",
                "원본 언어: 자동 감지가 기본입니다. 자막이 엉뚱한 언어로 나오면 언어를 직접 고릅니다.",
                "음성 인식 모델: 실시간은 tiny 또는 base처럼 빠른 모델을 권장합니다.",
                "번역 모델: 정확 모델은 품질이 좋지만 느릴 수 있고, 빠른 모델은 반응이 빠르지만 표현이 단순할 수 있습니다.",
                "OPUS 2차 번역: OPUS 선택 시 일본어/중국어를 영어로 거친 뒤 한국어로 다시 번역할 때 사용합니다.",
                "시작: 실시간 자막을 시작합니다. 모델 준비, 캡처 시작, 처리 스레드 시작 과정은 아래 작업 기록에 남습니다.",
                "중지: 실시간 캡처와 번역 처리를 멈추고 관련 장치를 정리합니다. 정리 과정도 작업 기록에 남습니다.",
                "결과 열기: SRT 저장을 켠 경우 마지막 저장 파일을 엽니다.",
                "자막 창을 항상 위에 표시: 별도 자막창을 화면 위에 띄웁니다.",
                "자막 파일(.srt)로도 저장: 실시간 자막을 파일로 누적 저장합니다.",
                "자막창 보이기/숨기기: 실시간 작업 중 별도 자막창 표시를 바꿉니다.",
                "주의: PC 스피커 소리 캡처는 Windows 소리 장치 상태에 영향을 받습니다. 오류가 나면 사용 중인 재생 장치가 맞는지 확인합니다.",
                "주의: 인터넷 URL 스트림에는 ffmpeg가 필요합니다. 주소가 잘못됐거나 방송이 끊기면 캡처가 중단될 수 있습니다.",
            ],
        )
        add_section(
            "화면 녹화 메뉴",
            [
                "저장할 MKV: 녹화 결과 파일 경로입니다.",
                "찾아보기: 저장할 파일 위치를 직접 고릅니다.",
                "캡처 대상: 전체 화면, 모니터, 영역 지정 중 하나를 고릅니다.",
                "영역 선택: 영역 지정 모드에서 녹화할 사각형 영역을 마우스로 지정합니다.",
                "녹화 형식: FFV1은 무손실이라 크고, H.264는 용량이 작지만 압축됩니다.",
                "초당 화면 수: 24, 30, 60 중 선택합니다. 높을수록 부드럽지만 파일이 커질 수 있습니다.",
                "PC 스피커 소리도 녹음: 화면과 함께 PC 소리를 WAV로 임시 녹음한 뒤 영상과 합칩니다.",
                "녹화 시작: 화면 녹화를 시작합니다.",
                "중지: 녹화를 멈추고 임시 영상과 소리를 정리한 뒤 최종 파일을 저장합니다.",
                "결과 열기: 마지막 녹화 결과 파일을 엽니다.",
                "주의: 소리 녹음을 켠 경우 스피커 소리가 비어 있으면 영상만 저장될 수 있습니다.",
                "주의: 무손실 녹화는 파일 크기가 매우 커질 수 있으니 저장 공간을 먼저 확인합니다.",
            ],
        )
        add_section(
            "시스템 사양 메뉴",
            [
                "CPU/RAM/VGA: 현재 PC의 기본 사양을 보여줍니다.",
                "자원 사용: 최대 성능 또는 균형 모드를 고릅니다.",
                "최대 성능: CPU/GPU를 적극적으로 사용해 빠르게 처리합니다. 다른 작업이 느려질 수 있습니다.",
                "균형: PC 여유를 남겨 두는 설정입니다. 처리 속도는 조금 느릴 수 있습니다.",
                "사양 새로고침: PC 사양과 엔진 상태를 다시 확인합니다.",
                "주의: 작업 중에는 자원 사용 방식을 바꾸지 않는 것이 안전합니다.",
            ],
        )
        add_section(
            "실시간 로그",
            [
                "제목줄: 현재 상태와 예상 시간을 한 줄로 보여줍니다.",
                "진행률 바: 파일 자막처럼 진행률 계산이 가능한 작업에서 처리 정도를 보여줍니다.",
                "로그 창: 시작 버튼 클릭, 모델 준비, 캡처 시작, 파일 저장, 중지 요청, 정리 완료 같은 세부 로그가 남습니다.",
                "오류가 나오면 로그 창의 마지막 몇 줄을 먼저 확인합니다. 어떤 단계에서 멈췄는지 확인하는 단서입니다.",
                "로그는 아래 영역 안에서 스크롤됩니다. 오래된 기록은 위쪽에 남고 새 기록은 아래에 추가됩니다.",
            ],
        )
        add_section(
            "모델과 저장 공간 주의사항",
            [
                "모델 파일은 보통 .papastop_data 폴더에 저장됩니다. 모델에 따라 수백 MB에서 수십 GB까지 커질 수 있습니다.",
                "처음 실행하거나 새 모델을 고르면 다운로드 시간이 필요합니다.",
                "GPU가 없거나 CUDA가 준비되지 않은 PC에서는 CPU로 처리되어 속도가 느릴 수 있습니다.",
                "GitHub에는 모델, 로그, 테스트 산출물 같은 큰 파일을 올리지 않도록 .gitignore에 제외 규칙이 들어 있습니다.",
                "문제가 반복되면 실시간 로그의 오류 문구와 사용한 입력 방식, 선택한 모델을 함께 확인합니다.",
            ],
        )

    def _build_ui(self) -> None:
        root = self
        # 왼쪽 세로 사이드바(고정폭) + 오른쪽 콘텐츠/로그 영역(가변폭)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        fam = self._ui_font_family
        self.font_base = ctk.CTkFont(family=fam, size=13)
        self.font_section = ctk.CTkFont(family=fam, size=14, weight="bold")
        self.font_caption = ctk.CTkFont(family=fam, size=15)
        # 선택된 사이드바 탭만 굵게 — 지금 어느 화면인지 한눈에 보이게 한다.
        self.font_nav_bold = ctk.CTkFont(family=fam, size=13, weight="bold")

        self._ghost_btn = dict(
            fg_color="transparent",
            hover_color=UI_SURFACE_HOVER,
            text_color=UI_ACCENT,
            border_width=1,
            border_color=UI_BORDER_STRONG,
            corner_radius=4,
        )
        self._danger_btn = dict(fg_color=UI_DANGER, hover_color=UI_DANGER_HOVER, text_color=UI_SURFACE)
        self._entry_kw = dict(
            height=34,
            corner_radius=4,
            border_width=1,
            border_color=UI_BORDER_STRONG,
            fg_color=UI_SURFACE,
            text_color=UI_TEXT,
            font=self.font_base,
        )
        # 선택창은 입력칸처럼: 흰 배경 + 본문 글자색, 오른쪽 화살표 칸에만 옅은 인디고.
        # (CTkOptionMenu는 테두리 인자를 지원하지 않아 _make_option_menu 래퍼로 1px 테두리를 입힌다.)
        self._option_kw = dict(
            height=34,
            corner_radius=4,
            font=self.font_base,
            dropdown_font=self.font_base,
            fg_color=UI_SURFACE,
            text_color=UI_TEXT,
            button_color=UI_ACCENT,
            button_hover_color=UI_ACCENT_HOVER,
            dropdown_fg_color=UI_SURFACE,
            dropdown_hover_color=UI_SURFACE_HOVER,
            dropdown_text_color=UI_TEXT,
        )

        # 모든 탭에서 똑같이 쓰는 주요 동작 버튼 규격 (시작 / 중지 / 결과 열기)
        self._primary_btn_kw = dict(width=140, height=34, corner_radius=4, font=self.font_base)
        self._stop_btn_kw = dict(width=84, height=34, corner_radius=4, **self._danger_btn)
        self._open_btn_kw = dict(width=96, height=34, **self._ghost_btn)

        # 입력 행 왼쪽 라벨 공통 폭(=입력칸 시작 위치 정렬)과 보조 버튼 공통 높이
        self._label_w = 130
        self._sec_btn_h = 34

        # ── 왼쪽 세로 사이드바: 탭 버튼 4개 ───────────────────────
        sidebar = ctk.CTkFrame(
            root, fg_color=UI_SURFACE, corner_radius=6, border_width=1,
            border_color=UI_CARD_BORDER, width=156,
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        sidebar.grid_propagate(False)
        sidebar.pack_propagate(False)

        # 사이드바 상단 브랜드 줄 + 얇은 구분선 (탭 목록과 시각적으로 분리)
        sidebar_header = ctk.CTkFrame(sidebar, fg_color="transparent")
        sidebar_header.pack(fill="x", padx=14, pady=(14, 8))
        logo = ctk.CTkFrame(sidebar_header, width=28, height=28, corner_radius=7, fg_color=UI_ACCENT)
        logo.pack(side="left")
        logo.pack_propagate(False)
        ctk.CTkLabel(
            logo, text="P", font=ctk.CTkFont(family=fam, size=16, weight="bold"),
            text_color=UI_SURFACE,
        ).pack(fill="both", expand=True)
        ctk.CTkLabel(
            sidebar_header, text=APP_NAME, font=self.font_section, text_color=UI_ACCENT, anchor="w"
        ).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ctk.CTkFrame(sidebar, height=1, fg_color=UI_BORDER).pack(fill="x", padx=12, pady=(0, 8))

        # ── 오른쪽 영역: 위(탭 콘텐츠) + 아래(실시간 로그) ────────
        right = ctk.CTkFrame(root, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=14)
        right.grid_columnconfigure(0, weight=1)
        # 실시간 로그가 한 줄 더 위로 올라오도록 아래 영역 비율을 조금 키운다.
        right.grid_rowconfigure(0, weight=4)
        right.grid_rowconfigure(1, weight=3)
        self._right_panel = right

        content = ctk.CTkFrame(right, fg_color="transparent")
        content.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self._content_panel = content

        # 탭을 같은 칸에 겹쳐 쌓고 tkraise로 전환한다.
        file_tab_host = ctk.CTkFrame(content, fg_color="transparent")
        live_tab_host = ctk.CTkFrame(content, fg_color="transparent")
        record_tab_host = ctk.CTkFrame(content, fg_color="transparent")
        system_tab_host = ctk.CTkFrame(content, fg_color="transparent")
        help_tab_host = ctk.CTkFrame(content, fg_color="transparent")
        for host in (file_tab_host, live_tab_host, record_tab_host, system_tab_host, help_tab_host):
            host.grid(row=0, column=0, sticky="nsew")

        self.file_tab = ctk.CTkFrame(file_tab_host, fg_color="transparent")
        self.file_tab.pack(fill="both", expand=True)
        self.live_tab = live_tab_host
        self.record_tab = record_tab_host
        self.system_tab = system_tab_host
        self.help_tab = help_tab_host

        self._tab_hosts = {
            "file": file_tab_host,
            "live": live_tab_host,
            "record": record_tab_host,
            "system": system_tab_host,
            "help": help_tab_host,
        }
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in (
            ("file", "파일 자막"),
            ("live", "실시간 자막"),
            ("record", "화면 녹화"),
            ("system", "시스템 사양"),
            ("help", "도움말"),
        ):
            btn = ctk.CTkButton(
                sidebar,
                text=label,
                anchor="w",
                command=lambda k=key: self._show_tab(k),
                height=38,
                corner_radius=4,
                fg_color="transparent",
                hover_color=UI_SURFACE_HOVER,
                text_color=UI_TEXT,
                font=self.font_base,
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav_buttons[key] = btn

        # 공통 로그/진행 영역(아래)은 도움말 탭에서만 숨긴다.

        self.system_cpu_var = tk.StringVar(value="확인 중 ...")
        self.system_ram_var = tk.StringVar(value="확인 중 ...")
        self.system_vga_var = tk.StringVar(value="확인 중 ...")
        self.resource_mode_var = tk.StringVar(value=RESOURCE_MODE_LABELS[DEFAULT_RESOURCE_MODE])
        self.resource_plan_var = tk.StringVar(value=build_resource_settings(DEFAULT_RESOURCE_MODE).summary)

        system_card, system_body = self._make_card(self.system_tab, "현재 PC 사양")
        system_card.pack(fill="x", pady=(2, 10))
        system_form = self._make_form_frame(system_body)
        system_form.pack(fill="x")
        system_form.grid_columnconfigure(2, weight=1)

        def add_system_row(row_index: int, label: str, variable: tk.StringVar) -> None:
            self._form_label(system_form, label).grid(row=row_index, column=0, sticky="nw", pady=4)
            self._live_label(
                system_form,
                variable,
                font=self.font_base,
                justify="left",
                anchor="w",
                wraplength=620,
                text_color=UI_TEXT,
            ).grid(row=row_index, column=1, columnspan=2, sticky="ew", padx=10, pady=4)

        add_system_row(0, "CPU", self.system_cpu_var)
        add_system_row(1, "RAM", self.system_ram_var)
        add_system_row(2, "VGA", self.system_vga_var)

        self._form_label(system_form, "자원 사용").grid(row=3, column=0, sticky="nw", pady=4)
        resource_mode_holder, self.resource_mode_combo = self._make_option_menu(
            system_form,
            variable=self.resource_mode_var,
            values=[RESOURCE_MODE_LABELS[key] for key in RESOURCE_MODE_ORDER],
            command=self._on_resource_mode_selected,
            width=280,
        )
        resource_mode_holder.grid(row=3, column=1, sticky="w", padx=10, pady=4)
        self._live_label(
            system_form,
            self.resource_plan_var,
            font=self.font_base,
            justify="left",
            anchor="w",
            wraplength=520,
            text_color=UI_TEXT_MUTED,
        ).grid(row=3, column=2, sticky="ew", padx=(0, 10), pady=4)

        system_action_row = ctk.CTkFrame(self.system_tab, fg_color="transparent")
        system_action_row.pack(fill="x", pady=6)
        ctk.CTkButton(
            system_action_row,
            text="사양 새로고침",
            command=self._refresh_system_specs,
            width=120,
            height=34,
            **self._ghost_btn,
        ).pack(side="left")

        # File tab
        self.file_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.file_status_var = tk.StringVar(value="대기 중")
        self.batch_status_var = tk.StringVar(value="0개 파일")
        self.whisper_model_var = tk.StringVar(value=WHISPER_MODEL_LABELS[DEFAULT_WHISPER_MODEL])
        self.source_lang_var = tk.StringVar(value=SOURCE_LANG_LABELS[SOURCE_LANG_AUTO])
        self.translation_profile_var = tk.StringVar(value=TRANSLATION_PROFILE_LABELS["auto"])
        self.live_url_var = tk.StringVar()
        self.live_source_var = tk.StringVar(value=LIVE_SOURCE_SPEAKER)
        self.record_output_var = tk.StringVar(value=str(default_recording_path()))
        self.record_status_var = tk.StringVar(value="대기 중")
        self.record_audio_var = tk.BooleanVar(value=True)
        self.record_fps_var = tk.StringVar(value=str(DEFAULT_RECORD_FPS))
        self._monitors = enumerate_monitors()
        self.record_target_var = tk.StringVar(value=CAPTURE_FULL_LABEL)
        self.record_codec_var = tk.StringVar(value=RECORD_CODEC_LABELS[DEFAULT_RECORD_CODEC])
        self.record_region: tuple[int, int, int, int] | None = None
        self.record_region_info_var = tk.StringVar(value="(영역 미지정)")

        def row_label(parent, text, **kw):
            return self._form_label(parent, text, **kw)

        # 음성 인식 / 번역 콤보를 같은 폭으로 두어야 뒤따르는 다운로드 버튼 열이 맞는다.
        MODEL_COMBO_W = 250
        MODEL_BTN_W = 160
        MODEL_DELETE_BTN_W = 74
        PATH_BTN1_W = 120   # 선택 / 같은 이름으로 (왼쪽 버튼 열)
        PATH_BTN2_W = 110   # 목록에 추가 / 폴더 선택 (오른쪽 버튼 열)

        # 경로 입력 행: grid로 라벨·입력칸·버튼 열을 정확히 맞춘다.
        path_form = ctk.CTkFrame(self.file_tab, fg_color="transparent")
        path_form.pack(fill="x", pady=(2, 0))
        path_form.grid_columnconfigure(0, minsize=self._label_w)
        path_form.grid_columnconfigure(1, weight=1)

        row_label(path_form, "원본 영상/음성").grid(row=0, column=0, sticky="w", pady=6)
        ctk.CTkEntry(path_form, textvariable=self.file_path_var, **self._entry_kw).grid(
            row=0, column=1, sticky="ew", padx=10, pady=6
        )
        self.select_media_btn = ctk.CTkButton(
            path_form, text="선택", command=self._select_media,
            width=PATH_BTN1_W, height=self._sec_btn_h, **self._ghost_btn,
        )
        self.select_media_btn.grid(row=0, column=2, sticky="ew", pady=6)
        self.add_file_btn = ctk.CTkButton(
            path_form, text="목록에 추가", command=self._add_input_to_queue,
            width=PATH_BTN2_W, height=self._sec_btn_h, **self._ghost_btn,
        )
        self.add_file_btn.grid(row=0, column=3, sticky="ew", padx=(8, 0), pady=6)

        row_label(path_form, "저장할 SRT / 폴더").grid(row=1, column=0, sticky="w", pady=6)
        ctk.CTkEntry(path_form, textvariable=self.output_path_var, **self._entry_kw).grid(
            row=1, column=1, sticky="ew", padx=10, pady=6
        )
        ctk.CTkButton(
            path_form, text="같은 이름으로", command=self._set_default_output,
            width=PATH_BTN1_W, height=self._sec_btn_h, **self._ghost_btn,
        ).grid(row=1, column=2, sticky="ew", pady=6)
        ctk.CTkButton(
            path_form, text="폴더 선택", command=self._select_output_dir,
            width=PATH_BTN2_W, height=self._sec_btn_h, **self._ghost_btn,
        ).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=6)

        # 시작 / 중지 / 결과 열기 행
        action_row = ctk.CTkFrame(self.file_tab, fg_color="transparent")
        action_row.pack(fill="x", pady=6)
        self.start_file_btn = ctk.CTkButton(
            action_row, text="자막 만들기 시작", command=self._start_file_job, **self._primary_btn_kw
        )
        self.cancel_file_btn = ctk.CTkButton(
            action_row, text="중지", command=self._cancel_file_job, state="disabled", **self._stop_btn_kw
        )
        self.start_file_btn.pack(side="left")
        self.cancel_file_btn.pack(side="left", padx=8)
        ctk.CTkButton(
            action_row, text="결과 열기", command=self._open_output, **self._open_btn_kw
        ).pack(side="right")
        ctk.CTkButton(
            action_row, text="폴더 열기", command=self._open_output_folder, **self._open_btn_kw
        ).pack(side="right", padx=8)
        self.retry_failed_btn = ctk.CTkButton(
            action_row, text="실패 재시도", command=self._retry_failed_files,
            state="disabled", **self._open_btn_kw
        )
        self.retry_failed_btn.pack(side="right")

        # 모델 선택 행: 콤보·다운로드 버튼 열을 위 경로 행과 같은 왼쪽 기준으로 맞춘다.
        model_form = ctk.CTkFrame(self.file_tab, fg_color="transparent")
        model_form.pack(fill="x", pady=(0, 2))
        model_form.grid_columnconfigure(0, minsize=self._label_w)

        row_label(model_form, "음성 인식 모델").grid(row=0, column=0, sticky="w", pady=6)
        whisper_holder, self.whisper_model_combo = self._make_option_menu(
            model_form,
            variable=self.whisper_model_var,
            values=[WHISPER_MODEL_LABELS[key] for key in WHISPER_MODEL_ORDER],
            command=self._on_whisper_model_selected,
            width=MODEL_COMBO_W,
        )
        whisper_holder.grid(row=0, column=1, sticky="w", padx=10, pady=6)
        asr_action_frame = ctk.CTkFrame(model_form, fg_color="transparent")
        asr_action_frame.grid(row=0, column=2, sticky="w", pady=6)
        self.prepare_asr_btn = ctk.CTkButton(
            asr_action_frame,
            text="위스퍼 모델 다운로드",
            command=self._prepare_asr_model,
            width=MODEL_BTN_W,
            height=self._sec_btn_h,
            **self._ghost_btn,
        )
        self.prepare_asr_btn.pack(side="left")
        self.delete_asr_btn = ctk.CTkButton(
            asr_action_frame,
            text="삭제",
            command=self._delete_file_asr_model,
            width=MODEL_DELETE_BTN_W,
            height=self._sec_btn_h,
            **self._danger_btn,
        )
        self.delete_asr_btn.pack(side="left", padx=(8, 0))
        row_label(model_form, "원본 언어").grid(row=0, column=3, sticky="w", padx=(18, 0), pady=6)
        file_lang_holder, self.file_source_lang_combo = self._make_option_menu(
            model_form,
            variable=self.source_lang_var,
            values=[SOURCE_LANG_LABELS[key] for key in SOURCE_LANG_ORDER],
            command=self._on_source_lang_selected,
            width=120,
        )
        file_lang_holder.grid(row=0, column=4, sticky="w", padx=10, pady=6)

        row_label(model_form, "번역 모델").grid(row=1, column=0, sticky="w", pady=6)
        translation_holder, self.translation_profile_combo = self._make_option_menu(
            model_form,
            variable=self.translation_profile_var,
            values=[TRANSLATION_PROFILE_LABELS[key] for key in TRANSLATION_PROFILE_ORDER],
            command=self._on_translation_profile_selected,
            width=MODEL_COMBO_W,
        )
        translation_holder.grid(row=1, column=1, sticky="w", padx=10, pady=6)
        translation_action_frame = ctk.CTkFrame(model_form, fg_color="transparent")
        translation_action_frame.grid(row=1, column=2, sticky="w", pady=6)
        self.prepare_translation_btn = ctk.CTkButton(
            translation_action_frame,
            text="번역 모델 다운로드",
            command=self._prepare_translation_model,
            width=MODEL_BTN_W,
            height=self._sec_btn_h,
            **self._ghost_btn,
        )
        self.prepare_translation_btn.pack(side="left")
        self.delete_translation_btn = ctk.CTkButton(
            translation_action_frame,
            text="삭제",
            command=self._delete_file_translation_model,
            width=MODEL_DELETE_BTN_W,
            height=self._sec_btn_h,
            **self._danger_btn,
        )
        self.delete_translation_btn.pack(side="left", padx=(8, 0))
        self._hint_label(
            model_form,
            text="고품질 LLM은 GPU·VRAM 사용 / 첫 선택 시 수 GB 다운로드",
            wraplength=460,
        ).grid(row=2, column=1, columnspan=4, sticky="w", padx=(10, 0), pady=(0, 6))

        file_status_row, self.file_asr_status_lbl, self.file_trans_status_lbl = self._make_model_status_row(self.file_tab)
        file_status_row.pack(fill="x", pady=(0, 4))

        queue_card, queue_body = self._make_card(self.file_tab, "작업할 파일 목록")
        queue_card.pack(fill="both", expand=True, pady=(4, 6))

        list_area = ctk.CTkFrame(queue_body, fg_color="transparent")
        list_area.pack(fill="both", expand=True)

        listbox_wrap = ctk.CTkFrame(list_area, fg_color=UI_SURFACE, corner_radius=4, border_width=1, border_color=UI_BORDER_STRONG)
        listbox_wrap.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.file_listbox_wrap = listbox_wrap
        self.file_listbox = tk.Listbox(
            listbox_wrap,
            height=6,
            selectmode="extended",
            exportselection=False,
            bg=UI_SURFACE,
            fg=UI_TEXT,
            font=(self._ui_font_family, 13),
            selectbackground=UI_ACCENT,
            selectforeground=UI_SURFACE,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=(8, 8), pady=8)
        self.file_list_scroll = ctk.CTkScrollbar(listbox_wrap, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=self.file_list_scroll.set)

        queue_button_frame = ctk.CTkFrame(list_area, fg_color="transparent")
        queue_button_frame.pack(side="left", fill="y", padx=(10, 0))

        def queue_btn(text, command):
            btn = ctk.CTkButton(queue_button_frame, text=text, command=command, width=110, height=32, **self._ghost_btn)
            btn.pack(fill="x", pady=(0, 6))
            return btn

        self.add_queue_btn = queue_btn("파일 추가", self._add_files_to_queue)
        self.add_folder_btn = queue_btn("폴더 추가", self._add_folder_to_queue)
        self.remove_queue_btn = queue_btn("선택 삭제", self._remove_selected_files)
        self.clear_queue_btn = queue_btn("전체 비우기", self._clear_file_queue)
        self.move_up_btn = queue_btn("위로", self._move_selected_up)
        self.move_down_btn = queue_btn("아래로", self._move_selected_down)

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
            self.delete_asr_btn,
            self.translation_profile_combo,
            self.prepare_translation_btn,
            self.delete_translation_btn,
        ]

        status_row = self._make_status_row(
            self.file_tab,
            [("파일 상태", self.file_status_var), ("전체 진행", self.batch_status_var)],
        )
        status_row.pack(fill="x", pady=(4, 4))

        self._refresh_file_queue()

        # Live tab
        self.live_status_var = tk.StringVar(value="대기 중")
        self.live_preview_var = tk.StringVar(value="실시간 한국어 자막이 여기에 표시됩니다.")
        self.show_overlay_var = tk.BooleanVar(value=True)
        self.live_save_srt_var = tk.BooleanVar(value=False)
        self.live_whisper_model_var = tk.StringVar(value=WHISPER_MODEL_LABELS[self.live_asr.model_name])
        self.live_translation_var = tk.StringVar(value=TRANSLATION_PROFILE_LABELS[self.live_translator.preferred])
        self.live_use_secondary_var = tk.BooleanVar(value=self.live_translator.opus_use_secondary)
        self.live_secondary_var = tk.StringVar(value=LIVE_SECONDARY_LABELS[self.live_translator.opus_secondary])

        source_card, source_body = self._make_card(self.live_tab, "실시간 소리 입력")
        source_card.pack(fill="x", pady=(2, 10))

        live_form = self._make_form_frame(source_body)
        live_form.pack(fill="x")
        live_form.grid_columnconfigure(1, weight=0)
        live_form.grid_columnconfigure(4, weight=1)

        self._form_label(live_form, "입력").grid(row=0, column=0, sticky="w", pady=4)
        live_source_group = ctk.CTkFrame(live_form, fg_color="transparent")
        live_source_group.grid(row=0, column=1, sticky="w", padx=10, pady=4)
        ctk.CTkRadioButton(
            live_source_group,
            text="PC 스피커 소리",
            variable=self.live_source_var,
            value=LIVE_SOURCE_SPEAKER,
            command=self._on_live_source_selected,
            font=self.font_base,
        ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(
            live_source_group,
            text="인터넷 URL 스트림",
            variable=self.live_source_var,
            value=LIVE_SOURCE_URL,
            command=self._on_live_source_selected,
            font=self.font_base,
        ).pack(side="left")
        self._form_label(live_form, "원본 언어", width=76).grid(row=0, column=2, sticky="w", padx=(12, 0), pady=4)
        live_lang_holder, self.live_source_lang_combo = self._make_option_menu(
            live_form,
            variable=self.source_lang_var,
            values=[SOURCE_LANG_LABELS[key] for key in SOURCE_LANG_ORDER],
            command=self._on_source_lang_selected,
            width=120,
        )
        live_lang_holder.grid(row=0, column=3, sticky="w", padx=10, pady=4)

        self._form_label(live_form, "URL").grid(row=1, column=0, sticky="w", pady=4)
        self.live_url_entry = ctk.CTkEntry(live_form, textvariable=self.live_url_var, **self._entry_kw)
        self.live_url_entry.grid(row=1, column=1, columnspan=4, sticky="ew", padx=10, pady=4)

        self._form_label(live_form, "음성 인식 모델").grid(row=2, column=0, sticky="w", pady=4)
        live_model_holder, self.live_whisper_model_combo = self._make_option_menu(
            live_form,
            variable=self.live_whisper_model_var,
            values=[WHISPER_MODEL_LABELS[key] for key in WHISPER_MODEL_ORDER],
            command=self._on_live_whisper_model_selected,
            width=MODEL_COMBO_W,
        )
        live_model_holder.grid(row=2, column=1, sticky="w", padx=10, pady=4)
        live_asr_action = ctk.CTkFrame(live_form, fg_color="transparent")
        live_asr_action.grid(row=2, column=2, sticky="w", pady=4)
        self.prepare_live_asr_btn = ctk.CTkButton(
            live_asr_action,
            text="위스퍼 모델 다운로드",
            command=self._prepare_live_asr_model,
            width=MODEL_BTN_W,
            height=self._sec_btn_h,
            **self._ghost_btn,
        )
        self.prepare_live_asr_btn.pack(side="left")
        self.delete_live_asr_btn = ctk.CTkButton(
            live_asr_action,
            text="삭제",
            command=self._delete_live_asr_model,
            width=MODEL_DELETE_BTN_W,
            height=self._sec_btn_h,
            **self._danger_btn,
        )
        self.delete_live_asr_btn.pack(side="left", padx=(8, 0))
        self._hint_label(live_form, "실시간은 빠른 tiny/base 권장", wraplength=360).grid(
            row=2, column=3, columnspan=2, sticky="ew", padx=(10, 10), pady=4
        )

        self._form_label(live_form, "번역 모델").grid(row=3, column=0, sticky="w", pady=4)
        live_trans_holder, self.live_translation_combo = self._make_option_menu(
            live_form,
            variable=self.live_translation_var,
            values=[TRANSLATION_PROFILE_LABELS[key] for key in TRANSLATION_PROFILE_ORDER],
            command=self._on_live_translation_profile_selected,
            width=MODEL_COMBO_W,
        )
        live_trans_holder.grid(row=3, column=1, sticky="w", padx=10, pady=4)
        live_trans_action = ctk.CTkFrame(live_form, fg_color="transparent")
        live_trans_action.grid(row=3, column=2, sticky="w", pady=4)
        self.prepare_live_translation_btn = ctk.CTkButton(
            live_trans_action,
            text="번역 모델 다운로드",
            command=self._prepare_live_translation_model,
            width=MODEL_BTN_W,
            height=self._sec_btn_h,
            **self._ghost_btn,
        )
        self.prepare_live_translation_btn.pack(side="left")
        self.delete_live_translation_btn = ctk.CTkButton(
            live_trans_action,
            text="삭제",
            command=self._delete_live_translation_model,
            width=MODEL_DELETE_BTN_W,
            height=self._sec_btn_h,
            **self._danger_btn,
        )
        self.delete_live_translation_btn.pack(side="left", padx=(8, 0))
        self._hint_label(live_form, "파일 탭과 동일 · LLM은 실시간이 느려질 수 있음", wraplength=360).grid(
            row=3, column=3, columnspan=2, sticky="ew", padx=(10, 10), pady=4
        )

        # OPUS-MT(영어 1차) 선택 시 영어→한국어 2차 번역 설정
        self._form_label(live_form, "OPUS 2차 번역").grid(row=4, column=0, sticky="w", pady=4)
        self.live_secondary_check = ctk.CTkCheckBox(
            live_form,
            text="영어→한국어 2차 번역",
            variable=self.live_use_secondary_var,
            onvalue=True,
            offvalue=False,
            command=self._on_live_secondary_toggled,
            font=self.font_base,
        )
        self.live_secondary_check.grid(row=4, column=1, sticky="w", padx=10, pady=4)
        live_secondary_holder, self.live_secondary_combo = self._make_option_menu(
            live_form,
            variable=self.live_secondary_var,
            values=[LIVE_SECONDARY_LABELS[key] for key in LIVE_SECONDARY_ORDER],
            command=self._on_live_secondary_selected,
            width=MODEL_COMBO_W,
        )
        live_secondary_holder.grid(row=4, column=2, sticky="w", padx=(0, 10), pady=4)
        self._hint_label(live_form, "OPUS 선택 시에만 적용(일·중→영→한)", wraplength=300).grid(
            row=4, column=3, columnspan=2, sticky="ew", padx=(0, 10), pady=4
        )
        self._update_live_secondary_state()
        self._on_live_source_selected()  # 기본값(PC 스피커)에 맞춰 URL 칸 초기 활성 상태를 정한다.

        live_status_row, self.live_asr_status_lbl, self.live_trans_status_lbl = self._make_model_status_row(source_body)
        live_status_row.pack(fill="x", pady=(2, 0))

        live_row1 = ctk.CTkFrame(self.live_tab, fg_color="transparent")
        live_row1.pack(fill="x", pady=6)
        self.start_live_btn = ctk.CTkButton(
            live_row1, text="시작", command=self._start_live_job, **self._primary_btn_kw
        )
        self.stop_live_btn = ctk.CTkButton(
            live_row1, text="중지", command=self._stop_live_job, state="disabled", **self._stop_btn_kw
        )
        self.start_live_btn.pack(side="left")
        self.stop_live_btn.pack(side="left", padx=8)
        ctk.CTkButton(
            live_row1, text="결과 열기", command=self._open_output, **self._open_btn_kw
        ).pack(side="right")

        live_row2 = ctk.CTkFrame(self.live_tab, fg_color="transparent")
        live_row2.pack(fill="x", pady=6)
        ctk.CTkCheckBox(
            live_row2,
            text="자막 창을 항상 위에 표시",
            variable=self.show_overlay_var,
            onvalue=True,
            offvalue=False,
            command=self._apply_live_overlay_preference,
            font=self.font_base,
        ).pack(side="left")
        ctk.CTkCheckBox(
            live_row2,
            text="자막 파일(.srt)로도 저장",
            variable=self.live_save_srt_var,
            onvalue=True,
            offvalue=False,
            font=self.font_base,
        ).pack(side="left", padx=(20, 0))
        self.toggle_overlay_btn = ctk.CTkButton(
            live_row2,
            text="자막창 보이기",
            command=self._toggle_live_overlay,
            width=120,
            height=self._sec_btn_h,
            state="disabled",
            **self._ghost_btn,
        )
        self.toggle_overlay_btn.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(
            self.live_tab, text="실시간 자막 미리보기", font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w"
        ).pack(fill="x", pady=(12, 2))
        preview_card = ctk.CTkFrame(
            self.live_tab, fg_color=UI_SURFACE, corner_radius=6, border_width=1, border_color=UI_BORDER_STRONG
        )
        preview_card.pack(fill="x")
        self.live_preview = self._live_label(
            preview_card,
            self.live_preview_var,
            font=self.font_caption,
            justify="left",
            anchor="w",
            wraplength=820,
            text_color=UI_TEXT,
        )
        self.live_preview.pack(fill="x", padx=16, pady=16)
        # 창 폭이 넓어지면 미리보기 줄바꿈 폭도 넓혀 넓은 화면을 활용한다(안쪽 여백 32 + 여유).
        preview_card.bind(
            "<Configure>",
            lambda event: self.live_preview.configure(wraplength=max(300, int(event.width) - 40)),
        )
        live_status_line = self._make_status_row(self.live_tab, [("실시간 상태", self.live_status_var)])
        live_status_line.pack(fill="x", pady=6)

        # Screen record tab
        record_card, record_body = self._make_card(self.record_tab, "녹화 설정")
        record_card.pack(fill="x", pady=(2, 10))

        record_form = self._make_form_frame(record_body)
        record_form.pack(fill="x")
        record_form.grid_columnconfigure(1, weight=0)
        record_form.grid_columnconfigure(3, weight=1)

        self._form_label(record_form, "저장할 MKV").grid(row=0, column=0, sticky="w", pady=4)
        self.record_output_entry = ctk.CTkEntry(record_form, textvariable=self.record_output_var, **self._entry_kw)
        self.record_output_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=10, pady=4)
        self.browse_record_btn = ctk.CTkButton(
            record_form, text="찾아보기", command=self._select_record_output,
            width=110, height=self._sec_btn_h, **self._ghost_btn
        )
        self.browse_record_btn.grid(row=0, column=4, sticky="ew", pady=4)

        # 캡처 대상: 전체 화면 / 모니터별 / 영역 지정
        self._form_label(record_form, "캡처 대상").grid(row=1, column=0, sticky="w", pady=4)
        target_holder, self.record_target_combo = self._make_option_menu(
            record_form,
            variable=self.record_target_var,
            values=self._capture_target_labels(),
            command=self._on_capture_target_selected,
            width=MODEL_COMBO_W,
        )
        target_holder.grid(row=1, column=1, sticky="w", padx=10, pady=4)
        self.select_region_btn = ctk.CTkButton(
            record_form, text="영역 선택", command=self._select_capture_region_dialog,
            width=110, height=self._sec_btn_h, state="disabled", **self._ghost_btn,
        )
        self.select_region_btn.grid(row=1, column=2, sticky="w", pady=4)
        self._live_label(
            record_form, self.record_region_info_var, font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w"
        ).grid(row=1, column=3, columnspan=2, sticky="ew", padx=(8, 10), pady=4)

        # 녹화 형식: 무손실(FFV1) / 고화질 압축(H.264)
        self._form_label(record_form, "녹화 형식").grid(row=2, column=0, sticky="w", pady=4)
        codec_holder, self.record_codec_combo = self._make_option_menu(
            record_form,
            variable=self.record_codec_var,
            values=[RECORD_CODEC_LABELS[key] for key in RECORD_CODEC_ORDER],
            width=MODEL_COMBO_W,
        )
        codec_holder.grid(row=2, column=1, sticky="w", padx=10, pady=4)
        self._hint_label(record_form, "무손실=원본화질·대용량, H.264=용량 1/10~", wraplength=360).grid(
            row=2, column=2, columnspan=3, sticky="ew", padx=(6, 10), pady=4
        )

        self._form_label(record_form, "초당 화면 수").grid(row=3, column=0, sticky="w", pady=4)
        record_fps_holder, self.record_fps_combo = self._make_option_menu(
            record_form,
            variable=self.record_fps_var,
            values=["24", "30", "60"],
            width=90,
        )
        record_fps_holder.grid(row=3, column=1, sticky="w", padx=10, pady=4)
        self.record_audio_check = ctk.CTkCheckBox(
            record_form,
            text="PC 스피커 소리도 녹음",
            variable=self.record_audio_var,
            onvalue=True,
            offvalue=False,
            font=self.font_base,
        )
        self.record_audio_check.grid(row=3, column=2, columnspan=3, sticky="w", padx=(6, 10), pady=4)

        record_row2 = ctk.CTkFrame(self.record_tab, fg_color="transparent")
        record_row2.pack(fill="x", pady=6)
        self.start_record_btn = ctk.CTkButton(
            record_row2, text="녹화 시작", command=self._start_record_job, **self._primary_btn_kw
        )
        self.stop_record_btn = ctk.CTkButton(
            record_row2, text="중지", command=self._stop_record_job, state="disabled", **self._stop_btn_kw
        )
        self.start_record_btn.pack(side="left")
        self.stop_record_btn.pack(side="left", padx=8)
        ctk.CTkButton(
            record_row2, text="결과 열기", command=self._open_output, **self._open_btn_kw
        ).pack(side="right")

        record_status_line = self._make_status_row(self.record_tab, [("녹화 상태", self.record_status_var)])
        record_status_line.pack(fill="x", pady=6)
        self._record_controls = [
            self.record_output_entry,
            self.browse_record_btn,
            self.record_target_combo,
            self.select_region_btn,
            self.record_codec_combo,
            self.record_fps_combo,
            self.record_audio_check,
            self.start_record_btn,
            self.stop_record_btn,
        ]

        self._build_help_tab()

        # Bottom common: 오른쪽 영역 아래 칸에 실시간 로그를 둔다.
        common = ctk.CTkFrame(
            self._right_panel, fg_color=UI_SURFACE, corner_radius=6, border_width=1, border_color=UI_CARD_BORDER
        )
        common.grid(row=1, column=0, sticky="nsew")
        self._common_log_panel = common

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="대기 중")
        self.eta_var = tk.StringVar(value="예상 시간: 대기 중")
        self.log_header_detail_var = tk.StringVar()
        self.status_var.trace_add("write", lambda *_: self._sync_log_header())
        self.eta_var.trace_add("write", lambda *_: self._sync_log_header())
        self._sync_log_header()

        log_header = ctk.CTkFrame(common, fg_color="transparent")
        log_header.pack(fill="x", padx=14, pady=(10, 6))
        ctk.CTkLabel(log_header, text="실시간 로그", font=self.font_section, text_color=UI_TEXT, anchor="w").pack(
            side="left"
        )
        self._live_label(
            log_header,
            self.log_header_detail_var,
            font=self.font_base,
            text_color=UI_TEXT_MUTED,
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.progress = ctk.CTkProgressBar(common, height=10, corner_radius=3, fg_color=UI_BORDER)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=14, pady=(0, 8))

        self.log_text = ctk.CTkTextbox(
            common,
            height=170,
            wrap="word",
            fg_color=UI_SURFACE,
            text_color=UI_TEXT,
            border_color=UI_BORDER_STRONG,
            border_width=1,
            corner_radius=4,
            font=self.font_base,
        )
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._configure_log_indent()
        self.log_text.configure(state="disabled")

        self._append_log("앱을 시작했습니다.")
        self._refresh_model_status()
        self._show_tab("file")

    def _show_tab(self, key: str) -> None:
        host = self._tab_hosts.get(key)
        if host is None:
            return
        if hasattr(self, "_common_log_panel"):
            if key == "help":
                self._common_log_panel.grid_remove()
                self._content_panel.grid_configure(rowspan=2, pady=0)
            else:
                self._content_panel.grid_configure(rowspan=1, pady=(0, 8))
                self._common_log_panel.grid(row=1, column=0, sticky="nsew")
        host.tkraise()
        for nav_key, btn in self._nav_buttons.items():
            if nav_key == key:
                btn.configure(fg_color=UI_ACCENT_SOFT, text_color=UI_ACCENT, font=self.font_nav_bold)
            else:
                btn.configure(fg_color="transparent", text_color=UI_TEXT, font=self.font_base)

    def _sync_log_header(self) -> None:
        status = str(self.status_var.get()).strip() or "대기 중"
        eta = str(self.eta_var.get()).strip() or "예상 시간: 대기 중"
        self.log_header_detail_var.set(f"({status}, {eta})")

    def _begin_timed_progress(self) -> None:
        self.progress_started_at = time.time()
        self._set_progress(0)
        self.eta_var.set("예상 시간: 계산 중")

    def _update_eta(self, progress: float) -> None:
        self.eta_var.set(build_eta_text(progress, self.progress_started_at))

    def _set_progress(self, value: float) -> None:
        value = max(0.0, min(100.0, float(value)))
        self.progress_var.set(value)
        self.progress.set(value / 100.0)

    def _set_eta_status(self, text: str) -> None:
        self.eta_var.set(f"예상 시간: {text}")

    def _handle_download_progress(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        percent = float(data.get("percent") or 0.0)
        desc = str(data.get("desc") or "").strip()
        total = data.get("total")
        n = data.get("n") or 0
        eta_seconds = data.get("eta_seconds")

        label = desc if desc else "모델 파일"
        if len(label) > 40:
            label = "…" + label[-39:]

        size_text = ""
        if total:
            size_text = f" ({format_file_size(int(n))}/{format_file_size(int(total))})"
        self.status_var.set(f"모델 다운로드 중: {label} {percent:.0f}%{size_text}")
        self._set_progress(percent)

        if percent < 3:
            self.eta_var.set("예상 시간: 계산 중")
        elif eta_seconds is not None:
            finish = time.strftime("%H:%M", time.localtime(time.time() + eta_seconds))
            self.eta_var.set(
                f"예상 시간: 약 {format_duration_short(eta_seconds)} 남음 · 완료 예상 {finish}"
            )

    def _refresh_resource_plan_label(self) -> None:
        self.resource_plan_var.set(build_resource_settings(self._resource_mode_from_selection()).summary)

    def _configure_log_indent(self) -> None:
        # 로그가 길어 자동 줄바꿈될 때, 둘째 줄부터 시각("HH:MM:SS - ") 폭만큼 들여써
        # 시각 컬럼을 침범하지 않게 한다(hanging indent).
        try:
            import tkinter.font as tkfont

            textbox = self.log_text._textbox
            font = tkfont.Font(font=textbox.cget("font"))
            self._log_indent_px = font.measure("00:00:00 - ")
            textbox.tag_configure("logline", lmargin2=self._log_indent_px)
        except Exception:
            pass

    def _append_log(self, msg: object) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n", "logline")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_system_specs(self) -> None:
        self.system_cpu_var.set("확인 중 ...")
        self.system_ram_var.set("확인 중 ...")
        self.system_vga_var.set("확인 중 ...")

        def worker() -> None:
            try:
                self._emit("system_specs", collect_system_specs())
            except Exception as exc:
                self._emit("log", f"시스템 사양 확인 실패: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _clear_file_preview(self) -> None:
        # 별도 미리보기 박스를 없앴으므로 작업 기록은 지우지 않고 그대로 둡니다.
        pass

    def _append_file_preview(self, line: object) -> None:
        text = str(line).strip()
        if not text:
            return
        self._append_log(f"[추출 미리보기] {text}")

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
            elif kind == "system_specs":
                specs = payload if isinstance(payload, dict) else {}
                self.system_cpu_var.set(str(specs.get("cpu", "알 수 없음")))
                self.system_ram_var.set(str(specs.get("ram", "알 수 없음")))
                self.system_vga_var.set(str(specs.get("vga", "알 수 없음")))
                self._refresh_resource_plan_label()
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
            elif kind == "live_worker_error":
                # 실시간 발화 처리 중 일시적 오류. 치명적 오류와 달리 모달 팝업을 띄우거나
                # 실행 상태를 해제하지 않고, 로그와 상태줄에만 표시해 자막이 계속 흐르게 한다.
                self._append_log(payload)
                self.live_status_var.set(str(payload))
            elif kind == "record_status":
                self.record_status_var.set(str(payload))
            elif kind == "progress":
                progress = float(payload)
                self._set_progress(progress)
                self._update_eta(progress)
            elif kind == "download_progress":
                self._handle_download_progress(payload)
            elif kind == "eta_status":
                self.progress_started_at = None
                self._set_eta_status(str(payload))
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
            elif kind == "live_subtitle_output":
                self.last_output = Path(str(payload))
                self._append_log(f"실시간 자막을 저장하는 중: {self.last_output}")
            elif kind == "file_retryable":
                self._failed_sources = list(payload) if isinstance(payload, list) else []
                self._update_retry_button()
            elif kind == "busy_file":
                self._set_file_busy(bool(payload))
            elif kind == "busy_live":
                self._set_live_busy(bool(payload))
            elif kind == "busy_record":
                self._set_record_busy(bool(payload))
            elif kind == "error":
                self._append_log(payload)
                if self.progress_started_at is not None:
                    self.progress_started_at = None
                    self._set_eta_status("계산 불가")
                if self._is_live_running or self._is_file_running or self._is_record_running:
                    messagebox.showerror(APP_NAME, str(payload))
            elif kind == "done":
                self._append_log(payload)
                if self.progress_started_at is not None or self.progress_var.get() >= 100:
                    self.progress_started_at = None
                    self._set_eta_status("완료")
                messagebox.showinfo(APP_NAME, str(payload))
            elif kind == "subtitle_output":
                path = Path(str(payload))
                self.last_output = path
                self.output_path_var.set(str(path))
            elif kind == "asr_prepare_done":
                payload_data = payload
                self._set_asr_preparing(False)
                self._refresh_model_status()
                if isinstance(payload_data, dict) and payload_data.get("message"):
                    self._append_log(payload_data["message"])
            elif kind == "translation_prepare_done":
                payload_data = payload
                self._set_translation_preparing(False)
                self._refresh_model_status()
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
                state = "normal"
            widget.configure(state=state)
        if self._is_asr_preparing:
            self.prepare_asr_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
        if self._is_translation_preparing:
            self.prepare_translation_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
        self._refresh_resource_mode_control_state()
        self._refresh_model_action_states()
        self._update_retry_button()

    def _update_retry_button(self) -> None:
        if not hasattr(self, "retry_failed_btn"):
            return
        can_retry = bool(self._failed_sources) and not self._is_file_running
        self.retry_failed_btn.configure(state="normal" if can_retry else "disabled")

    def _refresh_resource_mode_control_state(self) -> None:
        if not hasattr(self, "resource_mode_combo"):
            return
        busy = (
            self._is_file_running
            or self._is_live_running
            or self._is_record_running
            or self._is_asr_preparing
            or self._is_translation_preparing
        )
        self.resource_mode_combo.configure(state="disabled" if busy else "normal")

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
        self._refresh_resource_mode_control_state()
        self._refresh_model_action_states()

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
        self._refresh_resource_mode_control_state()
        self._refresh_model_action_states()

    def _set_live_busy(self, busy: bool) -> None:
        self._is_live_running = busy
        self.start_live_btn.configure(state="disabled" if busy else "normal")
        self.stop_live_btn.configure(state="normal" if busy else "disabled")
        if not busy and self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        self._sync_live_overlay_button()
        self._refresh_resource_mode_control_state()
        self._refresh_model_action_states()

    def _live_overlay_is_visible(self) -> bool:
        if self.overlay is None:
            return False
        try:
            return bool(self.overlay.winfo_exists()) and self.overlay.is_visible()
        except tk.TclError:
            return False

    def _show_live_overlay(self) -> None:
        if self.overlay is None:
            self.overlay = LiveSubtitleOverlay(self)
        else:
            try:
                if not self.overlay.winfo_exists():
                    self.overlay = LiveSubtitleOverlay(self)
            except tk.TclError:
                self.overlay = LiveSubtitleOverlay(self)

        preview_text = self.live_preview_var.get().strip()
        if preview_text:
            self.overlay.set_text(preview_text)
        self.overlay.show()

    def _hide_live_overlay(self) -> None:
        if self.overlay is None:
            return
        try:
            if self.overlay.winfo_exists():
                self.overlay.hide()
        except tk.TclError:
            self.overlay = None

    def _apply_live_overlay_preference(self) -> None:
        if not self._is_live_running:
            self._sync_live_overlay_button()
            return
        if self.show_overlay_var.get():
            self._show_live_overlay()
        else:
            self._hide_live_overlay()
        self._sync_live_overlay_button()

    def _toggle_live_overlay(self) -> None:
        if not self._is_live_running:
            return
        self.show_overlay_var.set(not self._live_overlay_is_visible())
        self._apply_live_overlay_preference()

    def _sync_live_overlay_button(self) -> None:
        if not hasattr(self, "toggle_overlay_btn"):
            return
        visible = self._live_overlay_is_visible()
        self.toggle_overlay_btn.configure(
            text="자막창 숨기기" if visible else "자막창 보이기",
            state="normal" if self._is_live_running else "disabled",
        )

    def _set_record_busy(self, busy: bool) -> None:
        self._is_record_running = busy
        for widget in self._record_controls:
            state = "disabled" if busy else "normal"
            if widget is self.stop_record_btn:
                state = "normal" if busy else "disabled"
            if widget is self.record_fps_combo and not busy:
                state = "normal"
            widget.configure(state=state)
        if self._is_asr_preparing or self._is_translation_preparing:
            self.start_record_btn.configure(state="disabled")
        self._refresh_resource_mode_control_state()
        self._refresh_model_action_states()

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

    def _capture_target_labels(self) -> list[str]:
        labels = [CAPTURE_FULL_LABEL]
        for index, monitor in enumerate(self._monitors, start=1):
            tag = ", 주" if monitor["primary"] else ""
            labels.append(f"모니터 {index} ({monitor['w']}×{monitor['h']}{tag})")
        labels.append(CAPTURE_REGION_LABEL)
        return labels

    def _on_capture_target_selected(self, _event: object | None = None) -> None:
        is_region = self.record_target_var.get() == CAPTURE_REGION_LABEL
        try:
            self.select_region_btn.configure(state="normal" if is_region else "disabled")
        except Exception:
            pass
        # 영역 지정을 고르면 곧바로 드래그 선택창을 띄운다(재선택 시에도 항상 새로 지정).
        if is_region:
            self._select_capture_region_dialog()

    def _resolve_capture_region(self) -> tuple[tuple[int, int, int, int] | None, bool]:
        """선택한 캡처 대상으로부터 (region, ok)를 돌려준다. ok=False면 영역 미지정."""
        label = self.record_target_var.get()
        if label == CAPTURE_FULL_LABEL:
            return None, True
        if label == CAPTURE_REGION_LABEL:
            return self.record_region, self.record_region is not None
        for index, monitor in enumerate(self._monitors, start=1):
            if label.startswith(f"모니터 {index} ("):
                return (monitor["x"], monitor["y"], monitor["w"], monitor["h"]), True
        return None, True

    def _select_capture_region_dialog(self) -> None:
        # 가상 데스크톱 전체를 덮는 반투명 창에서 마우스 드래그로 녹화 영역을 고른다.
        monitors = self._monitors or enumerate_monitors()
        if monitors:
            vx = min(m["x"] for m in monitors)
            vy = min(m["y"] for m in monitors)
            vr = max(m["x"] + m["w"] for m in monitors)
            vb = max(m["y"] + m["h"] for m in monitors)
        else:
            vx, vy = 0, 0
            vr, vb = self.winfo_screenwidth(), self.winfo_screenheight()
        vw, vh = vr - vx, vb - vy

        overlay = tk.Toplevel(self)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        try:
            overlay.attributes("-alpha", 0.3)
        except Exception:
            pass
        overlay.configure(bg=OVERLAY_BG)
        overlay.geometry(f"{vw}x{vh}+{vx}+{vy}")

        canvas = tk.Canvas(overlay, cursor="cross", bg=OVERLAY_BG, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_text(
            vw // 2, 30, text="드래그로 녹화할 영역을 지정하세요 (ESC: 취소)",
            fill=OVERLAY_FG, font=(self._ui_font_family, 16),
        )

        state: dict[str, object] = {"x0": 0, "y0": 0, "rect": None, "result": None}

        def on_press(event: object) -> None:
            state["x0"], state["y0"] = event.x_root, event.y_root
            if state["rect"] is not None:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                event.x_root - vx, event.y_root - vy, event.x_root - vx, event.y_root - vy,
                outline=UI_ACCENT, width=2,
            )

        def on_drag(event: object) -> None:
            if state["rect"] is not None:
                canvas.coords(
                    state["rect"], state["x0"] - vx, state["y0"] - vy, event.x_root - vx, event.y_root - vy
                )

        def on_release(event: object) -> None:
            x0, y0 = int(state["x0"]), int(state["y0"])
            x1, y1 = int(event.x_root), int(event.y_root)
            left, top = min(x0, x1), min(y0, y1)
            width, height = abs(x1 - x0), abs(y1 - y0)
            if width >= 16 and height >= 16:
                state["result"] = (left, top, width, height)
            overlay.destroy()

        def on_cancel(_event: object | None = None) -> None:
            overlay.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", on_cancel)
        # 두 번째 이후 호출에서도 창이 확실히 앞에 뜨고 입력을 받도록 매핑 완료 후 grab한다.
        overlay.update_idletasks()
        overlay.lift()
        overlay.focus_force()
        try:
            overlay.grab_set()
        except Exception:
            pass
        self.wait_window(overlay)

        result = state["result"]
        if result is not None:
            self.record_region = result  # type: ignore[assignment]
            x, y, w, h = result
            self.record_region_info_var.set(f"{w}×{h} @({x},{y})")
        elif self.record_region is None:
            # 취소했고 기존 영역도 없으면 전체 화면으로 되돌린다.
            self.record_target_var.set(CAPTURE_FULL_LABEL)
            self.select_region_btn.configure(state="disabled")
            self.record_region_info_var.set("(영역 미지정)")

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
        tools.append(
            "음성 인식 모델: 파일 " + self.asr.model_label() + " / 실시간 " + self.live_asr.model_label()
        )
        tools.append(
            "번역 방식: 파일 " + self.translator.profile_label() + " / 실시간 " + self.live_translator.profile_label()
        )
        tools.append("자원 사용: " + build_resource_settings(self._resource_mode_from_selection()).summary)
        for line in tools:
            self._append_log(line)

    def _resource_mode_id_from_label(self, label: str) -> str:
        for key, value in RESOURCE_MODE_LABELS.items():
            if value == label:
                return key
        return DEFAULT_RESOURCE_MODE

    def _resource_mode_from_selection(self) -> str:
        return self._resource_mode_id_from_label(self.resource_mode_var.get())

    def _apply_resource_mode_from_selection(self) -> None:
        mode = self._resource_mode_from_selection()
        self.asr.set_resource_mode(mode)
        self.live_asr.set_resource_mode(mode)
        self.translator.set_resource_mode(mode)
        self.live_translator.set_resource_mode(mode)
        self._refresh_resource_plan_label()

    def _on_resource_mode_selected(self, _event: object | None = None) -> None:
        if (
            self._is_file_running
            or self._is_live_running
            or self._is_asr_preparing
            or self._is_translation_preparing
        ):
            current = self.asr.resource_mode
            self.resource_mode_var.set(RESOURCE_MODE_LABELS.get(current, RESOURCE_MODE_LABELS[DEFAULT_RESOURCE_MODE]))
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 자원 사용 방식을 바꿔 주세요.")
            return

        self._apply_resource_mode_from_selection()
        self._append_log(f"자원 사용 방식을 설정했습니다: {self.asr.resource_summary()}")

    @staticmethod
    def _strip_status_prefix(label: str) -> str:
        for prefix in ("✓ ", "↓ "):
            if label.startswith(prefix):
                return label[len(prefix):]
        return label

    @staticmethod
    def _decorate_label(label: str, ready: bool) -> str:
        return ("✓ " if ready else "↓ ") + label

    def _whisper_model_id_from_label(self, label: str) -> str:
        label = self._strip_status_prefix(label)
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
        self._refresh_model_status()
        self._append_log(f"음성 인식 모델을 설정했습니다: {self.asr.model_label(model_name)}")

    def _live_whisper_model_from_selection(self) -> str:
        return self._whisper_model_id_from_label(self.live_whisper_model_var.get())

    def _on_live_whisper_model_selected(self, _event: object | None = None) -> None:
        if self._is_file_running or self._is_live_running or self._is_asr_preparing:
            self.live_whisper_model_var.set(self.live_asr.model_label())
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 모델을 바꿔 주세요.")
            return
        model_name = self._live_whisper_model_from_selection()
        self.live_asr.set_model(model_name)
        self._refresh_model_status()
        self._append_log(f"실시간 음성 인식 모델을 설정했습니다: {self.live_asr.model_label(model_name)}")

    def _live_translation_profile_from_selection(self) -> str:
        label = self._strip_status_prefix(self.live_translation_var.get())
        for key, value in TRANSLATION_PROFILE_LABELS.items():
            if value == label:
                return key
        return "balanced"

    def _on_live_translation_profile_selected(self, _event: object | None = None) -> None:
        if self._is_live_running or self._is_translation_preparing:
            self.live_translation_var.set(
                TRANSLATION_PROFILE_LABELS.get(self.live_translator.preferred, TRANSLATION_PROFILE_LABELS["balanced"])
            )
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 번역 모델을 바꿔 주세요.")
            return
        profile = self._live_translation_profile_from_selection()
        self.live_translator.set_profile(profile)
        self._update_live_secondary_state()
        self._refresh_model_status()
        label = TRANSLATION_PROFILE_LABELS.get(profile, self.live_translator.profile_label(profile))
        self._append_log(f"실시간 번역 모델을 설정했습니다: {label}")

    def _update_live_secondary_state(self) -> None:
        # OPUS-MT를 골랐을 때만 2차 번역 체크박스·드롭다운을 활성화한다.
        is_opus = self.live_translator.preferred == "opus"
        state = "normal" if is_opus else "disabled"
        for widget in (getattr(self, "live_secondary_check", None), getattr(self, "live_secondary_combo", None)):
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _refresh_model_status(self) -> None:
        # 선택된 음성 인식·번역 모델이 이미 받아져 있는지 각 탭의 상태 줄에 표시한다.
        def apply(label_widget, ready: bool) -> None:
            if label_widget is None:
                return
            text = "✓ 사용 가능" if ready else "↓ 다운로드 필요"
            color = UI_OK if ready else UI_WARN
            try:
                label_widget.configure(text=text, text_color=color)
            except Exception:
                pass

        try:
            apply(getattr(self, "file_asr_status_lbl", None), self.asr.is_model_ready(self._whisper_model_from_selection()))
            apply(getattr(self, "file_trans_status_lbl", None), self.translator.is_profile_ready(self._translation_profile_from_selection()))
            apply(getattr(self, "live_asr_status_lbl", None), self.live_asr.is_model_ready(self._live_whisper_model_from_selection()))
            apply(getattr(self, "live_trans_status_lbl", None), self.live_translator.is_profile_ready(self._live_translation_profile_from_selection()))
        except Exception:
            pass
        self._refresh_model_dropdowns()
        self._refresh_model_action_states()

    def _refresh_model_dropdowns(self) -> None:
        # 각 드롭다운 항목 앞에 ✓(사용 가능)/↓(다운로드 필요) 표시를 붙여 갱신한다.
        def apply_combo(combo, var, order, labels, ready_fn, current_key) -> None:
            if combo is None:
                return
            try:
                combo.configure(values=[self._decorate_label(labels[k], ready_fn(k)) for k in order])
                if current_key in labels:
                    var.set(self._decorate_label(labels[current_key], ready_fn(current_key)))
            except Exception:
                pass

        apply_combo(
            getattr(self, "whisper_model_combo", None), self.whisper_model_var,
            WHISPER_MODEL_ORDER, WHISPER_MODEL_LABELS, self.asr.is_model_ready,
            self._whisper_model_from_selection(),
        )
        apply_combo(
            getattr(self, "translation_profile_combo", None), self.translation_profile_var,
            TRANSLATION_PROFILE_ORDER, TRANSLATION_PROFILE_LABELS, self.translator.is_profile_ready,
            self._translation_profile_from_selection(),
        )
        apply_combo(
            getattr(self, "live_whisper_model_combo", None), self.live_whisper_model_var,
            WHISPER_MODEL_ORDER, WHISPER_MODEL_LABELS, self.live_asr.is_model_ready,
            self._live_whisper_model_from_selection(),
        )
        apply_combo(
            getattr(self, "live_translation_combo", None), self.live_translation_var,
            TRANSLATION_PROFILE_ORDER, TRANSLATION_PROFILE_LABELS, self.live_translator.is_profile_ready,
            self._live_translation_profile_from_selection(),
        )

    def _refresh_model_action_states(self) -> None:
        busy = (
            self._is_file_running
            or self._is_live_running
            or self._is_record_running
            or self._is_asr_preparing
            or self._is_translation_preparing
        )
        state = "disabled" if busy else "normal"
        for name in (
            "prepare_asr_btn",
            "delete_asr_btn",
            "prepare_live_asr_btn",
            "delete_live_asr_btn",
            "prepare_translation_btn",
            "delete_translation_btn",
            "prepare_live_translation_btn",
            "delete_live_translation_btn",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _model_change_busy(self) -> bool:
        return (
            self._is_file_running
            or self._is_live_running
            or self._is_record_running
            or self._is_asr_preparing
            or self._is_translation_preparing
        )

    def _confirm_model_delete(self, label: str) -> bool:
        return messagebox.askyesno(
            APP_NAME,
            (
                f"{label}\n\n"
                "내려받은 파일만 삭제합니다.\n"
                "목록의 모델 이름은 그대로 남고, 삭제 후에는 다시 다운로드할 수 있게 표시됩니다.\n\n"
                "삭제할까요?"
            ),
        )

    def _after_model_deleted(self, label: str, removed_bytes: int, removed_paths: list[Path]) -> None:
        self._refresh_model_status()
        if removed_paths:
            message = f"{label} 파일 삭제 완료: {format_file_size(removed_bytes)}"
        else:
            message = f"{label}은 이미 삭제된 상태입니다."
        self._append_log(message)
        messagebox.showinfo(APP_NAME, message)

    def _delete_file_asr_model(self) -> None:
        self._delete_asr_model(self._whisper_model_from_selection())

    def _delete_live_asr_model(self) -> None:
        self._delete_asr_model(self._live_whisper_model_from_selection())

    def _delete_asr_model(self, model_name: str) -> None:
        if self._model_change_busy():
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 모델을 삭제해 주세요.")
            return
        label = self.asr.model_label(model_name)
        if not self._confirm_model_delete(f"음성 인식 모델: {label}"):
            return

        if self.asr.model_name == model_name:
            self.asr.unload()
        if self.live_asr.model_name == model_name:
            self.live_asr.unload()
        try:
            removed_bytes, removed_paths = delete_whisper_model_files(model_name)
            self._after_model_deleted(f"음성 인식 모델 {label}", removed_bytes, removed_paths)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"음성 인식 모델 삭제 실패: {exc}")
            self._append_log(f"음성 인식 모델 삭제 실패: {exc}")

    def _delete_file_translation_model(self) -> None:
        self._delete_translation_profile(self._translation_profile_from_selection())

    def _delete_live_translation_model(self) -> None:
        self._delete_translation_profile(self._live_translation_profile_from_selection())

    def _delete_translation_profile(self, profile: str) -> None:
        if self._model_change_busy():
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 번역 모델을 삭제해 주세요.")
            return
        steps = TRANSLATION_PROFILE_ORDER_STEPS.get(profile, ())
        if not steps:
            return
        engine, model = steps[0]
        label = TRANSLATION_PROFILE_LABELS.get(profile, profile)
        if not self._confirm_model_delete(f"번역 모델: {label}"):
            return

        try:
            self.translator.release_runtime_cache(engine, model)
            self.live_translator.release_runtime_cache(engine, model)
            if engine == "llama" and model is not None:
                removed_bytes, removed_paths = delete_llm_model_files(model)
            elif engine == "nllb" and model is not None:
                removed_bytes, removed_paths = delete_nllb_model_files(model)
            elif engine == "opus":
                removed_bytes, removed_paths = delete_opus_model_files()
            elif engine == "argos":
                removed_bytes, removed_paths = delete_argos_package_files()
            else:
                removed_bytes, removed_paths = 0, []
            self._after_model_deleted(f"번역 모델 {label}", removed_bytes, removed_paths)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"번역 모델 삭제 실패: {exc}")
            self._append_log(f"번역 모델 삭제 실패: {exc}")

    def _live_secondary_from_selection(self) -> str:
        label = self.live_secondary_var.get()
        for key, value in LIVE_SECONDARY_LABELS.items():
            if value == label:
                return key
        return "nllb600"

    def _on_live_secondary_selected(self, _event: object | None = None) -> None:
        key = self._live_secondary_from_selection()
        self.live_translator.set_opus_secondary(key)
        self._append_log(f"OPUS 2차 번역기를 설정했습니다: {LIVE_SECONDARY_LABELS.get(key, key)}")

    def _on_live_secondary_toggled(self) -> None:
        enabled = self.live_use_secondary_var.get()
        self.live_translator.set_opus_use_secondary(enabled)
        self._append_log(
            "OPUS 2차 번역(영어→한국어)을 " + ("켰습니다." if enabled else "껐습니다. 영어 자막으로 표시됩니다.")
        )

    def _source_lang_id_from_label(self, label: str) -> str:
        for key, value in SOURCE_LANG_LABELS.items():
            if value == label:
                return key
        return SOURCE_LANG_AUTO

    def _on_source_lang_selected(self, _event: object | None = None) -> None:
        if self._is_file_running or self._is_live_running:
            self.source_lang_var.set(SOURCE_LANG_LABELS.get(self.asr.source_language, SOURCE_LANG_LABELS[SOURCE_LANG_AUTO]))
            messagebox.showwarning(APP_NAME, "현재 작업이 끝난 뒤 원본 언어를 바꿔 주세요.")
            return
        code = self._source_lang_id_from_label(self.source_lang_var.get())
        self.asr.set_source_language(code)
        self.live_asr.set_source_language(code)
        self._append_log(f"원본 언어: {SOURCE_LANG_LABELS.get(code, code)}")

    def _prepare_asr_model(self) -> None:
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        model_name = self._whisper_model_from_selection()
        self.asr.set_model(model_name)
        self._apply_resource_mode_from_selection()
        self._set_asr_preparing(True)
        self._emit("log", f"위스퍼 모델을 미리 다운로드하는 중입니다: {self.asr.model_label(model_name)}")

        def worker() -> None:
            try:
                self.asr.prepare_model(model_name)
                self._emit(
                    "asr_prepare_done",
                    {
                        "ok": True,
                        "message": (
                            f"위스퍼 모델 준비 완료: {self.asr.model_label(model_name)} | "
                            f"{self.asr.resource_summary()}"
                        ),
                    },
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "asr_prepare_done",
                    {"ok": False, "message": f"위스퍼 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _translation_profile_id_from_label(self, label: str) -> str:
        label = self._strip_status_prefix(label)
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
        self._refresh_model_status()
        self._append_log(f"번역 방식을 설정했습니다: {self.translator.profile_label()}")

    def _prepare_translation_model(self) -> None:
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        profile = self._translation_profile_from_selection()
        self.translator.set_profile(profile)
        self._apply_resource_mode_from_selection()
        self._set_translation_preparing(True)
        self._emit("log", f"번역 모델을 준비하는 중입니다: {self.translator.profile_label()}")

        def worker() -> None:
            try:
                self.translator.prepare_profile(profile)
                self._emit(
                    "translation_prepare_done",
                    {
                        "ok": True,
                        "message": (
                            f"번역 모델 준비 완료: {self.translator.profile_label(profile)} | "
                            f"{self.translator.resource_summary()}"
                        ),
                    },
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "translation_prepare_done",
                    {"ok": False, "message": f"번역 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _prepare_live_asr_model(self) -> None:
        # 실시간 탭 위스퍼 모델을 시작 전에 미리 내려받는다(파일 탭과 동일한 흐름, live_asr 대상).
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        model_name = self._live_whisper_model_from_selection()
        self.live_asr.set_model(model_name)
        self._apply_resource_mode_from_selection()
        self._set_asr_preparing(True)
        self._emit("log", f"실시간 위스퍼 모델을 미리 다운로드하는 중입니다: {self.live_asr.model_label(model_name)}")

        def worker() -> None:
            try:
                self.live_asr.prepare_model(model_name)
                self._emit(
                    "asr_prepare_done",
                    {
                        "ok": True,
                        "message": (
                            f"실시간 위스퍼 모델 준비 완료: {self.live_asr.model_label(model_name)} | "
                            f"{self.live_asr.resource_summary()}"
                        ),
                    },
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "asr_prepare_done",
                    {"ok": False, "message": f"실시간 위스퍼 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _prepare_live_translation_model(self) -> None:
        # 실시간 탭 번역 모델을 시작 전에 미리 내려받는다(live_translator 대상).
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if self._is_asr_preparing or self._is_translation_preparing:
            return

        profile = self._live_translation_profile_from_selection()
        self.live_translator.set_profile(profile)
        self._apply_resource_mode_from_selection()
        self._set_translation_preparing(True)
        self._emit("log", f"실시간 번역 모델을 준비하는 중입니다: {self.live_translator.profile_label()}")

        def worker() -> None:
            try:
                self.live_translator.prepare_profile(profile)
                self._emit(
                    "translation_prepare_done",
                    {
                        "ok": True,
                        "message": (
                            f"실시간 번역 모델 준비 완료: {self.live_translator.profile_label(profile)} | "
                            f"{self.live_translator.resource_summary()}"
                        ),
                    },
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit(
                    "translation_prepare_done",
                    {"ok": False, "message": f"실시간 번역 모델 준비 실패: {exc}"},
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_live_source_selected(self, _event: object | None = None) -> None:
        # 입력 방식이 'PC 스피커 소리'면 URL 칸을 비활성화(회색)해, 입력해도 무시되는 혼동을 없앤다.
        entry = getattr(self, "live_url_entry", None)
        if entry is None:
            return
        is_url = self.live_source_var.get() == LIVE_SOURCE_URL
        try:
            entry.configure(state="normal" if is_url else "disabled")
        except Exception:
            pass

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
        self._sync_file_list_scrollbar()
        self.batch_status_var.set(f"{len(self.file_queue)}개 파일 대기 중")

    def _sync_file_list_scrollbar(self) -> None:
        visible_rows = int(self.file_listbox.cget("height"))
        needs_scroll = len(self.file_queue) > visible_rows
        if needs_scroll and not self.file_list_scroll.winfo_manager():
            self.file_list_scroll.pack(side="right", fill="y", padx=(2, 4), pady=8)
        elif not needs_scroll:
            self.file_list_scroll.pack_forget()

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
        self._apply_resource_mode_from_selection()

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
        self._failed_sources = []
        self._set_file_busy(True)
        self._begin_timed_progress()
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
            failed_sources: list[str] = []
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
                        failed_sources.append(media_source)
                        self._emit("log", f"[{index}/{total}] 실패: {exc}")
                        continue

                    if self.file_cancel_event.is_set():
                        break

                if self.file_cancel_event.is_set():
                    self._emit("progress", (saved_count / total) * 100)
                    self._emit("eta_status", "중지됨")
                    self._emit("file_status", f"취소됨 ({saved_count}/{total})")
                    self._emit("status", "취소됨")
                else:
                    self._emit("progress", (saved_count / total) * 100 if saved_count < total else 100)
                    self._emit("file_status", f"완료 ({saved_count}/{total})")
                    self._emit("done", f"작업이 끝났습니다. 성공: {saved_count}개, 실패: {failed_count}개")
            except Exception as exc:
                self._emit("error", str(exc))
            finally:
                self._emit("file_retryable", list(failed_sources))
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

        region, region_ok = self._resolve_capture_region()
        if not region_ok:
            messagebox.showwarning(APP_NAME, "녹화할 영역을 먼저 지정해 주세요. ('영역 선택' 버튼)")
            return

        codec = DEFAULT_RECORD_CODEC
        codec_label = self.record_codec_var.get()
        for key, value in RECORD_CODEC_LABELS.items():
            if value == codec_label:
                codec = key
                break

        self.record_cancel_event = threading.Event()
        self.record_cancel_event.clear()
        self.progress_started_at = None
        self._set_progress(0)
        self._set_eta_status("녹화 중에는 표시하지 않음")
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
                    region=region,
                    codec=codec,
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit("eta_status", "대기 중")
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

        self.live_asr.set_model(self._live_whisper_model_from_selection())
        self.live_translator.set_profile(self._live_translation_profile_from_selection())
        self._apply_resource_mode_from_selection()

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
        self.progress_started_at = None
        self._set_progress(0)
        self._set_eta_status("실시간 작업은 표시하지 않음")
        source_text = f"인터넷 URL 스트림: {live_url}" if live_source == LIVE_SOURCE_URL else "PC 스피커 소리"
        srt_text = "사용" if self.live_save_srt_var.get() else "사용 안 함"
        overlay_text = "표시" if self.show_overlay_var.get() else "숨김"
        self._emit("log", f"실시간 자막 시작 버튼을 눌렀습니다. 입력: {source_text}")
        self._emit("log", f"실시간 설정: 음성 인식 {self.live_asr.model_label()} | 번역 {self.live_translator.profile_label()} | SRT 저장 {srt_text} | 자막창 {overlay_text}")
        if self.live_asr.source_language == SOURCE_LANG_AUTO:
            self._emit(
                "log",
                "안내: 소스 언어가 '자동 감지'입니다. 듣는 언어가 정해져 있다면 소스 언어를 "
                "직접 지정하면 짧은 발화의 언어 오인식에 따른 오역을 크게 줄일 수 있습니다.",
            )
        self._emit("busy_live", True)
        self._emit("live_status", "연결 중 ...")
        self._emit("status", "실시간 자막")

        if self.show_overlay_var.get():
            self._show_live_overlay()
        self._sync_live_overlay_button()

        save_srt_path = default_live_srt_path() if self.live_save_srt_var.get() else None

        def worker() -> None:
            try:
                self.live_service.start(
                    self._emit,
                    self.live_cancel_event,
                    internet_url=live_url if live_source == LIVE_SOURCE_URL else None,
                    save_srt_path=save_srt_path,
                )
            except Exception as exc:
                self._emit("error", str(exc))
                self._emit("eta_status", "대기 중")
                self._emit("busy_live", False)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_live_job(self) -> None:
        if not self._is_live_running:
            return

        self._append_log("실시간 자막 중지 버튼을 눌렀습니다.")
        self._emit("live_status", "중지하는 중 ...")
        try:
            self.live_service.stop(emit=self._emit)
        except Exception as exc:
            self._append_log(f"실시간 자막 중지 오류: {exc}")
        self._emit("busy_live", False)
        self._emit("live_status", "대기 중")
        self._emit("status", "대기 중")
        self._emit("eta_status", "대기 중")
        self._append_log("실시간 자막 중지가 완료되었습니다.")

    def _retry_failed_files(self) -> None:
        if self._is_file_running or self._is_live_running or self._is_record_running:
            return
        if not self._failed_sources:
            return
        self.file_queue = list(self._failed_sources)
        self._refresh_file_queue()
        self._start_file_job()

    def _open_output(self) -> None:
        if self.last_output is None or not self.last_output.exists():
            messagebox.showwarning(APP_NAME, "아직 열 수 있는 결과 파일이 없습니다.")
            return
        if os.name == "nt":
            os.startfile(str(self.last_output))
        else:
            self._emit("log", f"이 운영체제에서는 자동 열기를 사용할 수 없습니다: {self.last_output}")

    def _open_output_folder(self) -> None:
        if self.last_output is None or not self.last_output.exists():
            messagebox.showwarning(APP_NAME, "아직 열 수 있는 결과 폴더가 없습니다.")
            return
        folder = self.last_output.parent
        if os.name == "nt":
            os.startfile(str(folder))
        else:
            self._emit("log", f"이 운영체제에서는 자동 열기를 사용할 수 없습니다: {folder}")
def main() -> None:
    if not APP_ROOT.exists():
        raise RuntimeError(f"앱 폴더를 확인할 수 없습니다: {APP_ROOT}")

    app = PapaStopApp()
    app.mainloop()


if __name__ == "__main__":
    main()


