from __future__ import annotations

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

NLLB_FAST_MODEL = "facebook/nllb-200-distilled-1.3B"
NLLB_ACCURATE_MODEL = "facebook/nllb-200-3.3B"
NLLB_TOKENIZER_MODEL = NLLB_ACCURATE_MODEL
NLLB_CT2_MODEL_DIRS = {
    NLLB_FAST_MODEL: PAPASTOP_DATA / "nllb-200-distilled-1.3B-ct2",
    NLLB_ACCURATE_MODEL: PAPASTOP_DATA / "nllb-200-3.3B-ct2",
}

TRANSLATION_PROFILE_ORDER = ("auto", "fast", "balanced", "accurate")
TRANSLATION_PROFILE_LABELS = {
    "auto": "자동 (NLLB 우선, 실패 시 Argos)",
    "fast": "빠름 (Argos 번역)",
    "balanced": "균형 (NLLB 1.3B)",
    "accurate": "정확도 우선 (NLLB 3.3B)",
}
TRANSLATION_PROFILE_ORDER_STEPS = {
    "auto": (("nllb", NLLB_FAST_MODEL), ("argos", None)),
    "fast": (("argos", None),),
    "balanced": (("nllb", NLLB_FAST_MODEL),),
    "accurate": (("nllb", NLLB_ACCURATE_MODEL), ("nllb", NLLB_FAST_MODEL)),
}

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
UI_BORDER_STRONG = "#b9c1d1"  # 카드/입력 테두리 (또렷하게)
UI_CARD_BORDER = "#c4ccda"   # 카드 구분용 테두리
UI_TEXT = "#0f172a"          # 본문 텍스트
UI_TEXT_MUTED = "#64748b"    # 보조 텍스트
UI_ACCENT = "#4f46e5"        # 인디고 강조색
UI_ACCENT_HOVER = "#4338ca"
UI_ACCENT_ACTIVE = "#3730a3"
OVERLAY_BG = "#0f172a"       # 자막 오버레이 배경 (어두운 톤 유지)
OVERLAY_FG = "#f8fafc"

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


def get_primary_cuda_vram_bytes() -> int | None:
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


def default_live_srt_path(output_dir: Path | None = None) -> Path:
    folder = output_dir if output_dir is not None else PAPASTOP_DATA / "subtitles"
    return folder / f"live-{time.strftime('%Y%m%d-%H%M%S')}.ko.srt"


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

    def model_label(self, model_name: str | None = None) -> str:
        target = model_name or self.model_name
        return WHISPER_MODEL_LABELS.get(target, target)

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
        audio_path: Path,
        on_line=None,
        should_stop: Callable[[], bool] | None = None,
        on_segment: Callable[[SubtitleSegment], None] | None = None,
    ) -> tuple[list[SubtitleSegment], str]:
        self._load_model(on_line=on_line)
        assert self._model is not None

        if should_stop is not None and should_stop():
            raise OperationCancelled("취소되었습니다.")

        settings = build_resource_settings(self.resource_mode)
        forced_language = None if self.source_language == SOURCE_LANG_AUTO else self.source_language
        transcribe_kwargs = {
            "language": forced_language,
            "beam_size": settings.asr_beam_size,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if self._batched_model is not None:
            transcribe_kwargs["batch_size"] = settings.asr_batch_size
            transcribe_kwargs["without_timestamps"] = False
            segments, info = self._batched_model.transcribe(str(audio_path), **transcribe_kwargs)
        else:
            segments, info = self._model.transcribe(str(audio_path), **transcribe_kwargs)

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
            on_line(f"자원 사용: {self.resource_summary()}")
            on_line(f"감지된 언어: {detected_language}, 자막 줄 수={len(rows)}")

        return rows, detected_language


class TranslationEngine:
    def __init__(self, preferred: str = "auto", resource_mode: str = DEFAULT_RESOURCE_MODE):
        self.preferred = preferred
        self.resource_mode = normalize_resource_mode(resource_mode)
        self._nllb_pipelines: dict[str, object] = {}
        self._nllb_ct2_translators: dict[str, object] = {}
        self._nllb_tokenizer = None
        self._pipeline_lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._resource_summary = build_resource_settings(self.resource_mode).summary

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
                beam_size=4 if self.preferred != "accurate" else 5,
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
            on_line(f"번역 자원 사용: {self.translator.resource_summary()}")
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
        self._srt_path: Path | None = None
        self._srt_rows: list[SubtitleSegment] = []
        self._srt_emitted = False
        self._dropped_chunks = 0
        self._lag_warned_at = 0.0

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
        self._temp_dir = Path(tempfile.mkdtemp(prefix="papastop_live_"))
        self._source_url = internet_url.strip() if internet_url else None
        self._srt_path = save_srt_path
        self._srt_rows = []
        self._srt_emitted = False
        self._dropped_chunks = 0
        self._lag_warned_at = 0.0
        if self._srt_path is not None:
            self._srt_path.parent.mkdir(parents=True, exist_ok=True)

        # 첫 청크에서 모델을 로딩하면 10초 가까이 걸려 그 사이 들어온 앞 구간이 버려진다.
        # 캡처를 시작하기 전에 음성 인식·번역 모델을 미리 올려 두어 첫 구간부터 따라잡게 한다.
        emit("live_status", "음성 인식·번역 모델을 준비하는 중입니다 ...")
        try:
            self.asr.prepare_model()
            self.translator.prepare_profile()
        except Exception as exc:
            emit("log", f"실시간 모델 사전 준비 중 경고: {exc}")
        if stop_event.is_set():
            return
        emit("live_status", "실시간 자막을 시작합니다 ...")

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

            work_started_at = time.monotonic()
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
                    pairs = [(s, (t.strip() or s.text.strip())) for s, t in zip(segments, translated) if (t or s.text).strip()]
                    lines = [text for _, text in pairs]
                    if lines:
                        emit("live_caption", "\n".join(lines[-2:]))
                    if self._srt_path is not None and pairs:
                        self._append_live_srt(chunk_index, pairs, emit)
                self._report_processing_lag(time.monotonic() - work_started_at, emit)
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

    def _report_processing_lag(self, elapsed: float, emit: Callable[[str, object], None]) -> None:
        # 한 청크 분석 시간이 입력 간격보다 길면 따라가지 못해 일부 구간을 건너뛴다.
        if elapsed <= self.chunk_seconds * 1.15:
            return
        now = time.monotonic()
        if now - self._lag_warned_at < 10.0:
            return
        self._lag_warned_at = now
        message = (
            f"⏱ 실시간 분석이 느립니다: 분석 {elapsed:.1f}초 > 입력 간격 {self.chunk_seconds:.1f}초. "
            "따라가지 못해 일부 구간을 건너뜁니다"
        )
        if self._dropped_chunks:
            message += f" (지금까지 건너뜀 {self._dropped_chunks}구간)"
        message += ". 더 빠른 모델(예: tiny/base)이나 균형 모드를 권장합니다."
        emit("live_status", message)
        emit("log", message)

    def _append_live_srt(
        self,
        chunk_index: int,
        pairs: list[tuple[SubtitleSegment, str]],
        emit: Callable[[str, object], None],
    ) -> None:
        if self._srt_path is None:
            return
        offset = chunk_index * self.chunk_seconds
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
        self.configure(bg=UI_ACCENT)
        self._drag_start_x = 0
        self._drag_start_y = 0

        # 윈도우 상단 바가 없는 상태에서도 마우스로 위치를 이동할 수 있게 드래그 이벤트 등록
        self.bind("<Button-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._on_drag)

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
        )
        self.label.bind("<Button-1>", self._begin_drag)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.pack(fill="both", expand=True, padx=2, pady=2)

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


class PapaStopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self._configure_main_window()
        self._apply_theme()
        self.configure(fg_color=UI_BG)

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
        self.progress_started_at: float | None = None

        self._build_ui()
        self._refresh_status()
        self._refresh_system_specs()
        self.after(80, self._process_events)

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
        assign("CTkOptionMenu", "fg_color", indigo)
        assign("CTkOptionMenu", "button_color", indigo_hover)
        assign("CTkOptionMenu", "button_hover_color", indigo_active)
        assign("CTkCheckBox", "fg_color", indigo)
        assign("CTkCheckBox", "hover_color", indigo_hover)
        assign("CTkRadioButton", "fg_color", indigo)
        assign("CTkRadioButton", "hover_color", indigo_hover)
        assign("CTkSegmentedButton", "selected_color", ["#e0e7ff", "#e0e7ff"])
        assign("CTkSegmentedButton", "selected_hover_color", ["#c7d2fe", "#c7d2fe"])
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
            parent, fg_color=UI_SURFACE, corner_radius=12, border_width=1, border_color=UI_CARD_BORDER
        )
        ctk.CTkLabel(
            card, text=title, font=self.font_section, text_color=UI_TEXT, anchor="w"
        ).pack(fill="x", padx=16, pady=(12, 2))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        return card, body

    def _live_label(self, parent, variable: tk.StringVar, **kwargs):
        """StringVar 값이 바뀌면 자동으로 글자를 갱신하는 CTkLabel을 만듭니다."""
        label = ctk.CTkLabel(parent, text=variable.get(), **kwargs)
        variable.trace_add("write", lambda *_a, v=variable, w=label: w.configure(text=v.get()))
        return label

    def _build_ui(self) -> None:
        root = self
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=3)
        root.grid_rowconfigure(1, weight=2)

        fam = self._ui_font_family
        self.font_base = ctk.CTkFont(family=fam, size=13)
        self.font_section = ctk.CTkFont(family=fam, size=14, weight="bold")
        self.font_caption = ctk.CTkFont(family=fam, size=15)

        self._ghost_btn = dict(
            fg_color="transparent",
            hover_color=UI_SURFACE_HOVER,
            text_color=UI_ACCENT,
            border_width=1,
            border_color=UI_BORDER_STRONG,
            corner_radius=8,
        )
        self._danger_btn = dict(fg_color="#dc2626", hover_color="#b91c1c", text_color="#ffffff")
        self._entry_kw = dict(
            height=34,
            corner_radius=8,
            border_color=UI_BORDER_STRONG,
            fg_color="#ffffff",
            text_color=UI_TEXT,
            font=self.font_base,
        )
        self._option_kw = dict(height=34, corner_radius=8, font=self.font_base, dropdown_font=self.font_base)

        # 모든 탭에서 똑같이 쓰는 주요 동작 버튼 규격 (시작 / 중지 / 결과 열기)
        self._primary_btn_kw = dict(width=150, height=40, corner_radius=8, font=self.font_base)
        self._stop_btn_kw = dict(width=96, height=40, corner_radius=8, **self._danger_btn)
        self._open_btn_kw = dict(width=110, height=40, **self._ghost_btn)

        self.tabview = ctk.CTkTabview(
            root,
            fg_color=UI_BG,
            segmented_button_fg_color=UI_SURFACE,
            segmented_button_unselected_color=UI_SURFACE,
            segmented_button_unselected_hover_color=UI_SURFACE_HOVER,
            text_color=UI_TEXT,
            corner_radius=12,
        )
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=14, pady=(12, 6))

        file_tab_frame = self.tabview.add("파일 자막")
        live_tab_frame = self.tabview.add("실시간 자막")
        record_tab_frame = self.tabview.add("화면 녹화")
        system_tab_frame = self.tabview.add("시스템 사양")

        self.file_tab = ctk.CTkScrollableFrame(file_tab_frame, fg_color="transparent")
        self.file_tab.pack(fill="both", expand=True)
        self.live_tab = live_tab_frame
        self.record_tab = record_tab_frame
        self.system_tab = system_tab_frame

        self.system_cpu_var = tk.StringVar(value="확인 중 ...")
        self.system_ram_var = tk.StringVar(value="확인 중 ...")
        self.system_vga_var = tk.StringVar(value="확인 중 ...")
        self.resource_mode_var = tk.StringVar(value=RESOURCE_MODE_LABELS[DEFAULT_RESOURCE_MODE])
        self.resource_plan_var = tk.StringVar(value=build_resource_settings(DEFAULT_RESOURCE_MODE).summary)

        system_card, system_body = self._make_card(self.system_tab, "현재 PC 사양")
        system_card.pack(fill="x", pady=(2, 10))

        def add_system_row(label: str, variable: tk.StringVar) -> None:
            row = ctk.CTkFrame(system_body, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(
                row, text=label, width=80, anchor="w", text_color=UI_TEXT_MUTED, font=self.font_base
            ).pack(side="left", anchor="n")
            self._live_label(
                row,
                variable,
                font=self.font_base,
                justify="left",
                anchor="w",
                wraplength=620,
                text_color=UI_TEXT,
            ).pack(side="left", fill="x", expand=True)

        add_system_row("CPU", self.system_cpu_var)
        add_system_row("RAM", self.system_ram_var)
        add_system_row("VGA", self.system_vga_var)

        resource_row = ctk.CTkFrame(system_body, fg_color="transparent")
        resource_row.pack(fill="x", pady=4)
        ctk.CTkLabel(
            resource_row, text="자원 사용", width=80, anchor="w", text_color=UI_TEXT_MUTED, font=self.font_base
        ).pack(side="left", anchor="n")
        self.resource_mode_combo = ctk.CTkOptionMenu(
            resource_row,
            variable=self.resource_mode_var,
            values=[RESOURCE_MODE_LABELS[key] for key in RESOURCE_MODE_ORDER],
            command=self._on_resource_mode_selected,
            width=280,
            **self._option_kw,
        )
        self.resource_mode_combo.pack(side="left", padx=(0, 10))
        self._live_label(
            resource_row,
            self.resource_plan_var,
            font=self.font_base,
            justify="left",
            anchor="w",
            wraplength=520,
            text_color=UI_TEXT_MUTED,
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            self.system_tab,
            text="사양 새로고침",
            command=self._refresh_system_specs,
            width=120,
            height=34,
            **self._ghost_btn,
        ).pack(anchor="w", pady=6)

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

        def file_row() -> ctk.CTkFrame:
            row = ctk.CTkFrame(self.file_tab, fg_color="transparent")
            row.pack(fill="x", pady=6)
            return row

        def row_label(parent, text, **kw):
            return ctk.CTkLabel(parent, text=text, font=self.font_base, text_color=UI_TEXT, **kw)

        file_row0 = file_row()
        row_label(file_row0, "원본 영상/음성").pack(side="left")
        ctk.CTkEntry(file_row0, textvariable=self.file_path_var, **self._entry_kw).pack(
            side="left", fill="x", expand=True, padx=10
        )
        self.select_media_btn = ctk.CTkButton(
            file_row0, text="선택", command=self._select_media, width=70, height=34, **self._ghost_btn
        )
        self.select_media_btn.pack(side="left")
        self.add_file_btn = ctk.CTkButton(
            file_row0, text="목록에 추가", command=self._add_input_to_queue, width=100, height=34, **self._ghost_btn
        )
        self.add_file_btn.pack(side="left", padx=8)

        file_row1 = file_row()
        row_label(file_row1, "저장할 SRT / 폴더").pack(side="left")
        ctk.CTkEntry(file_row1, textvariable=self.output_path_var, **self._entry_kw).pack(
            side="left", fill="x", expand=True, padx=10
        )
        ctk.CTkButton(
            file_row1, text="같은 이름으로", command=self._set_default_output, width=110, height=34, **self._ghost_btn
        ).pack(side="left")
        ctk.CTkButton(
            file_row1, text="폴더 선택", command=self._select_output_dir, width=90, height=34, **self._ghost_btn
        ).pack(side="left", padx=8)

        file_row2 = file_row()
        self.start_file_btn = ctk.CTkButton(
            file_row2, text="자막 만들기 시작", command=self._start_file_job, **self._primary_btn_kw
        )
        self.cancel_file_btn = ctk.CTkButton(
            file_row2, text="중지", command=self._cancel_file_job, state="disabled", **self._stop_btn_kw
        )
        self.start_file_btn.pack(side="left")
        self.cancel_file_btn.pack(side="left", padx=8)
        ctk.CTkButton(
            file_row2, text="결과 열기", command=self._open_output, **self._open_btn_kw
        ).pack(side="right")

        file_row3 = file_row()
        row_label(file_row3, "음성 인식 모델").pack(side="left")
        self.whisper_model_combo = ctk.CTkOptionMenu(
            file_row3,
            variable=self.whisper_model_var,
            values=[WHISPER_MODEL_LABELS[key] for key in WHISPER_MODEL_ORDER],
            command=self._on_whisper_model_selected,
            width=220,
            **self._option_kw,
        )
        self.whisper_model_combo.pack(side="left", padx=10)
        self.prepare_asr_btn = ctk.CTkButton(
            file_row3,
            text="위스퍼 모델 다운로드",
            command=self._prepare_asr_model,
            width=150,
            height=34,
            **self._ghost_btn,
        )
        self.prepare_asr_btn.pack(side="left", padx=6)
        row_label(file_row3, "원본 언어").pack(side="left", padx=(18, 0))
        self.file_source_lang_combo = ctk.CTkOptionMenu(
            file_row3,
            variable=self.source_lang_var,
            values=[SOURCE_LANG_LABELS[key] for key in SOURCE_LANG_ORDER],
            command=self._on_source_lang_selected,
            width=130,
            **self._option_kw,
        )
        self.file_source_lang_combo.pack(side="left", padx=10)

        file_row4 = file_row()
        row_label(file_row4, "번역 모델").pack(side="left")
        self.translation_profile_combo = ctk.CTkOptionMenu(
            file_row4,
            variable=self.translation_profile_var,
            values=[TRANSLATION_PROFILE_LABELS[key] for key in TRANSLATION_PROFILE_ORDER],
            command=self._on_translation_profile_selected,
            width=320,
            **self._option_kw,
        )
        self.translation_profile_combo.pack(side="left", padx=10)
        self.prepare_translation_btn = ctk.CTkButton(
            file_row4,
            text="번역 모델 다운로드",
            command=self._prepare_translation_model,
            width=150,
            height=34,
            **self._ghost_btn,
        )
        self.prepare_translation_btn.pack(side="left", padx=6)

        queue_card, queue_body = self._make_card(self.file_tab, "작업할 파일 목록")
        queue_card.pack(fill="both", expand=True, pady=(4, 6))

        list_area = ctk.CTkFrame(queue_body, fg_color="transparent")
        list_area.pack(fill="both", expand=True)

        listbox_wrap = ctk.CTkFrame(list_area, fg_color="#ffffff", corner_radius=8, border_width=1, border_color=UI_BORDER_STRONG)
        listbox_wrap.pack(side="left", fill="both", expand=True)
        self.file_listbox = tk.Listbox(
            listbox_wrap,
            height=6,
            selectmode="extended",
            exportselection=False,
            bg="#ffffff",
            fg=UI_TEXT,
            font=(self._ui_font_family, 10),
            selectbackground=UI_ACCENT,
            selectforeground="#ffffff",
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        list_scroll = ctk.CTkScrollbar(listbox_wrap, command=self.file_listbox.yview)
        list_scroll.pack(side="right", fill="y", padx=(2, 4), pady=8)
        self.file_listbox.configure(yscrollcommand=list_scroll.set)

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
            self.translation_profile_combo,
            self.prepare_translation_btn,
        ]

        ctk.CTkLabel(
            self.file_tab,
            text="영상이나 음성에서 한국어 SRT 자막을 만들어 저장합니다.",
            font=self.font_base,
            text_color=UI_TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", pady=(8, 2))

        status_row = ctk.CTkFrame(self.file_tab, fg_color="transparent")
        status_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(status_row, text="파일 상태", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        self._live_label(status_row, self.file_status_var, font=self.font_base, text_color=UI_TEXT).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(status_row, text="전체 진행", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left", padx=(20, 0))
        self._live_label(status_row, self.batch_status_var, font=self.font_base, text_color=UI_TEXT).pack(side="left", padx=(8, 0))

        preview_card, preview_body = self._make_card(self.file_tab, "실시간 추출 미리보기")
        preview_card.pack(fill="both", expand=True, pady=(4, 2))
        self.file_preview_text = ctk.CTkTextbox(
            preview_body,
            height=110,
            wrap="word",
            fg_color="#ffffff",
            text_color=UI_TEXT,
            border_color=UI_BORDER_STRONG,
            border_width=1,
            corner_radius=8,
            font=self.font_base,
        )
        self.file_preview_text.pack(fill="both", expand=True)
        self.file_preview_text.insert("end", "자막 만들기를 시작하면 추출된 문장이 여기에 표시됩니다.")
        self.file_preview_text.configure(state="disabled")

        self._refresh_file_queue()

        # Live tab
        self.live_status_var = tk.StringVar(value="대기 중")
        self.live_preview_var = tk.StringVar(value="실시간 한국어 자막이 여기에 표시됩니다.")
        self.show_overlay_var = tk.BooleanVar(value=True)
        self.live_save_srt_var = tk.BooleanVar(value=True)

        source_card, source_body = self._make_card(self.live_tab, "실시간 소리 입력")
        source_card.pack(fill="x", pady=(2, 10))

        live_row0 = ctk.CTkFrame(source_body, fg_color="transparent")
        live_row0.pack(fill="x", pady=4)
        ctk.CTkLabel(live_row0, text="입력", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        ctk.CTkRadioButton(
            live_row0,
            text="PC 스피커 소리",
            variable=self.live_source_var,
            value=LIVE_SOURCE_SPEAKER,
            font=self.font_base,
        ).pack(side="left", padx=(12, 14))
        ctk.CTkRadioButton(
            live_row0,
            text="인터넷 URL 스트림",
            variable=self.live_source_var,
            value=LIVE_SOURCE_URL,
            font=self.font_base,
        ).pack(side="left")
        ctk.CTkLabel(live_row0, text="원본 언어", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left", padx=(18, 0))
        self.live_source_lang_combo = ctk.CTkOptionMenu(
            live_row0,
            variable=self.source_lang_var,
            values=[SOURCE_LANG_LABELS[key] for key in SOURCE_LANG_ORDER],
            command=self._on_source_lang_selected,
            width=130,
            **self._option_kw,
        )
        self.live_source_lang_combo.pack(side="left", padx=10)

        live_row_url = ctk.CTkFrame(source_body, fg_color="transparent")
        live_row_url.pack(fill="x", pady=4)
        ctk.CTkLabel(live_row_url, text="URL", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        ctk.CTkEntry(live_row_url, textvariable=self.live_url_var, **self._entry_kw).pack(
            side="left", fill="x", expand=True, padx=10
        )

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

        ctk.CTkLabel(
            self.live_tab, text="실시간 자막 미리보기", font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w"
        ).pack(fill="x", pady=(12, 2))
        preview_card = ctk.CTkFrame(self.live_tab, fg_color=UI_SURFACE, corner_radius=12)
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
        self._live_label(
            self.live_tab, self.live_status_var, font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w"
        ).pack(fill="x", pady=6)

        # Screen record tab
        record_card, record_body = self._make_card(self.record_tab, "녹화 설정")
        record_card.pack(fill="x", pady=(2, 10))

        record_row0 = ctk.CTkFrame(record_body, fg_color="transparent")
        record_row0.pack(fill="x", pady=4)
        ctk.CTkLabel(record_row0, text="저장할 MKV", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        self.record_output_entry = ctk.CTkEntry(record_row0, textvariable=self.record_output_var, **self._entry_kw)
        self.record_output_entry.pack(side="left", fill="x", expand=True, padx=10)
        self.browse_record_btn = ctk.CTkButton(
            record_row0, text="찾아보기", command=self._select_record_output, width=90, height=34, **self._ghost_btn
        )
        self.browse_record_btn.pack(side="left")

        record_row1 = ctk.CTkFrame(record_body, fg_color="transparent")
        record_row1.pack(fill="x", pady=4)
        ctk.CTkLabel(record_row1, text="초당 화면 수", font=self.font_base, text_color=UI_TEXT_MUTED).pack(side="left")
        self.record_fps_combo = ctk.CTkOptionMenu(
            record_row1,
            variable=self.record_fps_var,
            values=["24", "30", "60"],
            width=90,
            **self._option_kw,
        )
        self.record_fps_combo.pack(side="left", padx=(10, 20))
        self.record_audio_check = ctk.CTkCheckBox(
            record_row1,
            text="PC 스피커 소리도 녹음",
            variable=self.record_audio_var,
            onvalue=True,
            offvalue=False,
            font=self.font_base,
        )
        self.record_audio_check.pack(side="left")

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

        self._live_label(
            self.record_tab, self.record_status_var, font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w"
        ).pack(fill="x", pady=6)
        self._record_controls = [
            self.record_output_entry,
            self.browse_record_btn,
            self.record_fps_combo,
            self.record_audio_check,
            self.start_record_btn,
            self.stop_record_btn,
        ]

        # Bottom common
        common = ctk.CTkFrame(
            root, fg_color=UI_SURFACE, corner_radius=12, border_width=1, border_color=UI_CARD_BORDER
        )
        common.grid(row=1, column=0, sticky="nsew", padx=14, pady=(6, 14))

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="대기 중")
        self.eta_var = tk.StringVar(value="예상 시간: 대기 중")

        ctk.CTkLabel(common, text="진행 상황과 작업 기록", font=self.font_section, text_color=UI_TEXT, anchor="w").pack(
            fill="x", padx=16, pady=(12, 2)
        )
        self._live_label(common, self.status_var, font=self.font_base, text_color=UI_TEXT, anchor="w").pack(
            fill="x", padx=16
        )
        self._live_label(common, self.eta_var, font=self.font_base, text_color=UI_TEXT_MUTED, anchor="w").pack(
            fill="x", padx=16, pady=(0, 6)
        )
        self.progress = ctk.CTkProgressBar(common, height=12, corner_radius=6, fg_color=UI_BORDER)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=16, pady=(0, 8))

        self.log_text = ctk.CTkTextbox(
            common,
            height=150,
            wrap="word",
            fg_color="#ffffff",
            text_color=UI_TEXT,
            border_color=UI_BORDER_STRONG,
            border_width=1,
            corner_radius=8,
            font=self.font_base,
        )
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.log_text.configure(state="disabled")

        self._append_log("앱을 시작했습니다.")

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

    def _refresh_resource_plan_label(self) -> None:
        self.resource_plan_var.set(build_resource_settings(self._resource_mode_from_selection()).summary)

    def _append_log(self, msg: object) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
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
            elif kind == "record_status":
                self.record_status_var.set(str(payload))
            elif kind == "progress":
                progress = float(payload)
                self._set_progress(progress)
                self._update_eta(progress)
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
                state = "normal"
            widget.configure(state=state)
        if self._is_asr_preparing:
            self.prepare_asr_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
        if self._is_translation_preparing:
            self.prepare_translation_btn.configure(state="disabled")
            self.start_file_btn.configure(state="disabled")
        self._refresh_resource_mode_control_state()

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

    def _set_live_busy(self, busy: bool) -> None:
        self._is_live_running = busy
        self.start_live_btn.configure(state="disabled" if busy else "normal")
        self.stop_live_btn.configure(state="normal" if busy else "disabled")
        if not busy and self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        self._refresh_resource_mode_control_state()

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
        tools.append("자원 사용: " + build_resource_settings(self._resource_mode_from_selection()).summary)
        self._append_log(" | ".join(tools))

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
        self.translator.set_resource_mode(mode)
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
        self._append_log(f"자원 사용 방식을 바꿨습니다: {self.asr.resource_summary()}")

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

        self.asr.set_model(self._whisper_model_from_selection())
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
        self._emit("busy_live", True)
        self._emit("live_status", "연결 중 ...")
        self._emit("status", "실시간 자막")

        if self.show_overlay_var.get():
            self.overlay = LiveSubtitleOverlay(self)
            self.overlay.lift()

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

        self._emit("live_status", "중지하는 중 ...")
        try:
            self.live_service.stop()
        except Exception as exc:
            self._append_log(f"실시간 자막 중지 오류: {exc}")
        self._emit("busy_live", False)
        self._emit("live_status", "대기 중")
        self._emit("status", "대기 중")
        self._emit("eta_status", "대기 중")

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



