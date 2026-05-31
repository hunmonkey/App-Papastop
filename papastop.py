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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_NAME = "PapaStop"
def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = app_dir()
REQUIREMENTS_PATH = APP_DIR / "requirements.txt"
# 모델·CUDA 라이브러리 등 무거운 데이터를 EXE 옆에 비압축으로 두는 외부 폴더입니다.
# (EXE는 코드만 담아 가볍게 유지하고, 자주 바뀌는 모델은 이 폴더에서 교체합니다.)
APP_CACHE_DIR = APP_DIR / ".papastop_data"
HF_CACHE_DIR = APP_CACHE_DIR / "huggingface"

ENGINE_PACKAGES = {
    "faster-whisper": "faster_whisper",
    "argostranslate": "argostranslate",
    "imageio-ffmpeg": "imageio_ffmpeg",
}

# NVIDIA 그래픽카드가 있을 때만 추가로 설치하는 GPU 가속 라이브러리입니다.
# faster-whisper(CTranslate2)가 GPU에서 추론하려면 cuda-runtime/cuBLAS/cuDNN이 필요합니다.
CUDA_RUNTIME_PACKAGES = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
]

MEDIA_FILE_TYPES = [
    (
        "Video and audio files",
        "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v *.mp3 *.wav *.m4a *.aac *.flac",
    ),
    ("All files", "*.*"),
]

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
}

SUBTITLE_FILE_TYPES = [("SRT subtitle", "*.srt"), ("All files", "*.*")]
SUBTITLE_SAVE_TYPES = [
    ("SRT subtitle", "*.srt"),
    ("WebVTT subtitle", "*.vtt"),
    ("Text file", "*.txt"),
    ("All files", "*.*"),
]

SUBTITLE_FORMATS = {
    "srt": "SRT",
    "vtt": "WebVTT",
    "txt": "TXT",
}

LANGUAGES = {
    "auto": "자동 감지",
    "ko": "한국어",
    "en": "영어",
    "ja": "일본어",
    "zh": "중국어",
    "es": "스페인어",
    "fr": "프랑스어",
    "de": "독일어",
    "it": "이탈리아어",
    "pt": "포르투갈어",
    "ru": "러시아어",
}

TRANSLATION_LANGUAGES = {key: value for key, value in LANGUAGES.items() if key != "auto"}

# 번역 엔진: argos(가벼움·빠름), nllb(고품질·GPU)
TRANSLATION_ENGINES = {
    "argos": "Argos (가벼움·빠름)",
    "nllb": "NLLB (고품질)",
}

# NLLB(ctranslate2) 모델 위치와 FLORES-200 언어 코드 매핑
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
NLLB_MODEL_DIR = APP_CACHE_DIR / "nllb-200-distilled-1.3B-ct2"
NLLB_LANGUAGE_CODES = {
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


@dataclass
class SubtitleBlock:
    index: str
    timing: str
    lines: list[str]


class OperationCancelled(RuntimeError):
    pass


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def windows_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def is_running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def install_target_label() -> str:
    if is_frozen_app():
        return "EXE 패키지 내부"
    if is_running_in_venv():
        return "프로젝트 전용 가상환경"
    return "현재 Windows 사용자 Python 폴더"


def prepare_model_cache_environment() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR / "home"))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def prepare_cuda_library_path() -> None:
    """faster-whisper(CTranslate2)가 GPU에서 cuda-runtime/cuBLAS/cuDNN을 찾도록
    설치된 nvidia 라이브러리 폴더를 DLL 검색 경로에 추가합니다.
    Windows에서는 PATH 등록이 있어야 GPU 추론이 정상 동작합니다.
    개발 환경(venv)과 EXE 패키지(frozen) 양쪽을 모두 지원합니다."""
    roots: list[Path] = []

    try:
        import nvidia

        roots.extend(Path(path) for path in getattr(nvidia, "__path__", []))
    except ImportError:
        pass

    # EXE 배포에서는 nvidia 라이브러리를 exe 옆 .papastop_data\nvidia 폴더에 비압축으로 둡니다.
    external_nvidia = APP_CACHE_DIR / "nvidia"
    if external_nvidia.is_dir():
        roots.append(external_nvidia)

    bin_dirs: list[str] = []
    for root in roots:
        for component in ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = str(root / component / "bin")
            if (root / component / "bin").is_dir() and bin_dir not in bin_dirs:
                bin_dirs.append(bin_dir)

    if not bin_dirs:
        return

    current_path = os.environ.get("PATH", "")
    missing = [bin_dir for bin_dir in bin_dirs if bin_dir not in current_path]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current_path

    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            pass


def run_command(
    command: list[str],
    on_line: Callable[[str], None] | None = None,
    cwd: Path | None = None,
) -> int:
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

    return process.wait()


def find_ffmpeg() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    if has_module("imageio_ffmpeg"):
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            return get_ffmpeg_exe()
        except Exception:
            return None

    return None


def detect_nvidia_gpus() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=8,
        )
    except Exception:
        return []

    results: list[str] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue

        name = parts[0]
        memory_mb = parts[1]
        driver = parts[2] if len(parts) >= 3 else ""

        try:
            memory_label = f"{float(memory_mb) / 1024:.1f} GB"
        except ValueError:
            memory_label = f"{memory_mb} MB"

        label = f"{name} / {memory_label}"
        if driver:
            label += f" / 드라이버 {driver}"
        results.append(label)

    return results


def check_engine_status() -> dict[str, str]:
    status: dict[str, str] = {}

    status["설치 위치"] = install_target_label()

    for package_name, module_name in ENGINE_PACKAGES.items():
        status[package_name] = "설치됨" if has_module(module_name) else "미설치"

    ffmpeg_path = find_ffmpeg()
    status["ffmpeg"] = ffmpeg_path if ffmpeg_path else "미설치"

    nvidia_gpus = detect_nvidia_gpus()
    status["nvidia"] = " / ".join(nvidia_gpus) if nvidia_gpus else "미감지"

    return status


def install_python_engines(on_line: Callable[[str], None]) -> None:
    if is_frozen_app():
        missing_packages = [
            package_name
            for package_name, module_name in ENGINE_PACKAGES.items()
            if not has_module(module_name)
        ]
        if missing_packages:
            raise RuntimeError(
                "EXE 패키지 안에서 엔진을 찾지 못했습니다. "
                "배포 파일을 다시 압축 해제하거나 새 패키지를 받아 주세요."
            )

        on_line("EXE 패키지에는 필요한 엔진이 이미 포함되어 있습니다.")
        on_line("별도 설치 없이 바로 자막 제작과 번역 기능을 사용할 수 있습니다.")
        return

    command = [sys.executable, "-m", "pip", "install", "--upgrade"]

    if not is_running_in_venv():
        command.append("--user")

    if REQUIREMENTS_PATH.exists():
        command.extend(["-r", str(REQUIREMENTS_PATH)])
    else:
        command.extend(ENGINE_PACKAGES.keys())

    if detect_nvidia_gpus():
        command.extend(CUDA_RUNTIME_PACKAGES)
        on_line("NVIDIA 그래픽카드가 감지되어 GPU 가속 라이브러리도 함께 설치합니다.")

    on_line("Python 자막/번역 엔진 설치를 시작합니다.")
    on_line(f"설치 위치: {install_target_label()}")
    on_line(" ".join(command))
    exit_code = run_command(command, on_line=on_line)

    if exit_code != 0:
        raise RuntimeError("엔진 설치 명령이 실패했습니다. 로그의 마지막 줄을 확인해 주세요.")

    on_line("엔진 설치가 끝났습니다. 앱을 다시 실행하면 새 엔진을 더 안정적으로 인식합니다.")


def run_powershell_json(command: str) -> object | None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return None

    script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$OutputEncoding=[System.Text.Encoding]::UTF8; "
        f"{command} | ConvertTo-Json -Depth 4"
    )

    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=12,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def bytes_to_gb(value: int | float | str | None) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "확인 필요"

    return f"{number / (1024 ** 3):.1f} GB"


def format_file_size(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def media_files_in_folder(folder_path: Path) -> list[Path]:
    if not folder_path.exists() or not folder_path.is_dir():
        return []

    return sorted(
        (
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def language_display(code: str) -> str:
    return LANGUAGES.get(code, code)


def language_code_from_display(value: str) -> str:
    if value in LANGUAGES:
        return value

    for code, label in LANGUAGES.items():
        if value == label:
            return code

    for code, label in LANGUAGES.items():
        if value.startswith(f"{label} "):
            return code

    return value


def language_values(allow_auto: bool = True) -> list[str]:
    source = LANGUAGES if allow_auto else TRANSLATION_LANGUAGES
    return [language_display(code) for code in source]


def translation_engine_values() -> list[str]:
    return list(TRANSLATION_ENGINES.values())


def translation_engine_code(value: str) -> str:
    if value in TRANSLATION_ENGINES:
        return value
    for code, label in TRANSLATION_ENGINES.items():
        if value == label:
            return code
    return "argos"


DEVICE_OPTIONS = {
    "auto": "자동 선택 (auto)",
    "cpu": "CPU 사용 (cpu)",
    "cuda": "GPU·NVIDIA (cuda)",
}


def device_values() -> list[str]:
    return list(DEVICE_OPTIONS.values())


def device_code(value: str) -> str:
    if value in DEVICE_OPTIONS:
        return value
    for code, label in DEVICE_OPTIONS.items():
        if value == label:
            return code
    return "auto"


def detect_hardware() -> dict[str, str]:
    hardware = {
        "운영체제": platform.platform(),
        "Python": sys.version.split()[0],
    }

    if os.name == "nt":
        cpu = run_powershell_json(
            "Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"
        )
        memory = run_powershell_json(
            "Get-CimInstance Win32_ComputerSystem | Select-Object -First 1 TotalPhysicalMemory"
        )
        gpus = run_powershell_json(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion"
        )

        if isinstance(cpu, dict):
            hardware["CPU"] = str(cpu.get("Name") or "확인 필요")
            hardware["CPU 코어"] = str(cpu.get("NumberOfCores") or "확인 필요")
            hardware["CPU 논리 코어"] = str(cpu.get("NumberOfLogicalProcessors") or "확인 필요")
            hardware["CPU 최대 속도"] = f"{cpu.get('MaxClockSpeed', '확인 필요')} MHz"

        if isinstance(memory, dict):
            hardware["RAM"] = bytes_to_gb(memory.get("TotalPhysicalMemory"))

        nvidia_gpus = detect_nvidia_gpus()
        if nvidia_gpus:
            for index, gpu in enumerate(nvidia_gpus, start=1):
                hardware[f"GPU {index}"] = gpu
        elif isinstance(gpus, dict):
            gpus = [gpus]

        if not nvidia_gpus and isinstance(gpus, list):
            for index, gpu in enumerate(gpus, start=1):
                if not isinstance(gpu, dict):
                    continue
                name = gpu.get("Name") or "확인 필요"
                ram = bytes_to_gb(gpu.get("AdapterRAM"))
                hardware[f"GPU {index}"] = f"{name} / {ram}"

    hardware.setdefault("CPU", platform.processor() or "확인 필요")

    if "RAM" not in hardware and has_module("psutil"):
        try:
            import psutil

            hardware["RAM"] = bytes_to_gb(psutil.virtual_memory().total)
        except Exception:
            pass

    return hardware


def largest_gpu_memory_gb(nvidia_gpus: list[str]) -> float:
    best = 0.0
    for label in nvidia_gpus:
        match = re.search(r"([\d.]+)\s*GB", label)
        if match:
            best = max(best, float(match.group(1)))
    return best


def recommend_transcribe_settings(nvidia_gpus: list[str] | None = None) -> tuple[dict[str, object], str]:
    if nvidia_gpus is None:
        nvidia_gpus = detect_nvidia_gpus()

    if nvidia_gpus:
        memory_gb = largest_gpu_memory_gb(nvidia_gpus)
        if memory_gb >= 8:
            model = "large-v3-turbo"
        elif memory_gb >= 5:
            model = "small"
        else:
            model = "base"
        memory_note = f"약 {memory_gb:.0f}GB" if memory_gb else "메모리 확인 필요"
        return (
            {
                "model": model,
                "device": "cuda",
                "compute": "float16",
                "beam": 5,
            },
            f"NVIDIA 그래픽카드({memory_note})에 맞춰 {model} 모델 GPU 설정을 적용했습니다.",
        )

    return (
        {
            "model": "base",
            "device": "cpu",
            "compute": "int8",
            "beam": 3,
        },
        "NVIDIA 그래픽카드가 감지되지 않아 CPU 중심 설정을 적용했습니다.",
    )


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(max(seconds, 0) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def wrap_subtitle_text(text: str, max_chars: int) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    if " " in text:
        lines: list[str] = []
        current = ""
        for word in text.split(" "):
            if current and len(current) + 1 + len(word) > max_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        return "\n".join(lines)

    return "\n".join(text[i : i + max_chars] for i in range(0, len(text), max_chars))


def render_srt(rows: Iterable[tuple[float, float, str]], max_line_chars: int = 42) -> str:
    blocks: list[str] = []

    for index, (start, end, text) in enumerate(rows, start=1):
        clean_text = wrap_subtitle_text(text, max_line_chars)
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(start)} --> {format_timestamp(end)}",
                    clean_text,
                ]
            )
        )

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_srt_blocks(text: str) -> list[SubtitleBlock]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks: list[SubtitleBlock] = []
    for raw_block in re.split(r"\n\s*\n", normalized):
        lines = raw_block.split("\n")
        if len(lines) < 3:
            continue

        index = lines[0].strip()
        timing = lines[1].strip()
        content_lines = [line.rstrip() for line in lines[2:]]
        blocks.append(SubtitleBlock(index=index, timing=timing, lines=content_lines))

    return blocks


def render_srt_blocks(blocks: Iterable[SubtitleBlock]) -> str:
    rendered: list[str] = []
    for block in blocks:
        rendered.append("\n".join([block.index, block.timing, *block.lines]))
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def render_vtt_blocks(blocks: Iterable[SubtitleBlock]) -> str:
    rendered = ["WEBVTT", ""]
    for block in blocks:
        rendered.append(block.timing.replace(",", "."))
        rendered.extend(block.lines)
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def render_text_blocks(blocks: Iterable[SubtitleBlock]) -> str:
    paragraphs: list[str] = []
    for block in blocks:
        paragraph = " ".join(line.strip() for line in block.lines if line.strip())
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def subtitle_extension(format_code: str) -> str:
    normalized = format_code.lower().strip(".")
    if normalized not in SUBTITLE_FORMATS:
        raise ValueError(f"지원하지 않는 자막 형식입니다: {format_code}")
    return f".{normalized}"


def ensure_subtitle_extension(path: Path, format_code: str) -> Path:
    extension = subtitle_extension(format_code)
    if path.suffix.lower() == extension:
        return path
    return path.with_suffix(extension)


def convert_srt_text(text: str, output_format: str) -> str:
    blocks = parse_srt_blocks(text)
    if not blocks:
        raise RuntimeError("SRT 자막 내용을 읽지 못했습니다.")

    normalized = output_format.lower().strip(".")
    if normalized == "srt":
        return render_srt_blocks(blocks)
    if normalized == "vtt":
        return render_vtt_blocks(blocks)
    if normalized == "txt":
        return render_text_blocks(blocks)

    raise ValueError(f"지원하지 않는 자막 형식입니다: {output_format}")


def convert_subtitle_file(
    source_path: Path,
    target_path: Path,
    output_format: str,
    on_line: Callable[[str], None],
    on_progress: Callable[[float], None],
) -> Path:
    on_line(f"자막 변환을 시작합니다: {SUBTITLE_FORMATS.get(output_format, output_format)}")
    on_progress(20)
    target_path = ensure_subtitle_extension(target_path, output_format)
    converted = convert_srt_text(read_text_file(source_path), output_format)
    on_progress(70)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(converted, encoding="utf-8")
    on_progress(100)
    on_line(f"변환 자막 저장 완료: {target_path}")
    return target_path


def ensure_not_cancelled(should_stop: Callable[[], bool] | None) -> None:
    if should_stop and should_stop():
        raise OperationCancelled("작업이 중지되었습니다.")


def validate_media_for_transcription(media_path: Path) -> None:
    if media_path.suffix.lower() not in MEDIA_EXTENSIONS:
        raise RuntimeError(f"지원하지 않는 파일 형식입니다: {media_path.suffix}")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg를 찾지 못했습니다. 먼저 엔진 설치를 눌러 주세요.")

    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(media_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("동영상 정보를 확인하는 데 시간이 너무 오래 걸렸습니다.") from exc

    probe_text = f"{completed.stdout}\n{completed.stderr}"
    if "Invalid data found" in probe_text or "No such file" in probe_text:
        raise RuntimeError("파일을 동영상/음성 파일로 읽지 못했습니다.")

    if "Audio:" not in probe_text:
        raise RuntimeError("이 파일에서 음성 트랙을 찾지 못했습니다. 자막을 만들려면 소리가 들어 있어야 합니다.")


def language_label(code: str) -> str:
    return f"{LANGUAGES.get(code, code)} ({code})"


def translate_srt_file(
    source_path: Path,
    target_path: Path,
    from_code: str,
    to_code: str,
    on_line: Callable[[str], None],
    on_progress: Callable[[float], None],
    should_stop: Callable[[], bool] | None = None,
    engine: str = "argos",
) -> None:
    if from_code == to_code:
        raise ValueError("원본 언어와 번역 언어가 같습니다.")

    if engine == "nllb":
        translate_srt_file_nllb(
            source_path, target_path, from_code, to_code, on_line, on_progress, should_stop
        )
        return

    if not has_module("argostranslate"):
        raise RuntimeError("Argos Translate가 설치되어 있지 않습니다. 먼저 엔진 설치를 눌러 주세요.")

    from argostranslate import package, translate

    on_line(f"번역 언어팩을 확인합니다: {language_label(from_code)} -> {language_label(to_code)}")

    installed_languages = translate.get_installed_languages()
    installed_from = next((language for language in installed_languages if language.code == from_code), None)
    installed_to = next((language for language in installed_languages if language.code == to_code), None)

    if installed_from is None or installed_to is None or installed_from.get_translation(installed_to) is None:
        on_line("언어팩이 없어 다운로드합니다. 첫 실행 때만 시간이 걸릴 수 있습니다.")
        package.update_package_index()
        available_packages = package.get_available_packages()
        matching_package = next(
            (
                available_package
                for available_package in available_packages
                if available_package.from_code == from_code and available_package.to_code == to_code
            ),
            None,
        )
        if matching_package is None:
            raise RuntimeError("Argos에서 해당 번역 언어팩을 찾지 못했습니다.")

        download_path = matching_package.download()
        package.install_from_path(download_path)
        on_line("언어팩 설치가 끝났습니다.")

    text = read_text_file(source_path)
    blocks = parse_srt_blocks(text)
    if not blocks:
        raise RuntimeError("SRT 자막 내용을 읽지 못했습니다.")

    translated_blocks: list[SubtitleBlock] = []
    total_lines = sum(len(block.lines) for block in blocks) or 1
    translated_lines = 0

    for block in blocks:
        ensure_not_cancelled(should_stop)
        new_lines: list[str] = []
        for line in block.lines:
            if line.strip():
                new_lines.append(translate.translate(line, from_code, to_code))
            else:
                new_lines.append(line)

            translated_lines += 1
            on_progress(min(95, translated_lines / total_lines * 95))

        translated_blocks.append(SubtitleBlock(block.index, block.timing, new_lines))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(render_srt_blocks(translated_blocks), encoding="utf-8")
    on_progress(100)
    on_line(f"번역 자막 저장 완료: {target_path}")


def nllb_model_dir() -> Path:
    # 개발 환경과 EXE 배포 모두 exe 옆 .papastop_data 폴더의 모델을 사용합니다.
    return NLLB_MODEL_DIR


def nllb_model_is_downloaded() -> bool:
    return nllb_model_dir().is_dir()


def ensure_nllb_model(on_line: Callable[[str], None]) -> None:
    if nllb_model_dir().is_dir():
        return

    on_line("NLLB 모델을 준비합니다. 처음 한 번만 큰 다운로드(약 5GB)와 변환이 필요하며 수 분 걸립니다.")
    from ctranslate2.converters import TransformersConverter

    APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    on_line("모델을 내려받아 변환하는 중입니다. 진행 표시줄이 움직이는 동안 기다려 주세요.")
    converter = TransformersConverter(NLLB_MODEL_NAME)
    converter.convert(str(NLLB_MODEL_DIR), quantization="int8_float16")
    on_line("NLLB 모델 준비가 끝났습니다.")


_nllb_translator = None
_nllb_tokenizers: dict[str, object] = {}


def load_nllb(
    from_code: str,
    to_code: str,
    on_line: Callable[[str], None],
) -> tuple[object, object, str]:
    nllb_from = NLLB_LANGUAGE_CODES.get(from_code)
    nllb_to = NLLB_LANGUAGE_CODES.get(to_code)
    if not nllb_from or not nllb_to:
        raise RuntimeError("NLLB가 지원하지 않는 언어입니다.")

    if not has_module("ctranslate2") or not has_module("transformers"):
        raise RuntimeError("NLLB 번역 엔진(ctranslate2/transformers)이 설치되어 있지 않습니다.")

    ensure_nllb_model(on_line)
    prepare_cuda_library_path()

    import ctranslate2
    from transformers import AutoTokenizer

    global _nllb_translator
    if _nllb_translator is None:
        on_line("NLLB 번역 모델을 불러옵니다.")
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        _nllb_translator = ctranslate2.Translator(str(nllb_model_dir()), device=device)

    if nllb_from not in _nllb_tokenizers:
        _nllb_tokenizers[nllb_from] = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=nllb_from)

    return _nllb_tokenizers[nllb_from], _nllb_translator, nllb_to


def translate_srt_file_nllb(
    source_path: Path,
    target_path: Path,
    from_code: str,
    to_code: str,
    on_line: Callable[[str], None],
    on_progress: Callable[[float], None],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    tokenizer, translator, nllb_to = load_nllb(from_code, to_code, on_line)

    on_line(f"NLLB로 번역합니다: {language_label(from_code)} -> {language_label(to_code)}")
    text = read_text_file(source_path)
    blocks = parse_srt_blocks(text)
    if not blocks:
        raise RuntimeError("SRT 자막 내용을 읽지 못했습니다.")

    translated_blocks: list[SubtitleBlock] = []
    total_lines = sum(len(block.lines) for block in blocks) or 1
    translated_lines = 0

    for block in blocks:
        ensure_not_cancelled(should_stop)
        new_lines: list[str] = []
        for line in block.lines:
            if line.strip():
                source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
                results = translator.translate_batch([source_tokens], target_prefix=[[nllb_to]])
                target_tokens = results[0].hypotheses[0]
                if target_tokens and target_tokens[0] == nllb_to:
                    target_tokens = target_tokens[1:]
                translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
                new_lines.append(translated)
            else:
                new_lines.append(line)

            translated_lines += 1
            on_progress(min(95, translated_lines / total_lines * 95))

        translated_blocks.append(SubtitleBlock(block.index, block.timing, new_lines))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(render_srt_blocks(translated_blocks), encoding="utf-8")
    on_progress(100)
    on_line(f"번역 자막 저장 완료: {target_path}")


def whisper_model_repo(model_size: str) -> str | None:
    try:
        from faster_whisper.utils import _MODELS
    except Exception:
        return None
    return _MODELS.get(model_size)


def local_whisper_model_path(model_size: str) -> Path | None:
    """캐시에 받아둔 Whisper 모델 폴더(snapshots/<버전>) 경로를 돌려줍니다(없으면 None).
    faster-whisper에 모델 이름 대신 이 폴더 경로를 주면 인터넷 검증을 건너뛰고
    (오프라인에서도) 바로 로드하므로, 이미 받은 모델을 다시 받는 문제를 막습니다."""
    repo = whisper_model_repo(model_size)
    if not repo:
        return None
    prepare_model_cache_environment()
    cache = Path(os.environ.get("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub")))
    snapshots = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.glob("*")):
        if (snapshot / "model.bin").is_file():
            return snapshot
    return None


def whisper_model_is_downloaded(model_size: str) -> bool:
    return local_whisper_model_path(model_size) is not None


def download_whisper_model(model_size: str, on_line: Callable[[str], None]) -> None:
    if not has_module("faster_whisper"):
        raise RuntimeError("faster-whisper가 설치되어 있지 않습니다. 먼저 엔진 설치를 눌러 주세요.")
    prepare_model_cache_environment()
    prepare_cuda_library_path()
    from faster_whisper import WhisperModel

    on_line(f"{model_size} 모델을 다운로드합니다. 크기에 따라 시간이 걸릴 수 있습니다.")
    WhisperModel(model_size, device="cpu", compute_type="int8")
    on_line(f"{model_size} 모델 다운로드가 끝났습니다.")


def load_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
    on_line: Callable[[str], None],
) -> object:
    if not has_module("faster_whisper"):
        raise RuntimeError("faster-whisper가 설치되어 있지 않습니다. 먼저 엔진 설치를 눌러 주세요.")

    prepare_model_cache_environment()
    prepare_cuda_library_path()

    from faster_whisper import WhisperModel

    compute_arg = "default" if compute_type == "auto" else compute_type

    local_path = local_whisper_model_path(model_size)
    if local_path is not None:
        # 받아둔 모델은 폴더 경로로 직접 로드합니다.
        # (모델 이름으로 주면 faster-whisper가 인터넷에서 캐시 무결성을 다시 확인하다가
        #  이미 있는 모델을 또 받을 수 있습니다. 폴더 경로면 그 과정을 건너뜁니다.)
        on_line(f"준비된 모델을 사용합니다: {model_size}")
        return WhisperModel(str(local_path), device=device, compute_type=compute_arg)

    on_line(f"모델을 준비합니다: {model_size}")
    on_line("처음 쓰는 모델은 자동으로 다운로드됩니다. 영상 파일이 큰 경우 시간이 걸립니다.")
    return WhisperModel(model_size, device=device, compute_type=compute_arg)


def transcribe_media(
    media_path: Path,
    output_dir: Path,
    model: object,
    language_code: str,
    beam_size: int,
    on_line: Callable[[str], None],
    on_progress: Callable[[float], None],
    should_stop: Callable[[], bool] | None = None,
    max_line_chars: int = 42,
) -> tuple[Path, str]:
    ensure_not_cancelled(should_stop)
    validate_media_for_transcription(media_path)
    ensure_not_cancelled(should_stop)

    output_dir.mkdir(parents=True, exist_ok=True)
    language_arg = None if language_code == "auto" else language_code

    on_progress(5)
    on_line("영상의 소리를 분석합니다.")

    detected_language = language_code or "audio"

    def collect_rows(vad_filter: bool) -> list[tuple[float, float, str]]:
        nonlocal detected_language
        segments, info = model.transcribe(
            str(media_path),
            language=language_arg,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        collected: list[tuple[float, float, str]] = []
        duration = getattr(info, "duration", None)
        detected_language = getattr(info, "language", None) or language_code or "audio"
        for segment in segments:
            ensure_not_cancelled(should_stop)
            collected.append((float(segment.start), float(segment.end), str(segment.text)))
            if duration:
                on_progress(5 + min(90, float(segment.end) / float(duration) * 90))
            elif len(collected) % 5 == 0:
                on_progress(min(95, 5 + len(collected)))
        return collected

    rows = collect_rows(True)
    if not rows:
        ensure_not_cancelled(should_stop)
        on_line("음성 구간을 찾지 못해, 무음 제거 없이 다시 분석합니다.")
        on_progress(5)
        rows = collect_rows(False)

    if not rows:
        raise RuntimeError("소리에서 자막으로 만들 문장을 찾지 못했습니다.")

    output_path = output_dir / f"{media_path.stem}.srt"
    output_path.write_text(render_srt(rows, max_line_chars), encoding="utf-8")
    on_progress(100)
    on_line(f"자막 저장 완료: {output_path}")
    return output_path, detected_language


def extract_embedded_subtitle(
    media_path: Path,
    output_dir: Path,
    on_line: Callable[[str], None],
    on_progress: Callable[[float], None],
) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg를 찾지 못했습니다. 먼저 엔진 설치를 눌러 주세요.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{media_path.stem}.embedded.srt"

    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(media_path),
        "-map",
        "0:s:0?",
        str(output_path),
    ]

    on_line("영상 안에 이미 들어 있는 자막을 찾습니다.")
    on_progress(20)
    exit_code = run_command(command, on_line=on_line)

    if exit_code != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError("영상 안에서 꺼낼 수 있는 SRT 자막을 찾지 못했습니다.")

    on_progress(100)
    on_line(f"기존 자막 추출 완료: {output_path}")
    return output_path


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event: tk.Event | None = None) -> None:
        if self.window or not self.text:
            return

        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self.window,
            text=self.text,
            background="#fff7cc",
            foreground="#111827",
            relief=tk.SOLID,
            borderwidth=1,
            padding=(8, 5),
            wraplength=360,
        )
        label.pack()

    def hide(self, _event: tk.Event | None = None) -> None:
        if self.window:
            self.window.destroy()
            self.window = None


class PapaStopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.minsize(1120, 720)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.cancel_event = threading.Event()
        self.media_table_paths: dict[str, Path] = {}
        self.selected_media_iids: set[str] = set()

        self.media_path_var = tk.StringVar()
        self.media_folder_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.subtitle_path_var = tk.StringVar()
        self.model_var = tk.StringVar(value="base")
        self.model_status_var = tk.StringVar(value="")
        self.source_lang_var = tk.StringVar(value=language_display("auto"))
        self.device_var = tk.StringVar(value=DEVICE_OPTIONS["auto"])
        self.compute_var = tk.StringVar(value="auto")
        self.beam_var = tk.IntVar(value=5)
        self.max_line_chars_var = tk.IntVar(value=42)
        self.auto_translate_var = tk.BooleanVar(value=False)
        self.auto_translate_to_var = tk.StringVar(value=language_display("ko"))
        self.expert_mode_var = tk.BooleanVar(value=False)
        self.translate_from_var = tk.StringVar(value=language_display("en"))
        self.translate_to_var = tk.StringVar(value=language_display("ko"))
        self.translate_engine_var = tk.StringVar(value=TRANSLATION_ENGINES["argos"])
        self.nllb_status_var = tk.StringVar(value="")
        self.translate_overwrite_var = tk.BooleanVar(value=False)
        self.translate_output_path_var = tk.StringVar()
        self.convert_format_var = tk.StringVar(value="vtt")
        self.convert_output_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="대기 중")
        self.progress_var = tk.DoubleVar(value=0)
        self.task_var = tk.StringVar(value="make")  # 'make', 'extract', 'translate', 'convert', 'engine'

        self._configure_style()
        self._build_ui()
        self.after(100, self._process_events)
        self.after(250, self.refresh_hardware)
        self.after(350, self._check_and_prompt_for_initial_setup)
        self.after(450, self.refresh_engines)
        self.after(650, self.refresh_model_status)
        self.after(750, self.refresh_nllb_status)
        self.task_var.trace_add("write", self._on_task_changed)
        self.translate_overwrite_var.trace_add("write", self._on_overwrite_changed)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.configure(bg="#f6f7f9")
        style.configure(".", font=("Malgun Gothic", 10))
        style.configure("TFrame", background="#f6f7f9")
        style.configure("Panel.TLabelframe", background="#f6f7f9", padding=12)
        style.configure("Panel.TLabelframe.Label", font=("Malgun Gothic", 11, "bold"))
        style.configure("Title.TLabel", font=("Malgun Gothic", 20, "bold"), background="#f6f7f9")
        style.configure("Subtle.TLabel", foreground="#4b5563", background="#f6f7f9")
        style.configure("Status.TLabel", foreground="#1f2937", background="#f6f7f9")
        style.configure("Accent.TButton", font=("Malgun Gothic", 10, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(header, text="PapaStop", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor=tk.E,
        ).pack(side=tk.RIGHT)

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, width=460)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)

        self._build_file_panel(left)

        self._build_task_selection_panel(left)

        self.settings_container = ttk.Frame(left)
        self.settings_container.pack(fill=tk.X, expand=True, pady=(0, 10))

        # 각 작업 패널을 만들고 인스턴스 변수에 저장합니다.
        # 각 _build_*_panel 메서드는 이제 pack()을 호출하지 않고 프레임을 반환합니다.
        self.transcribe_frame = self._build_transcribe_panel(self.settings_container)
        self.extract_frame = self._build_extract_panel(self.settings_container)
        self.translate_frame = self._build_translate_panel(self.settings_container)
        self.convert_frame = self._build_convert_panel(self.settings_container)
        self.engine_frame = self._build_engine_panel(self.settings_container)

        self._build_action_panel(left)

        self._build_progress_panel(right)

        # 오른쪽 패널 (PC 장치, 작업 기록)은 그대로 유지합니다.
        right_tabs = ttk.Notebook(right)
        right_tabs.pack(fill=tk.BOTH, expand=True)

        hardware_tab = ttk.Frame(right_tabs, padding=(0, 8, 0, 0))
        log_tab = ttk.Frame(right_tabs, padding=(0, 8, 0, 0))
        right_tabs.add(hardware_tab, text="PC 장치")
        right_tabs.add(log_tab, text="작업 기록")

        self._build_hardware_panel(hardware_tab)
        self._build_log_panel(log_tab)

        # 앱 시작 시 기본 작업 패널을 보여줍니다.
        self._on_task_changed()

    def _label_with_help(self, parent: ttk.Frame, text: str, help_text: str) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=text).pack(side=tk.LEFT)
        help_label = ttk.Label(frame, text="?", width=2, anchor=tk.CENTER, foreground="#2563eb")
        help_label.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(help_label, help_text)
        return frame

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. 파일 선택", style="Panel.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 12))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="자막을 만들 영상/음성을 선택하세요").grid(row=0, column=0, sticky=tk.W)
        source_row = ttk.Frame(frame)
        source_row.grid(row=1, column=0, sticky=tk.EW, pady=(4, 8))
        source_row.columnconfigure(0, weight=1)
        source_row.columnconfigure(1, weight=1)
        ttk.Button(source_row, text="파일 1개 찾기", command=self.choose_media_file).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 3)
        )
        ttk.Button(source_row, text="폴더 전체 찾기", command=self.choose_media_folder).grid(
            row=0, column=1, sticky=tk.EW, padx=(3, 0)
        )

        table_frame = ttk.Frame(frame)
        table_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))
        table_frame.columnconfigure(0, weight=1)

        self.media_tree = ttk.Treeview(
            table_frame,
            columns=("selected", "name", "ext", "size", "status"),
            show="headings",
            height=2,
        )
        self.media_tree.heading("selected", text="선택")
        self.media_tree.heading("name", text="파일명")
        self.media_tree.heading("ext", text="확장자")
        self.media_tree.heading("size", text="파일용량")
        self.media_tree.heading("status", text="상태")
        self.media_tree.column("selected", width=70, stretch=False, anchor=tk.CENTER)
        self.media_tree.column("name", width=180, stretch=True)
        self.media_tree.column("ext", width=70, stretch=False, anchor=tk.CENTER)
        self.media_tree.column("size", width=90, stretch=False, anchor=tk.E)
        self.media_tree.column("status", width=80, stretch=False, anchor=tk.CENTER)
        self.media_tree.grid(row=0, column=0, sticky=tk.EW)
        self.media_tree.bind("<Button-1>", self._on_media_tree_click)

        media_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.media_tree.yview)
        media_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.media_tree.configure(yscrollcommand=media_scrollbar.set)

        selection_row = ttk.Frame(frame)
        selection_row.grid(row=3, column=0, sticky=tk.EW, pady=(0, 6))
        selection_row.columnconfigure(0, weight=1)
        selection_row.columnconfigure(1, weight=1)
        ttk.Button(selection_row, text="전체 선택", command=self.select_all_media_files).grid(row=0, column=0, sticky=tk.EW, padx=(0, 3))
        ttk.Button(selection_row, text="전체 해제", command=self.clear_media_file_selection).grid(row=0, column=1, sticky=tk.EW, padx=(3, 0))

        ttk.Label(frame, text="저장 폴더").grid(row=4, column=0, sticky=tk.W)
        output_row = ttk.Frame(frame)
        output_row.grid(row=5, column=0, sticky=tk.EW, pady=(4, 0))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_dir_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(output_row, text="폴더", command=self.choose_output_dir).grid(row=0, column=1, padx=(6, 0))

    def _build_task_selection_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. 작업 선택", style="Panel.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        tasks = {
            "자막 만들기": "make",
            "영상 속 자막 추출": "extract",
            "기존 자막 번역": "translate",
            "자막 형식 변환": "convert",
            "엔진 관리": "engine",
        }

        for index, (text, value) in enumerate(tasks.items()):
            ttk.Radiobutton(
                frame,
                text=text,
                variable=self.task_var,
                value=value
            ).grid(row=index // 3, column=index % 3, sticky=tk.W, padx=10, pady=2)

    def _build_engine_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="3. 세부 설정 (엔진 관리)", style="Panel.TLabelframe")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.engine_tree = ttk.Treeview(frame, columns=("value",), show="tree headings", height=6)
        self.engine_tree.heading("#0", text="항목")
        self.engine_tree.heading("value", text="상태")
        self.engine_tree.column("#0", width=150, stretch=False)
        self.engine_tree.column("value", width=240, stretch=True)
        self.engine_tree.grid(row=0, column=0, columnspan=2, sticky=tk.EW)

        ttk.Button(frame, text="엔진 확인", command=self.refresh_engines).grid(
            row=1, column=0, sticky=tk.EW, pady=(8, 0), padx=(0, 3)
        )
        ttk.Button(
            frame,
            text="엔진 설치",
            command=self.install_engines,
            style="Accent.TButton",
        ).grid(row=1, column=1, sticky=tk.EW, padx=(3, 0), pady=(8, 0))
        return frame

    def _build_extract_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="3. 세부 설정 (자막 추출)", style="Panel.TLabelframe")
        # 이 작업은 별도 설정이 필요 없습니다.
        # '파일' 패널에서 선택한 영상/폴더에 대해 작업이 실행됩니다.
        ttk.Label(
            frame,
            text="선택한 영상 파일에 내장된 SRT 형식 자막을 추출합니다.\n"
                 "별도 설정 없이 아래 '작업 시작' 버튼을 누르세요.",
            wraplength=380,
            justify=tk.LEFT,
        ).pack(pady=20, padx=10)
        return frame

    def _build_transcribe_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="3. 세부 설정 (자막 만들기)", style="Panel.TLabelframe")

        for column in range(2):
            frame.columnconfigure(column, weight=1)

        mode_switch = ttk.Checkbutton(
            frame,
            text="전문가 모드 (고급 설정 표시)",
            variable=self.expert_mode_var,
            command=self._on_mode_changed,
        )
        mode_switch.grid(row=0, column=0, columnspan=2, sticky=tk.E, pady=(0, 10))

        self._label_with_help(
            frame,
            "자막 품질",
            "소리를 글자로 바꾸는 모델입니다. base는 첫 테스트에 좋고, large 계열은 더 정확하지만 시간이 더 걸립니다.",
        ).grid(row=1, column=0, sticky=tk.W)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=2, column=0, sticky=tk.EW, pady=(4, 8), padx=(0, 6))
        model_frame.columnconfigure(0, weight=1)
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"),
            state="readonly",
        )
        model_combo.grid(row=0, column=0, sticky=tk.EW)
        model_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_model_status())
        ttk.Button(model_frame, text="받기", width=6, command=self.download_selected_model).grid(
            row=0, column=1, padx=(4, 0)
        )
        ttk.Label(model_frame, textvariable=self.model_status_var, style="Subtle.TLabel").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0)
        )

        ttk.Label(frame, text="원본 언어").grid(row=1, column=1, sticky=tk.W)
        ttk.Combobox(
            frame,
            textvariable=self.source_lang_var,
            values=language_values(allow_auto=True),
            state="readonly",
        ).grid(row=2, column=1, sticky=tk.EW, pady=(4, 8))

        self.device_label = self._label_with_help(
            frame,
            "사용 장치",
            "자막을 만들 때 CPU 또는 그래픽카드 GPU 중 무엇을 쓸지 정합니다. NVIDIA 그래픽카드가 있으면 cuda가 빠를 수 있습니다.",
        )
        self.device_label.grid(row=3, column=0, sticky=tk.W)
        self.device_combo = ttk.Combobox(
            frame,
            textvariable=self.device_var,
            values=device_values(),
            state="readonly",
        )
        self.device_combo.grid(row=4, column=0, sticky=tk.EW, pady=(4, 8), padx=(0, 6))

        self.compute_label = self._label_with_help(
            frame,
            "계산 방식",
            "계산을 얼마나 가볍게 할지 정합니다. int8은 가볍고 CPU에 유리하며, float16은 GPU에서 빠른 편입니다.",
        )
        self.compute_label.grid(row=3, column=1, sticky=tk.W)
        self.compute_combo = ttk.Combobox(
            frame,
            textvariable=self.compute_var,
            values=("auto", "int8", "float16", "float32"),
            state="readonly",
        )
        self.compute_combo.grid(row=4, column=1, sticky=tk.EW, pady=(4, 8))

        self.beam_label = self._label_with_help(
            frame,
            "정확도 단계",
            "문장을 고를 때 몇 가지 후보를 비교할지 정합니다. 숫자가 높으면 더 꼼꼼하지만 느려질 수 있습니다.",
        )
        self.beam_label.grid(row=5, column=0, sticky=tk.W)
        self.beam_spinbox = ttk.Spinbox(frame, from_=1, to=10, textvariable=self.beam_var)
        self.beam_spinbox.grid(
            row=6,
            column=0,
            sticky=tk.EW,
            pady=(4, 10),
            padx=(0, 6),
        )

        self.max_line_chars_label = self._label_with_help(
            frame,
            "한 줄 최대 글자",
            "자막 한 줄이 이 글자 수를 넘으면 자동으로 줄을 나눕니다. 0이면 줄바꿈하지 않습니다.",
        )
        self.max_line_chars_label.grid(row=5, column=1, sticky=tk.W)
        self.max_line_chars_spinbox = ttk.Spinbox(frame, from_=0, to=100, textvariable=self.max_line_chars_var)
        self.max_line_chars_spinbox.grid(row=6, column=1, sticky=tk.EW, pady=(4, 10))

        self.recommend_button = ttk.Button(
            frame,
            text="권장 설정 적용",
            command=self.apply_recommended_settings,
        )
        self.recommend_button.grid(row=7, column=0, columnspan=2, sticky=tk.EW)

        translate_row = ttk.Frame(frame)
        translate_row.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        translate_row.columnconfigure(1, weight=1)
        auto_translate_check = ttk.Checkbutton(
            translate_row,
            text="자막 만든 뒤 자동 번역 →",
            variable=self.auto_translate_var,
        )
        auto_translate_check.grid(row=0, column=0, sticky=tk.W)
        ToolTip(
            auto_translate_check,
            "체크하면 여러 영상의 자막을 만든 뒤 곧바로 선택한 언어로 번역까지 한 번에 진행합니다. "
            "번역 원본 언어는 위의 '원본 언어' 설정을 따르며, '자동 감지'면 소리에서 감지한 언어를 사용합니다.",
        )
        ttk.Combobox(
            translate_row,
            textvariable=self.auto_translate_to_var,
            values=language_values(allow_auto=False),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky=tk.E, padx=(8, 0))

        self._on_mode_changed()

        return frame

    def _build_translate_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="3. 세부 설정 (번역 저장)", style="Panel.TLabelframe")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="번역할 SRT 파일").grid(row=0, column=0, sticky=tk.W)
        subtitle_row = ttk.Frame(frame)
        subtitle_row.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        subtitle_row.columnconfigure(0, weight=1)
        ttk.Entry(subtitle_row, textvariable=self.subtitle_path_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(subtitle_row, text="찾기", command=self.choose_subtitle_file).grid(row=0, column=1, padx=(6, 0))

        language_row = ttk.Frame(frame)
        language_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        language_row.columnconfigure(0, weight=1)
        language_row.columnconfigure(1, weight=1)

        ttk.Label(language_row, text="원본").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            language_row,
            textvariable=self.translate_from_var,
            values=language_values(allow_auto=False),
            state="readonly",
        ).grid(row=1, column=0, sticky=tk.EW, padx=(0, 6), pady=(4, 0))

        ttk.Label(language_row, text="번역").grid(row=0, column=1, sticky=tk.W)
        ttk.Combobox(
            language_row,
            textvariable=self.translate_to_var,
            values=language_values(allow_auto=False),
            state="readonly",
        ).grid(row=1, column=1, sticky=tk.EW, pady=(4, 0))

        ttk.Label(language_row, text="번역 엔진").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        engine_combo = ttk.Combobox(
            language_row,
            textvariable=self.translate_engine_var,
            values=translation_engine_values(),
            state="readonly",
        )
        engine_combo.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0))
        ToolTip(
            engine_combo,
            "Argos는 가볍고 빠릅니다. NLLB는 더 정확하지만 첫 사용 시 모델 다운로드가 필요하고 GPU에서 빠릅니다. "
            "이 설정은 번역 저장·폴더 일괄 번역과 자막 제작의 자동 번역에 모두 적용됩니다.",
        )

        nllb_row = ttk.Frame(language_row)
        nllb_row.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0))
        nllb_row.columnconfigure(0, weight=1)
        ttk.Label(nllb_row, textvariable=self.nllb_status_var, style="Subtle.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Button(nllb_row, text="NLLB 모델 준비", command=self.prepare_nllb_model).grid(
            row=0, column=1, padx=(4, 0)
        )

        ttk.Label(frame, text="번역 저장 이름").grid(row=3, column=0, sticky=tk.W)
        translate_output_row = ttk.Frame(frame)
        translate_output_row.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        translate_output_row.columnconfigure(0, weight=1)
        self.translate_output_entry = ttk.Entry(translate_output_row, textvariable=self.translate_output_path_var)
        self.translate_output_entry.grid(row=0, column=0, sticky=tk.EW)
        self.translate_output_button = ttk.Button(translate_output_row, text="이름", command=self.choose_translate_output_file)
        self.translate_output_button.grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

        overwrite_check = ttk.Checkbutton(
            frame,
            text="번역 결과를 원본 SRT 파일에 덮어쓰기",
            variable=self.translate_overwrite_var,
        )
        overwrite_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ToolTip(
            overwrite_check,
            "이 옵션을 켜면 '번역 저장 이름' 설정은 무시되고, 번역할 원본 SRT 파일을 덮어씁니다.",
        )

        ttk.Button(
            frame,
            text="폴더 SRT 모두 번역",
            command=self.translate_subtitle_folder,
        ).grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))

        return frame

    def _build_convert_panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text="3. 세부 설정 (자막 변환)", style="Panel.TLabelframe")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="변환할 SRT 파일").grid(row=0, column=0, sticky=tk.W)
        subtitle_row = ttk.Frame(frame)
        subtitle_row.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        subtitle_row.columnconfigure(0, weight=1)
        ttk.Entry(subtitle_row, textvariable=self.subtitle_path_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(subtitle_row, text="찾기", command=self.choose_subtitle_file).grid(row=0, column=1, padx=(6, 0))

        convert_row = ttk.Frame(frame)
        convert_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        convert_row.columnconfigure(1, weight=1)
        ttk.Label(convert_row, text="변환 형식").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Combobox(
            convert_row,
            textvariable=self.convert_format_var,
            values=list(SUBTITLE_FORMATS.keys()),
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky=tk.EW)

        ttk.Label(frame, text="변환 저장 이름").grid(row=3, column=0, sticky=tk.W)
        convert_output_row = ttk.Frame(frame)
        convert_output_row.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        convert_output_row.columnconfigure(0, weight=1)
        ttk.Entry(convert_output_row, textvariable=self.convert_output_path_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(convert_output_row, text="이름", command=self.choose_convert_output_file).grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

        return frame

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="4. 실행", style="Panel.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        self.action_button = ttk.Button(
            frame,
            text="작업 시작",
            command=self.start_work,
            style="Accent.TButton",
        )
        self.action_button.grid(row=0, column=0, sticky=tk.EW)

    def _build_progress_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="진행 상황", style="Panel.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        progress_row = ttk.Frame(frame)
        progress_row.grid(row=0, column=0, sticky=tk.EW)
        progress_row.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_row, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(progress_row, text="작업 중지", command=self.stop_current_work).grid(row=0, column=1, padx=(8, 0))

        self.step_tree = ttk.Treeview(frame, columns=("status",), show="tree headings", height=6)
        self.step_tree.heading("#0", text="단계")
        self.step_tree.heading("status", text="상태")
        self.step_tree.column("#0", width=240, stretch=True)
        self.step_tree.column("status", width=120, stretch=False)
        self.step_tree.grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))

        for step_id, label in self._progress_steps_for_task(self.task_var.get()):
            self.step_tree.insert("", tk.END, iid=step_id, text=label, values=("대기",))

    def _build_hardware_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="PC 장치", style="Panel.TLabelframe")
        frame.pack(fill=tk.X, pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        self.hardware_tree = ttk.Treeview(frame, columns=("value",), show="tree headings", height=5)
        self.hardware_tree.heading("#0", text="항목")
        self.hardware_tree.heading("value", text="내용")
        self.hardware_tree.column("#0", width=150, stretch=False)
        self.hardware_tree.column("value", width=520, stretch=True)
        self.hardware_tree.grid(row=0, column=0, sticky=tk.EW)

        ttk.Button(frame, text="장치 새로고침", command=self.refresh_hardware).grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="작업 기록", style="Panel.TLabelframe")
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 10),
            background="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def choose_media_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=MEDIA_FILE_TYPES)
        if not path:
            return

        media_path = Path(path)
        self.media_path_var.set(str(media_path))
        self.media_folder_var.set(str(media_path.parent))
        if not self.output_dir_var.get():
            self.output_dir_var.set(str(media_path.parent))
        self._fill_media_table([media_path])
        self._set_step("file", "완료")

    def choose_media_folder(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return

        folder_path = Path(path)
        self.media_folder_var.set(str(folder_path))
        if not self.output_dir_var.get():
            self.output_dir_var.set(str(folder_path))

        media_files = media_files_in_folder(folder_path)
        self._fill_media_table(media_files)
        if media_files:
            self.media_path_var.set(str(media_files[0]))
            self._set_step("file", "완료")
            self._append_log(f"동영상/음성 파일 {len(media_files)}개를 찾았습니다.")
        else:
            self.media_path_var.set("")
            self._append_log("선택한 폴더에서 지원하는 동영상/음성 파일을 찾지 못했습니다.")

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def choose_subtitle_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=SUBTITLE_FILE_TYPES)
        if path:
            self.subtitle_path_var.set(path)
            self.translate_output_path_var.set("")
            self.convert_output_path_var.set("")

    def choose_translate_output_file(self) -> None:
        source_path = self._selected_subtitle_path()
        target_code = language_code_from_display(self.translate_to_var.get())
        initial_dir = source_path.parent if source_path else Path.cwd()
        initial_file = self._default_subtitle_name(source_path, target_code, "srt")
        path = filedialog.asksaveasfilename(
            defaultextension=".srt",
            filetypes=[("SRT subtitle", "*.srt"), ("All files", "*.*")],
            initialdir=str(initial_dir),
            initialfile=initial_file,
        )
        if path:
            self.translate_output_path_var.set(str(ensure_subtitle_extension(Path(path), "srt")))

    def choose_convert_output_file(self) -> None:
        output_format = self.convert_format_var.get()
        source_path = self._selected_subtitle_path()
        initial_dir = source_path.parent if source_path else Path.cwd()
        initial_file = self._default_subtitle_name(source_path, output_format, output_format)
        extension = subtitle_extension(output_format)
        path = filedialog.asksaveasfilename(
            defaultextension=extension,
            filetypes=SUBTITLE_SAVE_TYPES,
            initialdir=str(initial_dir),
            initialfile=initial_file,
        )
        if path:
            self.convert_output_path_var.set(str(ensure_subtitle_extension(Path(path), output_format)))

    def _fill_media_table(self, media_files: list[Path]) -> None:
        for item in self.media_tree.get_children():
            self.media_tree.delete(item)

        self.media_table_paths.clear()
        self.selected_media_iids.clear()

        for index, media_path in enumerate(media_files, start=1):
            iid = f"media-{index}"
            self.media_table_paths[iid] = media_path
            self.selected_media_iids.add(iid)
            try:
                size_label = format_file_size(media_path.stat().st_size)
            except OSError:
                size_label = "확인 필요"

            self.media_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=("☑", media_path.name, media_path.suffix.lower(), size_label, ""),
            )

    def _on_media_tree_click(self, event: tk.Event) -> None:
        if self.media_tree.identify_region(event.x, event.y) != "cell":
            return

        if self.media_tree.identify_column(event.x) != "#1":
            return

        iid = self.media_tree.identify_row(event.y)
        if not iid:
            return

        self._toggle_media_iid(iid)

    def _toggle_media_iid(self, iid: str) -> None:
        if iid in self.selected_media_iids:
            self.selected_media_iids.remove(iid)
            self.media_tree.set(iid, "selected", "☐")
        else:
            self.selected_media_iids.add(iid)
            self.media_tree.set(iid, "selected", "☑")

    def select_all_media_files(self) -> None:
        for iid in self.media_table_paths:
            self.selected_media_iids.add(iid)
            self.media_tree.set(iid, "selected", "☑")

    def clear_media_file_selection(self) -> None:
        self.selected_media_iids.clear()
        for iid in self.media_table_paths:
            self.media_tree.set(iid, "selected", "☐")

    def apply_recommended_settings(self) -> None:
        settings, reason = recommend_transcribe_settings()
        self.model_var.set(str(settings["model"]))
        self.device_var.set(DEVICE_OPTIONS.get(str(settings["device"]), str(settings["device"])))
        self.compute_var.set(str(settings["compute"]))
        self.beam_var.set(int(settings["beam"]))
        self._append_log(reason)
        self.refresh_model_status()

    def refresh_model_status(self) -> None:
        model = self.model_var.get()
        if whisper_model_is_downloaded(model):
            self.model_status_var.set(f"✓ {model} 모델 받음")
        else:
            self.model_status_var.set(f"✗ {model} 모델 없음 — '받기'로 미리 받을 수 있어요")

    def download_selected_model(self) -> None:
        model = self.model_var.get()
        if whisper_model_is_downloaded(model):
            messagebox.showinfo(APP_NAME, f"{model} 모델은 이미 받아져 있습니다.")
            return

        def work() -> None:
            self._emit("status", f"{model} 모델 다운로드 중")
            download_whisper_model(model, lambda line: self._emit("log", line))
            self._emit("status", "대기 중")
            self._emit("model_status", None)
            self._emit("done", f"{model} 모델 다운로드가 끝났습니다.")

        self._run_worker(work)

    def refresh_nllb_status(self) -> None:
        if nllb_model_is_downloaded():
            self.nllb_status_var.set("✓ NLLB 모델 준비됨")
        else:
            self.nllb_status_var.set("✗ NLLB 모델 없음 — 고품질 번역 전 '준비' 필요")

    def prepare_nllb_model(self) -> None:
        if nllb_model_is_downloaded():
            messagebox.showinfo(APP_NAME, "NLLB 모델은 이미 준비되어 있습니다.")
            return

        def work() -> None:
            self._emit("status", "NLLB 모델 준비 중")
            self._emit("busy_bar", True)
            try:
                ensure_nllb_model(lambda line: self._emit("log", line))
            finally:
                self._emit("busy_bar", False)
            self._emit("status", "대기 중")
            self._emit("nllb_status", None)
            self._emit("done", "NLLB 모델 준비가 끝났습니다.")

        self._run_worker(work)

    def start_work(self) -> None:
        task = self.task_var.get()
        if task == "make":
            self.make_subtitle()
        elif task == "extract":
            self.extract_embedded_subtitle()
        elif task == "translate":
            self.translate_subtitle()
        elif task == "convert":
            self.convert_subtitle()
        elif task == "engine":
            messagebox.showinfo(APP_NAME, "엔진 관리는 패널 안의 '엔진 설치' 또는 '엔진 확인' 버튼을 눌러주세요.")

    def _on_task_changed(self, *args) -> None:
        # 모든 설정 패널 숨기기
        if not hasattr(self, "transcribe_frame"):
            return  # UI가 아직 완전히 빌드되지 않음
        self.transcribe_frame.pack_forget()
        self.extract_frame.pack_forget()
        self.translate_frame.pack_forget()
        self.convert_frame.pack_forget()
        self.engine_frame.pack_forget()

        task = self.task_var.get()

        # 선택된 작업에 맞는 패널 보여주기 및 실행 버튼 텍스트 변경
        if task == "make":
            self.transcribe_frame.pack(fill=tk.X, expand=True)
            self.action_button.config(text="자막 만들기", state=tk.NORMAL)
        elif task == "extract":
            self.extract_frame.pack(fill=tk.X, expand=True)
            self.action_button.config(text="영상 속 자막 추출", state=tk.NORMAL)
        elif task == "translate":
            self.translate_frame.pack(fill=tk.X, expand=True)
            self.action_button.config(text="번역 자막 저장", state=tk.NORMAL)
        elif task == "convert":
            self.convert_frame.pack(fill=tk.X, expand=True)
            self.action_button.config(text="자막 변환 저장", state=tk.NORMAL)
        elif task == "engine":
            self.engine_frame.pack(fill=tk.X, expand=True)
            self.action_button.config(text="위의 엔진 버튼을 사용하세요", state=tk.DISABLED)

        self._set_progress_steps(task)

    def stop_current_work(self) -> None:
        if not self.busy:
            self._append_log("진행 중인 작업이 없습니다.")
            return

        self.cancel_event.set()
        self.status_var.set("중지 요청 중")
        self._append_log("작업 중지를 요청했습니다. 현재 처리 중인 작은 단계를 마친 뒤 멈춥니다.")

    def refresh_engines(self) -> None:
        def work() -> None:
            self._emit("status", "엔진 상태 확인 중")
            status = check_engine_status()
            self._emit("engines", status)
            self._emit("step", ("engine", "완료"))
            self._emit("status", "대기 중")

        self._run_worker(work)

    def refresh_hardware(self) -> None:
        def work() -> None:
            self._emit("status", "PC 장치 확인 중")
            hardware = detect_hardware()
            self._emit("hardware", hardware)
            self._emit("status", "대기 중")

        self._run_worker(work, allow_when_busy=True)

    def install_engines(self) -> None:
        def work() -> None:
            self._emit("status", "엔진 설치 중")
            self._emit("progress", 5)
            install_python_engines(lambda line: self._emit("log", line))
            self._emit("progress", 100)
            self._emit("engines", check_engine_status())
            self._emit("step", ("engine", "완료"))
            self._emit("status", "대기 중")
            self._emit("done", "엔진 설치가 끝났습니다. 앱을 다시 실행하면 더 안정적입니다.")

        self._run_worker(work)

    def _check_and_prompt_for_initial_setup(self) -> None:
        """앱 시작 시 필수 엔진이 없으면 사용자에게 설치를 안내합니다."""
        # EXE 패키지는 엔진이 포함되어 있으므로 이 검사를 건너뜁니다.
        if is_frozen_app():
            return

        # 핵심 엔진인 faster-whisper가 있는지 확인합니다.
        if has_module("faster_whisper"):
            return

        # 엔진이 없으면 사용자에게 설치 여부를 묻습니다.
        response = messagebox.askyesno(
            APP_NAME,
            "PapaStop을 사용하려면 초기 설정이 필요합니다.\n\n"
            "필수 자막/번역 엔진을 지금 설치하시겠습니까?\n"
            "(인터넷 연결이 필요하며, 수 분 정도 걸릴 수 있습니다.)",
        )

        if response:
            self.install_engines()

    def _on_mode_changed(self, *args) -> None:
        """'전문가 모드' 상태에 따라 고급 설정 위젯의 표시 여부를 변경합니다."""
        if not hasattr(self, "device_label"):
            return  # UI가 아직 완전히 빌드되지 않음

        is_expert = self.expert_mode_var.get()

        expert_widgets = [
            self.device_label,
            self.device_combo,
            self.compute_label,
            self.compute_combo,
            self.beam_label,
            self.beam_spinbox,
            self.max_line_chars_label,
            self.max_line_chars_spinbox,
            self.recommend_button,
        ]

        for widget in expert_widgets:
            if is_expert:
                widget.grid()
            else:
                widget.grid_remove()

    def _on_overwrite_changed(self, *args) -> None:
        """번역 덮어쓰기 옵션이 바뀌면 관련 UI 상태를 업데이트합니다."""
        if not hasattr(self, "translate_output_entry"):
            return  # UI가 아직 완전히 빌드되지 않음
        state = tk.DISABLED if self.translate_overwrite_var.get() else tk.NORMAL
        self.translate_output_entry.config(state=state)
        self.translate_output_button.config(state=state)

    def _update_media_tree_with_results(self, results: dict[str, tuple[str, str]]) -> None:
        """작업 결과를 파일 목록 Treeview에 시각적으로 반영합니다."""
        for iid, (status, message) in results.items():
            if not self.media_tree.exists(iid):
                continue
            if status == "success":
                self.media_tree.set(iid, "selected", "✅")
                self.media_tree.set(iid, "status", "성공")
            elif status == "error":
                self.media_tree.set(iid, "selected", "❌")
                self.media_tree.set(iid, "status", "오류")

    def make_subtitle(self) -> None:
        try:
            jobs = self._selected_media_jobs()
        except ValueError as exc:
            messagebox.showwarning(APP_NAME, str(exc))
            return

        # 작업 시작 전, 대상 파일들의 상태를 '대기 중'으로 초기화합니다.
        path_to_iid = {v: k for k, v in self.media_table_paths.items()}
        for media_path, _ in jobs:
            iid = path_to_iid.get(media_path)
            if iid and self.media_tree.exists(iid):
                self.media_tree.set(iid, "selected", "☑")
                self.media_tree.set(iid, "status", "대기 중...")

        auto_translate = bool(self.auto_translate_var.get())
        translate_to = language_code_from_display(self.auto_translate_to_var.get())
        source_lang = language_code_from_display(self.source_lang_var.get())
        engine = translation_engine_code(self.translate_engine_var.get())

        def work() -> None:
            self._emit("status", "자막 만드는 중")
            if not auto_translate:
                self._emit("step", ("translate", "사용 안 함"))
            self._emit("step", ("model", "진행 중"))
            path_to_iid = {v: k for k, v in self.media_table_paths.items()}
            job_results: dict[str, tuple[str, str]] = {}
            saved_paths: list[Path] = []
            translated_paths: list[Path] = []
            errors: list[str] = []
            total = len(jobs)
            subtitle_weight = 0.6 if auto_translate else 1.0

            ensure_not_cancelled(self.cancel_event.is_set)
            model = load_whisper_model(
                self.model_var.get(),
                device_code(self.device_var.get()),
                self.compute_var.get(),
                lambda line: self._emit("log", line),
            )
            self._emit("step", ("model", "완료"))

            for index, (media_path, output_dir) in enumerate(jobs, start=1):
                ensure_not_cancelled(self.cancel_event.is_set)
                iid = path_to_iid.get(media_path)
                self._emit("log", f"[{index}/{total}] 자막 생성 시작: {media_path.name}")
                try:
                    output_path, detected_language = transcribe_media(
                        media_path=media_path,
                        output_dir=output_dir,
                        model=model,
                        language_code=source_lang,
                        beam_size=int(self.beam_var.get()),
                        on_line=lambda line: self._emit("log", line),
                        on_progress=lambda value, item_index=index: self._emit(
                            "progress",
                            ((item_index - 1) + (value / 100) * subtitle_weight) / total * 100,
                        ),
                        should_stop=self.cancel_event.is_set,
                        max_line_chars=int(self.max_line_chars_var.get()),
                    )
                    saved_paths.append(output_path)
                    self._emit("subtitle_path", str(output_path))
                    if iid:
                        job_results[iid] = ("success", f"{output_path.name} 생성 완료")

                    if auto_translate:
                        from_code = source_lang if source_lang != "auto" else detected_language
                        if from_code not in TRANSLATION_LANGUAGES:
                            self._emit(
                                "log",
                                f"[{index}/{total}] 번역할 수 없는 원본 언어({from_code})라 "
                                f"번역을 건너뜁니다: {media_path.name}",
                            )
                        elif from_code == translate_to:
                            self._emit(
                                "log",
                                f"[{index}/{total}] 원본과 번역 언어가 같아 번역을 건너뜁니다: {media_path.name}",
                            )
                        else:
                            self._emit("step", ("translate", "진행 중"))
                            translated_path = output_path.with_name(
                                f"{output_path.stem}.{translate_to}.srt"
                            )
                            self._emit(
                                "log",
                                f"[{index}/{total}] 번역 시작: "
                                f"{language_label(from_code)} -> {language_label(translate_to)}",
                            )
                            translate_srt_file(
                                source_path=output_path,
                                target_path=translated_path,
                                from_code=from_code,
                                to_code=translate_to,
                                on_line=lambda line: self._emit("log", line),
                                on_progress=lambda value, item_index=index: self._emit(
                                    "progress",
                                    (
                                        (item_index - 1)
                                        + subtitle_weight
                                        + (value / 100) * (1 - subtitle_weight)
                                    )
                                    / total
                                    * 100,
                                ),
                                should_stop=self.cancel_event.is_set,
                                engine=engine,
                            )
                            translated_paths.append(translated_path)
                            self._emit("subtitle_path", str(translated_path))
                            self._emit("step", ("translate", "완료"))
                except OperationCancelled:
                    raise
                except Exception as exc:
                    errors.append(f"{media_path.name}: {exc}")
                    self._emit("log", f"오류: {media_path.name} - {exc}")
                    if iid:
                        job_results[iid] = ("error", str(exc))

            if not saved_paths and errors:
                raise RuntimeError("선택한 파일에서 자막을 만들지 못했습니다.\n" + "\n".join(errors[:3]))

            self._emit("step", ("audio", "완료"))
            self._emit("step", ("subtitle", "완료"))
            self._emit("status", "대기 중")
            self._emit("job_results", job_results)
            message = f"자막 {len(saved_paths)}개를 저장했습니다."
            if translated_paths:
                message += f"\n번역 자막 {len(translated_paths)}개도 저장했습니다."
            if errors:
                message += f"\n오류 {len(errors)}개는 작업 기록을 확인해 주세요."
            self._emit("done", message)

        self._reset_progress()
        self._set_step("file", "완료")
        self._set_step("engine", "진행 중")
        self._run_worker(work)

    def extract_embedded_subtitle(self) -> None:
        try:
            media_path, output_dir = self._validated_media_and_output()
        except ValueError as exc:
            messagebox.showwarning(APP_NAME, str(exc))
            return

        def work() -> None:
            self._emit("status", "기존 자막 추출 중")
            output_path = extract_embedded_subtitle(
                media_path=media_path,
                output_dir=output_dir,
                on_line=lambda line: self._emit("log", line),
                on_progress=lambda value: self._emit("progress", value),
            )
            self._emit("step", ("subtitle", "완료"))
            self._emit("subtitle_path", str(output_path))
            self._emit("status", "대기 중")
            self._emit("done", f"기존 자막을 저장했습니다.\n{output_path}")

        self._reset_progress()
        self._set_step("file", "완료")
        self._run_worker(work)

    def translate_subtitle(self) -> None:
        source_path = self._selected_subtitle_path()
        if source_path is None:
            messagebox.showwarning(APP_NAME, "번역할 SRT 파일을 먼저 선택해 주세요.")
            return

        source_code = language_code_from_display(self.translate_from_var.get())
        target_code = language_code_from_display(self.translate_to_var.get())
        engine = translation_engine_code(self.translate_engine_var.get())
        overwrite = bool(self.translate_overwrite_var.get())
        if source_code == target_code:
            messagebox.showwarning(APP_NAME, "원본 언어와 번역 언어가 같습니다.")
            return

        if overwrite:
            target_path = source_path
        else:
            target_path = self._subtitle_target_path(
                source_path,
                self.translate_output_path_var.get(),
                target_code,
                "srt",
            )

        if overwrite and not messagebox.askyesno(
            APP_NAME,
            "원본 SRT 파일 내용을 번역 결과로 바꿉니다.\n"
            "원본 파일을 따로 보관하지 않았다면 되돌리기 어렵습니다.\n\n"
            "계속하시겠습니까?",
        ):
            return

        def work() -> None:
            self._emit("status", "번역 저장 중")
            self._emit("step", ("translate", "진행 중"))
            translate_srt_file(
                source_path=source_path,
                target_path=target_path,
                from_code=source_code,
                to_code=target_code,
                on_line=lambda line: self._emit("log", line),
                on_progress=lambda value: self._emit("progress", value),
                engine=engine,
            )
            self._emit("step", ("translate", "완료"))
            self._emit("status", "대기 중")
            self._emit("done", f"번역 자막을 저장했습니다.\n{target_path}")

        self._reset_progress()
        self._run_worker(work)

    def translate_subtitle_folder(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return

        source_code = language_code_from_display(self.translate_from_var.get())
        target_code = language_code_from_display(self.translate_to_var.get())
        engine = translation_engine_code(self.translate_engine_var.get())
        overwrite = bool(self.translate_overwrite_var.get())
        if source_code == target_code:
            messagebox.showwarning(APP_NAME, "원본 언어와 번역 언어가 같습니다.")
            return

        srt_files = [
            path
            for path in sorted(Path(folder).glob("*.srt"))
            if not path.stem.endswith(f".{target_code}")
        ]
        if not srt_files:
            messagebox.showinfo(APP_NAME, "폴더에서 번역할 SRT 파일을 찾지 못했습니다.")
            return

        if overwrite and not messagebox.askyesno(
            APP_NAME,
            f"폴더 안의 SRT 파일 {len(srt_files)}개를 번역 결과로 바꿉니다.\n"
            "원본 파일을 따로 보관하지 않았다면 되돌리기 어렵습니다.\n\n"
            "계속하시겠습니까?",
        ):
            return

        def work() -> None:
            self._emit("status", "폴더 일괄 번역 중")
            self._emit("step", ("translate", "진행 중"))
            saved_paths: list[Path] = []
            errors: list[str] = []
            total = len(srt_files)

            for index, srt_path in enumerate(srt_files, start=1):
                ensure_not_cancelled(self.cancel_event.is_set)
                self._emit("log", f"[{index}/{total}] 번역 시작: {srt_path.name}")
                if overwrite:
                    target_path = srt_path
                else:
                    target_path = srt_path.with_name(f"{srt_path.stem}.{target_code}.srt")
                try:
                    translate_srt_file(
                        source_path=srt_path,
                        target_path=target_path,
                        from_code=source_code,
                        to_code=target_code,
                        on_line=lambda line: self._emit("log", line),
                        on_progress=lambda value, item_index=index: self._emit(
                            "progress",
                            ((item_index - 1) + value / 100) / total * 100,
                        ),
                        should_stop=self.cancel_event.is_set,
                        engine=engine,
                    )
                    saved_paths.append(target_path)
                    self._emit("subtitle_path", str(target_path))
                except OperationCancelled:
                    raise
                except Exception as exc:
                    errors.append(f"{srt_path.name}: {exc}")
                    self._emit("log", f"오류: {srt_path.name} - {exc}")

            if not saved_paths and errors:
                raise RuntimeError("폴더에서 번역에 성공한 파일이 없습니다.\n" + "\n".join(errors[:3]))

            self._emit("step", ("translate", "완료"))
            self._emit("status", "대기 중")
            message = f"폴더에서 자막 {len(saved_paths)}개를 번역했습니다."
            if errors:
                message += f"\n오류 {len(errors)}개는 작업 기록을 확인해 주세요."
            self._emit("done", message)

        self._reset_progress()
        self._run_worker(work)

    def convert_subtitle(self) -> None:
        source_path = self._selected_subtitle_path()
        if source_path is None:
            messagebox.showwarning(APP_NAME, "변환할 SRT 파일을 먼저 선택해 주세요.")
            return

        output_format = self.convert_format_var.get()
        target_path = self._subtitle_target_path(
            source_path,
            self.convert_output_path_var.get(),
            output_format,
            output_format,
        )

        def work() -> None:
            self._emit("status", "자막 변환 중")
            self._emit("step", ("subtitle", "진행 중"))
            saved_path = convert_subtitle_file(
                source_path=source_path,
                target_path=target_path,
                output_format=output_format,
                on_line=lambda line: self._emit("log", line),
                on_progress=lambda value: self._emit("progress", value),
            )
            self._emit("step", ("subtitle", "완료"))
            self._emit("status", "대기 중")
            self._emit("done", f"변환 자막을 저장했습니다.\n{saved_path}")

        self._reset_progress()
        self._run_worker(work)

    def _validated_media_and_output(self) -> tuple[Path, Path]:
        media_path = Path(self.media_path_var.get().strip())
        if not media_path.exists() or not media_path.is_file():
            raise ValueError("영상 또는 음성 파일을 먼저 선택해 주세요.")

        output_dir_text = self.output_dir_var.get().strip()
        output_dir = Path(output_dir_text) if output_dir_text else media_path.parent
        return media_path, output_dir

    def _selected_media_jobs(self) -> list[tuple[Path, Path]]:
        selected_paths = [
            self.media_table_paths[iid]
            for iid in self.media_table_paths
            if iid in self.selected_media_iids
        ]

        if not selected_paths:
            media_path = Path(self.media_path_var.get().strip())
            if media_path.exists() and media_path.is_file():
                selected_paths = [media_path]

        if not selected_paths:
            raise ValueError("자막을 만들 동영상 또는 음성 파일을 먼저 선택해 주세요.")

        output_dir_text = self.output_dir_var.get().strip()
        jobs: list[tuple[Path, Path]] = []
        for media_path in selected_paths:
            output_dir = Path(output_dir_text) if output_dir_text else media_path.parent
            jobs.append((media_path, output_dir))
        return jobs

    def _selected_subtitle_path(self) -> Path | None:
        path_text = self.subtitle_path_var.get().strip()
        if not path_text:
            return None

        path = Path(path_text)
        if path.exists() and path.is_file():
            return path
        return None

    def _default_subtitle_name(self, source_path: Path | None, suffix: str, format_code: str) -> str:
        stem = source_path.stem if source_path else "subtitle"
        return f"{stem}.{suffix}{subtitle_extension(format_code)}"

    def _subtitle_target_path(
        self,
        source_path: Path,
        chosen_path: str,
        suffix: str,
        format_code: str,
    ) -> Path:
        if chosen_path.strip():
            return ensure_subtitle_extension(Path(chosen_path.strip()), format_code)

        output_dir_text = self.output_dir_var.get().strip()
        output_dir = Path(output_dir_text) if output_dir_text else source_path.parent
        return output_dir / self._default_subtitle_name(source_path, suffix, format_code)

    def _run_worker(self, target: Callable[[], None], allow_when_busy: bool = False) -> None:
        if self.busy and not allow_when_busy:
            messagebox.showinfo(APP_NAME, "이미 작업이 진행 중입니다.")
            return

        if not allow_when_busy:
            self.cancel_event.clear()
            self.busy = True

        def wrapped() -> None:
            started = time.time()
            try:
                target()
            except OperationCancelled as exc:
                self._emit("cancelled", str(exc))
            except Exception as exc:
                self._emit("error", str(exc))
            finally:
                if not allow_when_busy:
                    self._emit("busy", False)
                    self._emit("log", f"작업 시간: {time.time() - started:.1f}초")

        threading.Thread(target=wrapped, daemon=True).start()

    def _emit(self, kind: str, payload: object) -> None:
        self.events.put((kind, payload))

    def _process_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "progress":
                self.progress_var.set(float(payload))
            elif kind == "step":
                step_id, status = payload  # type: ignore[misc]
                self._set_step(str(step_id), str(status))
            elif kind == "hardware":
                self._fill_tree(self.hardware_tree, payload)  # type: ignore[arg-type]
            elif kind == "engines":
                self._fill_tree(self.engine_tree, payload)  # type: ignore[arg-type]
            elif kind == "subtitle_path":
                self.subtitle_path_var.set(str(payload))
            elif kind == "model_status":
                self.refresh_model_status()
            elif kind == "nllb_status":
                self.refresh_nllb_status()
            elif kind == "busy_bar":
                if payload:
                    self.progress_bar.configure(mode="indeterminate")
                    self.progress_bar.start(15)
                else:
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.progress_var.set(0)
            elif kind == "busy":
                self.busy = bool(payload)
            elif kind == "done":
                messagebox.showinfo(APP_NAME, str(payload))
            elif kind == "cancelled":
                self.busy = False
                self.status_var.set("중지됨")
                self._append_log(str(payload))
            elif kind == "error":
                self.busy = False
                self.status_var.set("오류 발생")
                self._append_log(f"오류: {payload}")
                messagebox.showerror(APP_NAME, str(payload))
            elif kind == "job_results":
                self._update_media_tree_with_results(payload)  # type: ignore[arg-type]

        self.after(100, self._process_events)

    def _append_log(self, message: str) -> None:
        if not message:
            return

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.NORMAL)

    def _fill_tree(self, tree: ttk.Treeview, values: dict[str, str]) -> None:
        for item in tree.get_children():
            tree.delete(item)

        for key, value in values.items():
            tree.insert("", tk.END, text=key, values=(value,))

    def _progress_steps_for_task(self, task: str) -> tuple[tuple[str, str], ...]:
        steps_by_task: dict[str, tuple[tuple[str, str], ...]] = {
            "make": (
                ("file", "파일 선택"),
                ("engine", "자막 도구 확인"),
                ("model", "자막 품질 준비"),
                ("audio", "소리 분석"),
                ("subtitle", "자막 저장"),
                ("translate", "번역 저장"),
            ),
            "extract": (
                ("file", "파일 선택"),
                ("engine", "영상 도구 확인"),
                ("subtitle", "기존 자막 저장"),
            ),
            "translate": (
                ("file", "SRT 파일 선택"),
                ("engine", "번역 도구 확인"),
                ("translate", "번역 저장"),
            ),
            "convert": (
                ("file", "SRT 파일 선택"),
                ("subtitle", "변환 파일 저장"),
            ),
            "engine": (
                ("engine", "엔진 확인/설치"),
            ),
        }
        return steps_by_task.get(task, steps_by_task["make"])

    def _set_progress_steps(self, task: str) -> None:
        if not hasattr(self, "step_tree"):
            return

        for item in self.step_tree.get_children():
            self.step_tree.delete(item)

        for step_id, label in self._progress_steps_for_task(task):
            self.step_tree.insert("", tk.END, iid=step_id, text=label, values=("대기",))

    def _set_step(self, step_id: str, status: str) -> None:
        if self.step_tree.exists(step_id):
            self.step_tree.set(step_id, "status", status)

    def _reset_progress(self) -> None:
        self.progress_var.set(0)
        for step_id in self.step_tree.get_children():
            self.step_tree.set(step_id, "status", "대기")


def main() -> None:
    if "--self-test" in sys.argv:
        output_path = None
        argument_index = sys.argv.index("--self-test")
        if len(sys.argv) > argument_index + 1:
            output_path = Path(sys.argv[argument_index + 1])
        run_self_test(output_path)
        return

    app = PapaStopApp()
    app.mainloop()


def run_self_test(output_path: Path | None) -> None:
    engine_status = check_engine_status()
    import_results: dict[str, str] = {}

    for package_name, module_name in ENGINE_PACKAGES.items():
        try:
            __import__(module_name)
            import_results[package_name] = "ok"
        except Exception as exc:
            import_results[package_name] = f"error: {exc}"

    app = PapaStopApp()
    app.update_idletasks()
    window_size = [app.winfo_reqwidth(), app.winfo_reqheight()]
    app.destroy()

    result = {
        "app": APP_NAME,
        "frozen": is_frozen_app(),
        "app_dir": str(APP_DIR),
        "engine_status": engine_status,
        "engine_imports": import_results,
        "window_size": window_size,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
