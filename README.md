# PapaStop

> A Windows desktop app that generates subtitles from audio/video and shows live translated captions.
> Runs speech recognition (faster-whisper) and translation (NLLB) locally on your PC — no cloud API needed.

음성·영상에서 자막을 만들고, PC에서 나오는 소리를 실시간으로 받아 **한국어 번역 자막**으로 보여 주는 Windows 데스크톱 앱입니다.
음성 인식과 번역을 모두 내 PC 안에서 처리하므로 외부 서비스 가입이나 API 키가 필요 없습니다.

---

## 주요 기능

- **파일 자막** — 미디어 파일(또는 URL)의 음성을 글자로 받아 적어 `.srt` 자막 파일로 저장합니다.
- **실시간 자막** — PC 스피커로 나오는 소리나 인터넷 방송 주소를 받아, 번역된 자막을 바로 띄웁니다.
- **화면 녹화** — 크롬 등에서 재생 중인 영상을 무손실 MKV로 저장합니다. 스피커 소리도 함께 녹음할 수 있습니다.
- **시스템 사양 확인** — CPU, RAM, 그래픽카드 정보를 첫 화면에서 보여 줍니다.
- **모델 미리 받기** — 처음 쓰는 음성 인식·번역 모델을 버튼 하나로 내려받아 준비합니다.

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 화면 | Python · CustomTkinter |
| 음성 인식 | faster-whisper (CTranslate2) |
| 번역 | NLLB (Transformers / CTranslate2), Argos Translate 폴백 |
| 오디오·영상 | soundcard (PC 소리 캡처) · ffmpeg |
| GPU 가속 | CUDA 12 (PyTorch, cuBLAS, cuDNN) |
| 배포 | PyInstaller (일반판 / 최소판) |

## 빠른 시작

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python papastop.py
```

- 실행 환경: Windows · Python 3.10 이상 · ffmpeg (없으면 `imageio-ffmpeg`로 대신합니다)
- 일괄 실행 스크립트: `run_papastop.bat`, 엔진 설치: `install_engines.bat`

## 더 알아보기

화면별 사용 순서, 배포본 만드는 법(일반판·최소판), 점검 체크리스트는 **[docs/GUIDE.md](docs/GUIDE.md)** 에 정리해 두었습니다.

---

만든 사람: [hunmonkey](https://github.com/hunmonkey)
