# PapaStop

음성/영상에서 자막을 만들고, 실시간 자막을 보여주는 Windows 전용 데스크톱 앱입니다.  
현재 릴리스는 핵심 기능을 중심으로 유지보수하도록 정리했습니다.

- **파일 자막**: 미디어 파일(또는 URL) 오디오에서 텍스트를 추출해 `.srt` 생성
- **실시간 자막**: PC 스피커 출력 또는 인터넷 스트림 오디오를 실시간으로 받아 한국어 번역 결과를 미리보기로 표시
- **화면 녹화**: 크롬 등 화면에 재생 중인 영상을 무손실 MKV로 저장
- **시스템 사양**: CPU, RAM, VGA 정보를 앱 첫 탭에서 확인

---

## 현재 동작 흐름

### 1) 시스템 사양 확인
1. 앱을 실행하면 첫 번째 `시스템 사양` 탭이 열립니다.
2. CPU, RAM, VGA 정보를 확인합니다.
3. 필요하면 `사양 새로고침`을 클릭합니다.

### 2) 파일 자막 생성
1. `파일 자막` 탭에서 미디어 경로 입력/선택
2. 출력 `.srt` 경로를 지정(비워두면 원본 파일명 기반 생성)
3. 필요하면 `음성 인식 모델`과 `번역 모델` 선택
4. 처음 쓰는 모델은 `위스퍼 모델 다운로드`, `번역 모델 다운로드`로 미리 준비
5. `자막 만들기 시작` 클릭
6. `실시간 추출 미리보기`에서 추출되는 문장 확인
7. 완료 시 저장된 경로 표시

### 3) 실시간 자막
1. `실시간 자막` 탭에서 소스 선택
   - `PC 스피커 소리` : 현재 PC 출력 오디오 사용
   - `인터넷 URL 스트림` : 영상/방송 URL 사용
2. `시작` 클릭
3. 하단 미리보기에서 번역 결과 확인
4. `중지`로 종료

### 4) 화면 녹화
1. `화면 녹화` 탭에서 저장할 MKV 경로 확인
2. 필요하면 `PC 스피커 소리도 녹음` 체크
3. `녹화 시작` 클릭
4. 녹화를 끝낼 때 `중지` 클릭

---

## 설치 및 실행

### 실행 환경
- Python 3.10+
- Windows (실시간 PC 스피커 캡처는 Windows 전용)
- `ffmpeg` (PATH 또는 `imageio-ffmpeg` 자동 대체)

### 설치
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 앱 실행
```powershell
. .\.venv\Scripts\Activate.ps1
python papastop.py
```

### 일괄 실행 스크립트
- `run_papastop.bat` / `run_papastop.ps1`
- `install_engines.bat` / `install_engines.ps1`

---

## 패키징

`build_package.bat` 또는 `build_package.ps1` 실행 시 단일 EXE를 생성합니다.
`dist\PapaStop.exe`가 만들어지며, CUDA/모델 캐시 일부는 필요 시 `dist\.papastop_data`에 함께 생성됩니다.

---

## 의존성

- `faster-whisper` : 음성 인식
- `transformers` : NLLB 번역 (fallback 처리 포함)
- `argostranslate` : 번역 폴백
- `soundcard` : PC 스피커 실시간 캡처
- `imageio-ffmpeg` : ffmpeg 바인딩/대체
- `numpy`, `torch`

---

## 폴더 정리(현재 반영)

레포 동작 확인 후 불필요한 산출물은 아래처럼 정리했습니다.

- 삭제 대상(이번 정리 반영):
  - `dist`, `build`, `release`, `logs`, `recordings`
- 삭제 이유:
  - 빌드 산출물, 실행 로그, 임시 녹음 파일은 매 실행/빌드 시 재생성 가능해 소스 정합성 점검엔 불필요

---

## 변경 후 정합성 점검 체크리스트

아래 항목은 코드 변경 후 바로 확인할 수 있는 최소 점검 목록입니다.

- 정적 확인
  - `python -m py_compile papastop.py`
- 단위 테스트
  - `python -m unittest discover -s tests -v`
- 실행 점검
  1. 앱 실행 (UI 창 정상 오픈)
  2. `파일 자막`에서 로컬 파일 1개 처리
  3. `실시간 자막`에서 PC 스피커 또는 URL 1개로 `시작`/`중지` 확인
  4. `화면 녹화`에서 짧게 녹화 후 MKV 저장 확인
  5. `README.md`의 환경에서 `ffmpeg` 상태 및 필수 모듈 메시지 정상 표시 확인

---

## 현재 상태

이 버전은 기존의 부가 기능(자막 변환/기존 자막 번역/엔진 설치 UX/단일 인스턴스 보장 등)을 정리하고  
핵심 기능(파일 자막, 실시간 자막, 화면 녹화)에 집중한 단일판입니다.
