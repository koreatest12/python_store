# python_store

파이썬 학습, 예제 및 실험용 저장소입니다.

## Python 버전

이 저장소는 **Python 3.14.7**을 기준으로 구성되어 있습니다.

- `.python-version`: 로컬 Python 버전 고정
- `pyproject.toml`: Python 3.14.x 요구
- GitHub Actions: Python 3.14.7 자동 설치 및 실행 검증

## Windows 설치

Python 공식 설치 관리자를 사용하거나 Python.org에서 Python 3.14.7을 설치한 뒤 다음 명령으로 확인합니다.

```powershell
python --version
```

정상 설치 예시:

```text
Python 3.14.7
```

가상환경 생성 및 실행:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python main.py
```

## macOS / Linux

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python main.py
```

## 실행

```bash
python main.py
```

프로그램은 현재 사용 중인 Python 버전과 실행 파일 경로를 출력합니다.

## CI

`.github/workflows/python-ci.yml`에서 다음을 자동 검증합니다.

1. Python 3.14.7 설치
2. Python 버전 출력
3. pip / setuptools / wheel 업그레이드
4. `main.py` 스모크 테스트
5. Python 소스 컴파일 검사
