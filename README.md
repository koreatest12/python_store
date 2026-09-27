# python_store

파이썬 학습, 예제 및 실험용 저장소입니다.

## Python 버전

이 저장소는 **Python 3.14.7**을 기준으로 구성되어 있습니다.

- `.python-version`: 로컬 Python 버전 고정
- `pyproject.toml`: Python 3.14.x 요구
- GitHub Actions: Python 3.14.7 자동 설치 및 실행 검증

## 토크나이저

`tokenizer.py`에 외부 라이브러리 없이 동작하는 `SimpleTokenizer`가 포함되어 있습니다.

지원 기능:

- Unicode NFKC 정규화
- 한글, 영문, 숫자, 문장부호 토큰 분리
- 선택적 영문 소문자 변환
- Vocabulary 자동 생성
- `<PAD>`, `<UNK>` 특수 토큰
- 문자열 → Token ID 인코딩
- Token ID → 문자열 디코딩
- 최대 길이 기준 padding / truncation
- Vocabulary JSON 저장 / 불러오기

간단한 실행:

```powershell
python main.py "안녕하세요 Python tokenizer 3.14!"
```

Python 코드에서 사용:

```python
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer(lowercase=True)
tokenizer.fit([
    "안녕하세요 Python tokenizer",
    "Python 토크나이저 테스트",
])

tokens = tokenizer.tokenize("안녕하세요 Python!")
ids = tokenizer.encode("안녕하세요 Python!", max_length=8)

print(tokens)
print(ids)

tokenizer.save("tokenizer.json")
restored = SimpleTokenizer.load("tokenizer.json")
```

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

## 테스트

```bash
python -m unittest discover -s tests -v
```

## CI

`.github/workflows/python-ci.yml`에서 다음을 자동 검증합니다.

1. Python 3.14.7 설치
2. Python 버전 출력
3. pip / setuptools / wheel 업그레이드
4. 토크나이저 스모크 테스트
5. 토크나이저 단위 테스트
6. Python 소스 컴파일 검사


## 고급 토크나이저

추가로 `advanced_tokenizers.py`에서 다음 학습형 토크나이저를 제공합니다.

- `BPETokenizer`
- `WordPieceTokenizer`
- `SentencePieceStyleTokenizer`

> 주의: `SentencePieceStyleTokenizer`는 Google SentencePiece 라이브러리 자체가 아니라, 경계 기호(`▁`)와 서브워드 학습 개념을 학습용으로 구현한 경량 버전입니다.

### BPE 학습

```powershell
python train_tokenizer.py bpe examples/corpus.txt --output bpe.json --vocab-size 128
```

### WordPiece 학습

```powershell
python train_tokenizer.py wordpiece examples/corpus.txt --output wordpiece.json --vocab-size 128
```

### SentencePiece 스타일 학습

```powershell
python train_tokenizer.py sentencepiece examples/corpus.txt --output sentencepiece.json --vocab-size 128
```

각 모델 JSON에는 vocabulary와 학습 설정이 저장됩니다.

Python 코드에서 직접 사용할 수도 있습니다.

```python
from advanced_tokenizers import train_tokenizer

corpus = [
    "안녕하세요 파이썬 토크나이저",
    "hello python tokenizer",
]

tokenizer = train_tokenizer(
    "bpe",
    corpus,
    vocab_size=128,
    lowercase=True,
)

print(tokenizer.tokenize("Hello Python"))
print(tokenizer.encode("Hello Python"))
tokenizer.save("bpe.json")
```

## 토크나이저 구조

현재 저장소에는 두 계층의 토크나이저가 있습니다.

1. `SimpleTokenizer`
   - 빠른 규칙 기반 토큰 분리
   - padding / truncation
   - vocabulary 저장 및 복원

2. 학습형 서브워드 토크나이저
   - BPE
   - WordPiece
   - SentencePiece 스타일
   - corpus 기반 vocabulary 학습
   - JSON 모델 저장



## 토큰 다운로드 보관소

`token_repository/`는 학습된 토크나이저 모델과 vocabulary를 버전별로 관리하고 다운로드할 수 있도록 구성한 전용 보관소입니다.

주요 기능:

- BPE / WordPiece / SentencePiece 스타일 / SimpleTokenizer 모델 등록
- 모델 이름과 버전별 디렉터리 관리
- `manifest.json` 메타데이터 관리
- SHA256 무결성 검증
- 다운로드용 ZIP 자동 생성
- GitHub Actions에서 `python-token-repository` Artifact 자동 업로드
- CI 실행 결과의 Artifacts 영역에서 ZIP 및 모델 디렉터리 다운로드 가능

### 모델 등록

```powershell
python token_repository.py add bpe.json --name korean-demo --type bpe --version 1.0.0
```

### 등록 모델 확인

```powershell
python token_repository.py list
```

### 무결성 확인

```powershell
python token_repository.py verify
```

### 다운로드 ZIP 생성

```powershell
python token_repository.py export
```

생성 파일:

```text
token_repository/downloads/python-token-repository.zip
```

GitHub Actions가 실행되면 BPE, WordPiece, SentencePiece 스타일 예제 모델을 자동 학습하고 이 보관소에 등록한 뒤 `python-token-repository`라는 다운로드 Artifact로 제공합니다.


## 사전 토큰화 (Pre-tokenization)

`pretokenizer.py`는 토크나이저 학습이나 모델 입력 전에 텍스트를 정규화하고 1차 토큰 단위로 나누는 기능을 제공합니다.

지원 기능:

- Unicode NFKC 정규화
- 선택적 소문자 변환
- 선택적 accent 제거
- 연속 공백 정리
- 한글 / 영문 / 숫자 / 문장부호 분리
- token offset(start/end) 반환
- 여러 문장 batch pre-tokenization

사용 예시:

```python
from pretokenizer import Pretokenizer

pre = Pretokenizer(lowercase=True)
print(pre.split("안녕하세요 Python 3.14!"))
print(pre.split_with_offsets("Hello world!"))
```

## 데이터셋 기능

`dataset.py`에서 학습용 텍스트 데이터셋을 다룰 수 있습니다.

지원 기능:

- 일반 TXT 파일 로딩
- JSONL 로딩 / 저장
- CSV 로딩 / 저장
- `DatasetRecord(text, label, metadata)`
- train / validation / test 분할
- seed 기반 재현 가능한 shuffle
- batch 생성
- text 목록 추출
- tokenizer와 연결한 `input_ids` 데이터셋 생성

예시:

```python
from dataset import TextDataset

dataset = TextDataset.from_jsonl("examples/dataset.jsonl")
split = dataset.split(
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)

for batch in dataset.batches(2):
    print(batch)
```

### 사전 토큰화 데이터셋 생성 CLI

JSONL:

```powershell
python prepare_dataset.py examples/dataset.jsonl --format jsonl --output build/pretokenized.jsonl --lowercase
```

CSV:

```powershell
python prepare_dataset.py examples/dataset.csv --format csv --output build/pretokenized-csv.jsonl
```

출력 JSONL에는 다음 정보가 포함됩니다.

- record id
- 원문 text
- label
- pre-tokenized tokens
- 각 token의 start/end offset

이 기능은 이후 BPE / WordPiece / SentencePiece 스타일 vocabulary 학습 전처리 단계에도 활용할 수 있습니다.


## 데이터·토큰 관리 봇

`bot.py`는 이 저장소의 데이터셋과 토크나이저 모델을 한 곳에서 관리하기 위한 관리 봇입니다.

지원 명령:

- `status`: 토큰 보관소 무결성과 등록 모델 수 확인
- `prepare-dataset`: TXT/JSONL/CSV 데이터셋 사전 토큰화 및 분할 통계 생성
- `train-register`: BPE / WordPiece / SentencePiece 스타일 모델 학습 후 토큰 보관소 등록
- `export`: 토큰 다운로드 보관소 ZIP 생성

### 로컬 실행

상태 확인:

```powershell
python bot.py status
```

데이터셋 준비:

```powershell
python bot.py prepare-dataset examples/dataset.jsonl --format jsonl --lowercase
```

BPE 학습 및 등록:

```powershell
python bot.py train-register examples/corpus.txt --kind bpe --name korean-bot --version 1.0.0 --vocab-size 128
```

다운로드 보관소 생성:

```powershell
python bot.py export
```

### GitHub Actions Bot

`.github/workflows/data-token-bot.yml`에서 **Run workflow**로 다음 명령을 선택 실행할 수 있습니다.

- status
- prepare-dataset
- train-register
- export

실행 결과는 `data-token-bot-output` Artifact로 저장됩니다.

### GitHub Issue 요청 양식

`.github/ISSUE_TEMPLATE/data-token-bot.yml`도 추가되어 있어 GitHub Issue에서 데이터/토큰 관리 요청을 구조화해 등록할 수 있습니다.

현재 봇은 저장소 내부 Python CLI와 GitHub Actions를 연결하는 방식이며, 별도 외부 서버나 API 토큰 없이 GitHub 기본 권한 범위에서 동작합니다.
