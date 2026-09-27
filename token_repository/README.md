# Token Download Repository

이 디렉터리는 학습된 토크나이저 모델과 vocabulary 파일을 보관하고 다운로드 패키지로 만들기 위한 전용 저장소입니다.

## 구조

- `models/`: 모델 유형/이름/버전별 실제 토큰 모델 파일
- `manifest.json`: 모델 메타데이터와 SHA256 체크섬
- `downloads/`: 다운로드용 ZIP 생성 위치
- `SHA256SUMS.txt`: 다운로드 번들 무결성 검증용 체크섬

## 지원 모델

- SimpleTokenizer
- BPE
- WordPiece
- SentencePiece 스타일

## 사용

모델 등록:

```bash
python token_repository.py add bpe.json --name korean-demo --type bpe --version 1.0.0
```

목록 조회:

```bash
python token_repository.py list
```

무결성 검증:

```bash
python token_repository.py verify
```

다운로드 ZIP 생성:

```bash
python token_repository.py export
```
