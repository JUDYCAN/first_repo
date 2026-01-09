# AI 기반 문서 아카이브 시스템 (보고서용 README)

## 1. 프로젝트 개요
본 프로젝트는 이미지 형태의 문서를 업로드하면 AI 모델을 활용하여 문서 유형을 판별하고,
OCR, 이미지 전처리, 형태소 기반 키워드 추출, 요약, 메타데이터 분석을 수행한 뒤
데이터베이스에 저장·검색할 수 있는 AI 기반 문서 아카이브 시스템을 구현하는 것을 목표로 한다.

본 시스템은 단순한 OCR 도구가 아니라, 환경 제약(운영체제, 라이브러리 의존성 등)을 고려하여
안정적으로 동작하도록 설계된 통합 문서 분석 파이프라인이다.

---

## 2. 시스템 전체 흐름

1. 사용자 이미지 업로드  
2. 문서 이미지 분류 (DiT 기반)  
3. 이미지 전처리 수행 및 비교 가시화  
4. OCR 기반 텍스트 추출  
5. 형태소 기반 키워드 추출  
6. 텍스트 요약 생성  
7. 사진의 경우 EXIF 메타데이터 분석  
8. 결과 DB 저장 및 조회

---

## 3. 주요 기능

### 3.1 문서 / 사진 자동 판별
- DiT(Document Image Transformer) 모델을 사용하여 이미지 기반 문서 유형 분류
- OCR 결과가 거의 없는 경우 사진으로 판별

### 3.2 이미지 전처리 및 가시화
- OpenCV를 활용한 이미지 전처리
  - Grayscale 변환
  - Gaussian Blur
  - Adaptive Threshold
- 원본 이미지와 전처리 이미지 비교 시각화 제공

### 3.3 OCR (Optical Character Recognition)
- PaddleOCR 사용
- Windows + CPU 환경의 안정성을 고려하여 Text Detection(det) 비활성화
- 문자 인식(rec)만 사용하여 시스템 중단 방지
- OCR 실패 시에도 시스템이 중단되지 않도록 예외 처리 적용

### 3.4 형태소 기반 키워드 추출
- Java 의존성이 있는 Konlpy 대신 순수 Python 기반 soynlp 사용
- 단어 응집도 기반 후보 추출 후 TF-IDF 중요도 계산
- 상위 키워드를 문서 핵심 정보로 활용

### 3.5 텍스트 요약
- KoBART 모델 기반 요약
- OCR 특성을 고려한 텍스트 정제 후 요약 수행
- 텍스트 길이가 짧거나 의미가 부족한 경우 요약 생략

### 3.6 사진 메타데이터(EXIF) 분석
- 사진으로 판별된 경우 촬영 정보(EXIF) 추출
- JSON 형태로 메타데이터 시각화

### 3.7 데이터베이스 저장 및 조회
- SQLite + SQLModel 사용
- 이미지, 텍스트, 요약, 키워드, 메타데이터, 임베딩 정보 저장

---

## 4. 기술 스택

### Backend / AI
- Python 3.11
- PaddleOCR
- HuggingFace Transformers (DiT, KoBART)
- soynlp
- SentenceTransformer

### Frontend
- Streamlit

### Database
- SQLite
- SQLModel

### Image Processing
- OpenCV
- Pillow

---

## 5. 환경 제약 및 설계 판단

### 5.1 Java 미사용 설계
- Java(JDK) 의존성 문제로 Konlpy 사용 제외
- 순수 Python 기반 형태소/키워드 추출 방식 채택

### 5.2 OCR 안정성 확보
- Windows 환경에서 PaddleOCR text detector의 불안정성 확인
- detector(det) 비활성화 및 예외 처리로 시스템 안정성 확보

본 설계는 OCR 실패 여부와 관계없이 전체 시스템이 정상 동작하도록 하기 위한
현실적인 판단에 기반한다.

---

## 6. 실행 방법

```bash
pip install -r requirements.txt
streamlit run app1.py
```

