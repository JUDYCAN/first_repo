import os
# [에러 해결] 이 설정이 모든 import보다 위에 있어야 합니다.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
os.environ['FLAGS_use_mkldnn'] = '0' # MKLDNN 강제 비활성화

import streamlit as st
import torch
import io
import json
import re
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from PIL import Image, ExifTags

from sqlmodel import Field, Session, SQLModel, create_engine, select
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from soynlp.word import WordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import (
    AutoModelForImageClassification, AutoProcessor,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel,
    LayoutLMv3Processor, LayoutLMv3ForTokenClassification
)

# =========================
# DB MODEL
# =========================
class Document(SQLModel, table=True):
    __table_args__ = {"extend_existing": True} 
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    doc_type: str
    content: str
    summary: str
    keywords: str
    structured_data: str
    upload_date: datetime = Field(default_factory=datetime.now)
    image_data: bytes
    embedding: Optional[str] = None

engine = create_engine("sqlite:///archive.db")
SQLModel.metadata.create_all(engine)

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    dit_processor = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    dit_model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

    # [에러 해결] use_gpu=False와 enable_mkldnn=False를 명시적으로 설정
    ocr = PaddleOCR(
        lang='korean',
        use_gpu=False,
        cpu_threads=1,
        enable_mkldnn=False,
        show_log=False,
        det=True,        # 🔥 탐지 모델 완전 비활성화
        rec=True,
        cls=False
    )


    donut_processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    sum_tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")

    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    return (
        dit_processor, dit_model, ocr, donut_processor, donut_model,
        layout_processor, layout_model, sum_tokenizer, sum_model, embed_model
    )

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(binary)

# =========================
# DOCUMENT / PHOTO CLASSIFICATION
# =========================
def classify_document(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    logits = model(**inputs).logits
    label = model.config.id2label[logits.argmax(-1).item()]
    return label

# =========================
# OCR
# =========================
def extract_text(image, ocr):
    img = np.array(image)
    if img.dtype != np.uint8: img = img.astype(np.uint8)
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    result = ocr.ocr(img, cls=False)
    text = ""
    if result and result[0]:
        for line in result[0]:
            text += line[1][0] + " "
    return text.strip()

# =========================
# MORPHEME ANALYSIS
# =========================
def extract_keywords_morpheme(text, top_k=15):
    if not text or len(text.strip()) < 10: return [], []
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences: return [], []

    word_extractor = WordExtractor(min_frequency=2, min_cohesion_forward=0.05)
    word_extractor.train(sentences)
    words = word_extractor.extract()

    candidates = [w for w, score in words.items() if len(w) >= 2 and score.cohesion_forward > 0.1]
    if not candidates: return [], []

    vectorizer = TfidfVectorizer(vocabulary=candidates)
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray()[0]
    keywords = sorted(zip(vectorizer.get_feature_names_out(), scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [k for k, _ in keywords], candidates

# =========================
# SUMMARY (요약 품질 대폭 강화)
# =========================
def summarize_text(text, tokenizer, model):
    if not text or len(text.strip()) < 40:
        return text if text else "요약할 내용이 없습니다."

    # [요약 개선 1] OCR 노이즈 정제 (특수문자 및 파편 제거)
    # 한글, 영문, 숫자, 마침표, 쉼표만 남기고 모두 제거
    cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s.,]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if len(cleaned) < 30: return "인식된 텍스트가 너무 적어 요약이 불가능합니다."

    inputs = tokenizer(cleaned[:1024], return_tensors="pt", truncation=True)

    # [요약 개선 2] 생성 알고리즘 최적화
    output = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        num_beams=5,
        repetition_penalty=1.2,    # 반복 억제 대폭 강화 (중복 문장 방지)
        no_repeat_ngram_size=3,    # 3단어 이상 겹치는 문구 방지
        length_penalty=1.2,        # 문장이 너무 짧아지지 않도록 유도
        bad_words_ids=[[tokenizer.convert_tokens_to_ids(".")] * 3], # 마침표 도배 방지
        early_stopping=True
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 만약 요약이 비정상적으로 짧거나 이상하면 정제된 텍스트 앞부분 반환
    if len(summary) < 5: return cleaned[:150]
    return summary

# =========================
# EMBEDDING / EXIF
# =========================
def create_embedding(text, model):
    return model.encode(text).tolist()

def extract_exif(image):
    exif_data = {}
    try:
        raw = image._getexif()
        if raw:
            for tag, value in raw.items():
                name = ExifTags.TAGS.get(tag, tag)
                exif_data[name] = str(value)
    except: pass
    return exif_data

# =========================
# MAIN PROCESS
# =========================
def process_document(uploaded_file, models):
    (dit_p, dit_m, ocr, donut_p, donut_m, layout_p, layout_m, sum_t, sum_m, embed_m) = models

    image = Image.open(uploaded_file).convert("RGB")
    pre_img = preprocess_image_for_ocr(image)

    doc_type = classify_document(image, dit_p, dit_m)
    text = extract_text(image, ocr)

    is_photo = len(text.strip()) < 15
    summary = summarize_text(text, sum_t, sum_m)
    keywords, morphemes = extract_keywords_morpheme(text)
    embedding = create_embedding(text + summary, embed_m)

    structured = {"EXIF": extract_exif(image)} if is_photo else {}

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    return {
        "doc_type": "사진" if is_photo else doc_type,
        "text": text,
        "summary": summary,
        "keywords": keywords,
        "morphemes": morphemes,
        "structured": structured,
        "image": image,
        "pre_image": pre_img,
        "img_bytes": img_bytes.getvalue(),
        "embedding": embedding
    }

# =========================
# UI (결과 화면 동일 유지)
# =========================
st.title("📁 AI 아카이브 시스템")

models = load_models()
tab1, tab2 = st.tabs(["업로드 & 분석", "저장 문서"])

# TAB 1
with tab1:
    uploaded = st.file_uploader("이미지 업로드", type=["png","jpg","jpeg"])

    if uploaded:
        result = process_document(uploaded, models)

        st.subheader("① 이미지 전처리 비교")
        c1, c2 = st.columns(2)
        c1.image(result["image"], caption="원본")
        c2.image(result["pre_image"], caption="전처리 후")

        st.subheader("② 문서 / 사진 판별")
        st.write("📌 판별 결과:", result["doc_type"])

        st.subheader("③ OCR 결과")
        st.text_area("텍스트", result["text"], height=150)

        st.subheader("④ 형태소 분석 가시화")
        st.write("🔑 키워드:", result["keywords"])
        st.write("📚 형태소:", result["morphemes"])

        st.subheader("⑤ 요약")
        st.write(result["summary"])

        if result["structured"]:
            st.subheader("⑥ 사진 메타데이터")
            st.json(result["structured"])

        if st.button("DB 저장"):
            with Session(engine) as s:
                s.add(Document(
                    filename=uploaded.name,
                    doc_type=result["doc_type"],
                    content=result["text"],
                    summary=result["summary"],
                    keywords=",".join(result["keywords"]),
                    structured_data=json.dumps(result["structured"], ensure_ascii=False),
                    image_data=result["img_bytes"],
                    embedding=json.dumps(result["embedding"])
                ))
                s.commit()
            st.success("저장 완료")

# TAB 2
with tab2:
    with Session(engine) as s:
        docs = s.exec(select(Document)).all()
        for d in docs:
            with st.expander(d.filename):
                st.image(Image.open(io.BytesIO(d.image_data)))
                st.write("유형:", d.doc_type)
                st.write("요약:", d.summary)
                st.write("키워드:", d.keywords)