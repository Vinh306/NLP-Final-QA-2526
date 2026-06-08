# QA Hệ thống Quy chế TDTU — NLP Cuối Kỳ

> Hệ thống hỏi đáp tự động về quy chế Trường Đại học Tôn Đức Thắng (TDTU)  
> So sánh 4 cấu hình: LLM gốc, RAG, Fine-tuned và Fine-tuned + RAG
---

## Giới thiệu

Dự án xây dựng và đánh giá hệ thống QA (Question Answering) tiếng Việt dựa trên các tài liệu quy chế nội bộ của TDTU. Mô hình nền là **Qwen2.5-3B-Instruct**, được triển khai theo 4 cấu hình khác nhau để so sánh hiệu quả của RAG và Fine-tuning.

---

## Dữ liệu

| File | Mô tả |
|---|---|
| `qa.json` | 506 cặp câu hỏi – câu trả lời, trích xuất từ tài liệu quy chế TDTU |
| `data/pdfs/` | Các file PDF quy chế gốc (upload thủ công lên Colab) |
| `data/combined_corpus.txt` | Toàn bộ văn bản sau khi trích xuất và làm sạch từ PDF |

Mỗi mục trong `qa.json` gồm các trường: `id`, `question`, `answer`, `context`, `source`, `instruction`.

**Phân chia dữ liệu:** Train 80% · Val 10% · Test 10% (cố định bằng `SEED=42`)

---

## Kiến trúc hệ thống

### 4 cấu hình thực nghiệm

| Config | Mô tả |
|---|---|
| **A** | LLM gốc — Qwen2.5-3B-Instruct, không RAG, không Fine-tune |
| **B** | LLM gốc + RAG — Truy xuất ngữ cảnh từ corpus trước khi sinh câu trả lời |
| **C** | Fine-tuned (QLoRA) — Mô hình đã được tinh chỉnh, không RAG |
| **D** | Fine-tuned + RAG — Kết hợp cả hai kỹ thuật |

### RAG Pipeline

- **Embedding model:** `sentence-transformers` (đa ngôn ngữ)
- **Vector store:** FAISS
- **Chunking:** `RecursiveCharacterTextSplitter` (LangChain)
- Corpus được index một lần, cache lại cho các lần chạy sau

### Fine-tuning (QLoRA)

- **Phương pháp:** QLoRA với lượng tử hóa 4-bit (BitsAndBytes NF4)
- **LoRA config:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- **Số epoch:** 10, đánh giá mỗi epoch
- **Trainer:** `SFTTrainer` (TRL)
- Adapter được lưu tại `./checkpoints/lora_adapter`

---

## Cài đặt

Chạy trên **Google Colab** (yêu cầu GPU, khuyến nghị A100/T4).

```bash
# Tất cả thư viện được cài trong cell đầu notebook
pip install pymupdf pymupdf4llm underthesea google-generativeai
pip install transformers accelerate bitsandbytes sentencepiece
pip install rouge-score bert-score sacrebleu
pip install faiss-cpu sentence-transformers langchain-text-splitters
pip install peft trl datasets
pip install gradio
```

---

## Cách chạy

1. Mở file `NLP_CK_52300164_52300169.ipynb` trên Google Colab
2. Chạy lần lượt từ **Cell 1** đến hết:

| Bước | Cell | Mô tả |
|---|---|---|
| 1 | Cài thư viện | Cài đặt toàn bộ dependencies |
| 2 | Import & Cấu hình | Khai báo hyperparameter, seed, thư mục |
| 3 | Trích xuất PDF | Upload PDF quy chế, làm sạch và lưu văn bản |
| 4 | Chia dữ liệu | Đọc `qa.json`, chia train/val/test |
| 5 | Hàm tiện ích | Định nghĩa hàm prompt, generate, evaluate |
| 6 | RAG Pipeline | Xây dựng FAISS index từ corpus |
| 7 | Load model | Load Qwen2.5-3B-Instruct + tokenizer |
| 8–9 | Config A, B | Chạy inference LLM gốc (có/không RAG) |
| 10 | Fine-tune | Huấn luyện QLoRA adapter |
| 11–12 | Config C, D | Inference với model đã fine-tune |
| 13 | So sánh | In bảng tổng hợp 4 config, lưu `summary.csv` |
| 14 | Human Eval | Xuất file CSV để đánh giá tay 50 câu |
| 15 | Web Demo | Khởi động Gradio demo so sánh 4 config |

---

## Đánh giá

### Tự động (Automatic Metrics)

| Metric | Mô tả |
|---|---|
| BLEU | Đo độ chồng lặp n-gram giữa câu dự đoán và tham chiếu |
| ROUGE-1/2/L | Recall-based overlap ở mức unigram, bigram, LCS |
| BERTScore-F1 | Đo độ tương đồng ngữ nghĩa dựa trên embedding |
| Recall@K | Tỷ lệ ngữ cảnh liên quan được RAG truy xuất (Config B, D) |
| Latency | Thời gian sinh câu trả lời trung bình (giây/câu) |

### Human Evaluation

50 câu được đánh giá bởi nhiều annotators theo 3 tiêu chí (thang 1–5):

- **Accuracy** — Độ chính xác của thông tin
- **Completeness** — Mức độ đầy đủ của câu trả lời
- **Fluency** — Độ trôi chảy, tự nhiên của ngôn ngữ

Độ nhất quán giữa annotators được đo bằng **Cohen's Kappa**.

### Kết quả

Sau khi chạy đủ 4 config, kết quả được lưu tại thư mục `./results/`:

```
results/
├── summary.csv                  # Bảng so sánh tự động 4 config
├── human_eval_template.csv      # Template đánh giá tay
├── human_eval_summary.csv       # Tổng hợp điểm human eval
├── human_eval_kappa_detail.csv  # Cohen's Kappa chi tiết
├── human_eval_mean_scores.csv   # Điểm trung bình từng câu
├── human_eval_bars.png          # Biểu đồ cột theo tiêu chí
├── human_eval_radar.png         # Radar chart 4 config
└── human_eval_kappa_heatmap.png # Heatmap Cohen's Kappa
```

---

## Cấu hình mô hình

```python
MODEL_NAME      = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.1
TOP_P           = 0.9
DO_SAMPLE       = False
MAX_SEQ_LENGTH  = 512
SEED            = 42
```

---

## Cấu trúc thư mục

```
.
├── NLP_CK_52300164_52300169.ipynb   # Notebook chính
├── qa.json                          # Tập dữ liệu QA (506 mẫu)
├── data/
│   ├── pdfs/                        # File PDF quy chế gốc
│   ├── texts/                       # Văn bản đã trích xuất
│   ├── combined_corpus.txt          # Corpus tổng hợp
│   ├── train.json
│   ├── val.json
│   └── test.json
├── checkpoints/
│   └── lora_adapter/                # LoRA adapter sau fine-tune
├── results/                         # Kết quả đánh giá
└── web_demo.py                      # Script Gradio demo
```

---

## Công nghệ sử dụng

- **LLM:** Qwen2.5-3B-Instruct (Alibaba)
- **Fine-tuning:** QLoRA via PEFT + TRL (SFTTrainer)
- **RAG:** FAISS + Sentence-Transformers + LangChain Text Splitter
- **PDF parsing:** PyMuPDF / pymupdf4llm
- **Metrics:** rouge-score, bert-score, sacrebleu
- **Demo:** Gradio
- **Môi trường:** Google Colab (GPU)
