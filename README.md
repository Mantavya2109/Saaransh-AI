# Saaransh-AI (Text Summarizer)

A lightweight, end-to-end Generative AI web application for abstractive dialogue and text summarization powered by a fine-tuned **T5 Transformer**, **FastAPI**, and an interactive web interface.

---

## 🧠 GenAI & Model Architecture

Saaransh-AI leverages **Generative AI** using an encoder-decoder sequence-to-sequence transformer model fine-tuned for dialogue abstraction:

- **Base Architecture**: **T5 (Text-to-Text Transfer Transformer)**
  - **Encoder**: Processes bidirectional context of input conversations into dense vector representations.
  - **Decoder**: Autoregressively generates concise abstractive summaries conditioned on encoder hidden states and cross-attention mechanisms.
- **Fine-Tuning Dataset**: **SAMSum Dataset** (dialogues paired with human-written summaries).
- **Inference & Generation**:
  - **Tokenizer**: `T5Tokenizer` with SentencePiece subword tokenization (input max length: `512`, target max length: `150`).
  - **Decoding Strategy**: **Beam Search** (`num_beams=4`, `early_stopping=True`) to explore high-probability token sequences for coherent and grammatically sound summaries.
  - **Hugging Face Hub Model**: Loaded directly from [`Monti2109/text-summarizer-model`](https://huggingface.co/Monti2109/text-summarizer-model).

---

## 📁 Repository Overview

| Component | File | Description |
| :--- | :--- | :--- |
| **Model Training** | [`text_summarizer.ipynb`](./text_summarizer.ipynb) | End-to-end pipeline: data preprocessing, subword tokenization, fine-tuning `t5-small` with Hugging Face `Trainer`, and checkpoint export. |
| **Backend API** | [`app.py`](./app.py) | **FastAPI** application that loads the fine-tuned model onto GPU/CPU, handles text cleaning, runs generation inference, and exposes the `/summarize/` POST endpoint. |
| **Frontend UI** | [`index.html`](./index.html) | Responsive UI with real-time asynchronous JavaScript (`fetch`) to send dialogues to the backend and display generated summaries. |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Mantavya2109/Saaransh-AI.git
cd Saaransh-AI
python -m venv venv
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
uvicorn app:app --reload
```

Open your browser and visit: **`http://127.0.0.1:8000`**

---

## 🔌 API Reference

### POST `/summarize/`
- **Request Body**:
  ```json
  {
    "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure! I'll come over."
  }
  ```
- **Response**:
  ```json
  {
    "summary": "Amanda baked cookies and Jerry will come over to get some."
  }
  ```
