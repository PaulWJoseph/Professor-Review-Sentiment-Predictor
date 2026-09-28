# UMD Professor Review Sentiment Classifier

A PyTorch & Hugging Face NLP pipeline that extracts professor reviews from the [PlanetTerp API](https://planetterp.com/) and fine-tunes a pre-trained `bert-base-uncased` model to predict review ratings (1 to 5 stars) based on student feedback text.

## Overview

Student reviews often contain subtle sentiment nuances—ranging from workload complaints to praise for lecture quality. This project automates the extraction of review data for University of Maryland (UMD) professors and builds a multi-class sequence classification model using BERT to accurately predict student star ratings from raw review text.

Key highlights:
* **Dynamic API Ingestion:** Fetches real-world review text and ratings via the `planetterp` wrapper.
* **Text Preprocessing:** Tokenizes and dynamically pads text inputs using Hugging Face `transformers` and `datasets`.
* **BERT Fine-Tuning:** Fine-tunes `bert-base-uncased` for 5-class rating classification ($0 \to 4$ zero-indexed).
* **Tolerant Metric Evaluation:** Measures both exact-match accuracy and "within $\pm 1$ star" accuracy to account for subjective rating variance.

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch
* **NLP & Transformers:** Hugging Face (`transformers`, `datasets`)
* **API Ingestion:** `planetterp`
* **Optimization:** `torch.optim.AdamW`, Linear LR Scheduler with PyTorch/Hugging Face
* **Utilities:** `tqdm`, `numpy`, `matplotlib`

## Pipeline Architecture

```text
[ PlanetTerp API ] 
       │
       ▼ (Fetch Reviews & Ratings)
[ Data Preprocessing ] ──► Map 1-5 Stars to 0-4 Classes & Tokenize
       │
       ▼
[ Hugging Face Datasets ] ──► Dynamic Batching with DataCollatorWithPadding
       │
       ▼
[ BERT Fine-Tuning ] ──► AutoModelForSequenceClassification + AdamW (lr=3e-5)
       │
       ▼
[ Evaluation Engine ] ──► Calculate Exact Match & +/- 1 Star Accuracy
```

## Performance & Results

> *Note: Rerun the Python project on device. Or do it in Colab to get results since Colab uses external processing compared to a laptop.*

| Metric | Accuracy Score | Description |
| :--- | :--- | :--- |
| **Exact Match Accuracy** | `68.6%` | Predictions matching the exact star rating ($1 \to 5$) |
| **Within $\pm 1$ Star Accuracy** | `92.2%` | Predictions within $1$ star of the actual rating |

### Insights & Misclassifications
Analysis of misclassified samples revealed that reviews with mixed sentiment (e.g., praise for teaching quality paired with harsh grading complaints) presented the highest variance, highlighting the inherent subjectivity of 5-star rating systems.

## Project Structure

```text
Professor-Review-Sentiment-Analysis/
├── ProfessorReviewSentimentAnalysis.py   # Main end-to-end training & evaluation pipeline
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Files/directories excluded from Git
└── README.md                             # Project documentation
```

## Getting Started

### Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU (Optional, but recommended for faster training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PaulWJoseph/Professor-Review-Sentiment-Analysis.git
   cd Professor-Review-Sentiment-Analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script:**
   ```bash
   python ProfessorReviewSentimentAnalysis.py
   ```