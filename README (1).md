# 📄 AI Resume Skill Gap Analyzer

> An NLP-powered web application that compares a candidate's resume against a job description — quantifying match percentage, identifying skill gaps, and predicting job category using a machine learning model trained on real-world resume data.

---

## 🚀 Features

| Feature | Description |
|---|---|
| PDF Resume Upload | Extracts raw text from uploaded PDF resumes |
| Job Description Input | Accepts free-text job descriptions |
| Text Preprocessing | Tokenization, stopword removal, lemmatization via spaCy |
| TF-IDF Vectorization | Converts text to numerical feature vectors |
| Cosine Similarity | Calculates match score between resume and JD |
| Skill Extraction | Keyword-based matching against 180+ skills |
| Skill Gap Report | Shows matched, missing, and bonus skills |
| ML Classification | Predicts resume job category using trained Logistic Regression model |

---

## 🧠 Machine Learning Pipeline

```
Raw Text
   │
   ▼
Text Cleaning        ← lowercase, remove special characters
   │
   ▼
Lemmatization        ← spaCy (en_core_web_sm)
   │
   ▼
Stopword Removal     ← NLTK English stopwords
   │
   ▼
TF-IDF Vectorization ← scikit-learn (unigrams + bigrams, max 5000 features)
   │
   ├──► Cosine Similarity  →  Match % score
   │
   └──► Logistic Regression  →  Predicted job category
```

### Concepts Used

- Natural Language Processing (NLP)
- Text Cleaning & Normalization
- Lemmatization & Stopword Removal
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine Similarity
- Logistic Regression (Multi-class Classification)
- Label Encoding
- Train/Test Split & Model Evaluation

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| NLP | spaCy, NLTK |
| ML | scikit-learn |
| PDF Parsing | PyPDF2 |
| Language | Python 3.9+ |

---

## 📂 Project Structure

```
resume-skill-gap-analyzer/
│
├── app/
│   ├── app.py               # Streamlit UI & main application logic
│   ├── preprocessing.py     # Text cleaning & lemmatization pipeline
│   ├── similarity.py        # TF-IDF vectorization & cosine similarity
│   └── skill_extraction.py  # Keyword-based skill matching (180+ skills)
│
├── train_model.py           # ML training script (Kaggle dataset)
├── models/                  # Saved model artifacts (generated after training)
│   ├── tfidf_vectorizer.pkl
│   ├── resume_classifier.pkl
│   └── label_encoder.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd resume-skill-gap-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the app
```bash
streamlit run app/app.py
```

---

## 🤖 Training the ML Model (Kaggle Dataset)

The classifier is trained on the **Resume Dataset** from Kaggle, which contains 962 real resumes labeled across 25 job categories.

### Step 1 — Download the dataset

Go to: [https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

Download `UpdatedResumeDataSet.csv` and place it in the project root folder.

### Step 2 — Run the training script

```bash
python train_model.py --data UpdatedResumeDataSet.csv
```

**Optional flags:**
```bash
python train_model.py \
  --data UpdatedResumeDataSet.csv \
  --test_size 0.2 \        # 80/20 train-test split
  --max_features 5000      # TF-IDF vocabulary size
```

### Step 3 — What gets saved

After training, three files are saved in `./models/`:

```
models/
├── tfidf_vectorizer.pkl    # Fitted TF-IDF transformer
├── resume_classifier.pkl   # Trained Logistic Regression model
└── label_encoder.pkl       # Category label encoder
```

### Expected Training Output

```
📂 Loading dataset: UpdatedResumeDataSet.csv
✅ Loaded 962 records across 25 categories.

🔄 Preprocessing resume text...
📊 Train samples: 769 | Test samples: 193

🔢 Fitting TF-IDF vectorizer (max_features=5000)...
🤖 Training Logistic Regression classifier...

✅ Test Accuracy: ~96%

💾 Saved model artifacts to ./models/
```

### Dataset Categories (25 total)

`Data Science` · `HR` · `Advocate` · `Arts` · `Web Designing` · `Mechanical Engineer` · `Sales` · `Health and Fitness` · `Civil Engineer` · `Java Developer` · `Business Analyst` · `SAP Developer` · `Automation Testing` · `Electrical Engineering` · `Operations Manager` · `Python Developer` · `DevOps Engineer` · `Network Security Engineer` · `PMO` · `Database` · `Hadoop` · `ETL Developer` · `DotNet Developer` · `Blockchain` · `Testing`

---

## 🔍 How It Works

1. **PDF Parsing** — Resume is converted from PDF to plain text using PyPDF2.
2. **Preprocessing** — Text is cleaned, lowercased, lemmatized, and stripped of stopwords.
3. **Similarity Scoring** — TF-IDF vectors of the resume and job description are compared using cosine similarity to produce a match percentage.
4. **Skill Extraction** — 180+ tech and soft skills are matched using regex whole-word search across both texts.
5. **Gap Analysis** — Skills are categorized as Matched, Missing, or Extra (bonus).
6. **Category Prediction** — The trained Logistic Regression model predicts what job category the resume is best suited for.

---

## 📊 Example Output

```
Match Score:      72%
JD Skills Found:  14
Matched Skills:   9   ✅  python, sql, machine learning, pandas ...
Missing Skills:   5   ❌  tableau, power bi, spark, airflow, dbt
Extra Skills:     6   💡  flask, docker, git ...

Predicted Category: Data Science
```

---

## ⚠️ Limitations

- Skill extraction is keyword-based — it does not understand context or synonyms.
- Cosine similarity measures vocabulary overlap, not true semantic meaning.
- Works best with text-based PDFs (not scanned images or image-only PDFs).
- Model accuracy depends on resume formatting and text extraction quality.

---

## 🔮 Future Improvements

- [ ] BERT / Sentence-Transformers for semantic similarity
- [ ] Named Entity Recognition (NER) for smarter skill extraction
- [ ] Resume scoring dashboard with radar chart
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- [ ] OCR support for scanned PDF resumes
- [ ] Auto-generate learning roadmap for missing skills

---

## 👩‍💻 Author

**Riddhi Garg** — 25BHI10004
