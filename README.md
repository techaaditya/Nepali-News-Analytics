# Nepali News Analytics

Analyze Nepali news at scale: preprocessing, EDA, classification (Naive Bayes, SVM, Random Forest), sentiment analysis, extractive summarization, and NER — powered by R and Python.

## Table of Contents
- Project Overview
- Features
- Repository Structure
- Data & Schema
- Setup (Python, R, Git LFS)
- Quickstart
- Recreate Datasets (Python)
- Full Analysis (R Script & R Markdown)
- Models
- Outputs
- Troubleshooting
- Contributing
- References

---

## Project Overview
This project provides an end‑to‑end workflow to explore and model a large Nepali news corpus. It combines Python (for dataset preparation) and R (for analysis, modeling, and reporting) to deliver reproducible results and shareable artifacts (plots, summaries, models).

## Features
- Data curation with stratified sampling to maintain category balance
- Text cleaning pipeline tailored for Nepali news
- EDA on category/source distributions, article lengths, and top terms
- Supervised classification: Naive Bayes, SVM, Random Forest with comparisons
- Sentiment analysis using lexicons and modeling
- Named Entity Recognition (UDPipe-based) and extractive summarization (TextRank)
- Reproducible R Markdown report for end‑to‑end documentation

## Repository Structure
```
datasets/                      # CSV datasets (large files via Git LFS)
models/                        # Saved models & language resources
output/                        # Plots and result artifacts
R Basics/                      # In-depth documentation of the R code
dataset_parser.py              # Create balanced sample (50k)
train_test_code.py             # Create 40k/10k train/test split
R Project Main.R               # Main R pipeline (EDA → Models → NER/Summary)
R Project Main Final.Rmd       # Reproducible report (render to HTML)
```

## Data & Schema
Primary working datasets (tracked via Git LFS):
- `datasets/50k_news_dataset.csv`: 50,000-row balanced sample used as main corpus
- `datasets/nepali_news_train_40k.csv`: 40,000-row training split
- `datasets/nepali_news_test_10k.csv`: 10,000-row test split
- `datasets/nepali_sentiment_lexicon.csv`: Lexicon used for sentiment tasks
- `datasets/news_data.csv`, `datasets/news_data_20k_stratified.csv`: Additional sources

Expected columns (schema) used by the analysis:
- `heading` (str): Article title/headline
- `content` (str): Main body text
- `category` (str): Labeled news category
- `source` (str): Originating outlet/site

## Setup
The project uses Windows PowerShell commands below. Adapt paths as needed.

### 1) Clone and fetch large files
```powershell
git clone https://github.com/techaaditya/Nepali-News-Analytics.git
cd Nepali-News-Analytics
git lfs install
git lfs pull
```

### 2) Python environment (for dataset prep)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas scikit-learn
```

### 3) R packages (for analysis)
Install once in R/RStudio:
```r
install.packages(c(
  "readr","dplyr","stringr","tm","quanteda","quanteda.textmodels",
  "SparseM","doParallel","ranger","Matrix","pheatmap","reshape2",
  "textclean","glmnet","text2vec","tidytext","ggplot2","wordcloud",
  "RColorBrewer","caret","e1071","randomForest","udpipe","textrank"
))
```

## Quickstart
- Explore the finished HTML report: open `R-Project-Main-Final.html` (if present)
- Review key plots and tables in `output/`
- Skim explanations in `R Basics/Project Code Explanation.md`

If you want to reproduce results from scratch, follow the next two sections.

## Recreate Datasets (Python)
From the project root with the Python environment activated:

1) Generate a balanced 50k sample
```powershell
python dataset_parser.py
```

2) Split into train/test (40k/10k)
```powershell
python train_test_code.py
```

Notes:
- Both scripts expect an input CSV named `np20ng.csv` in the project root with at least the columns `heading`, `content`, `category`, `source`.
- Stratification uses `random_state=42` to ensure reproducibility.

## Full Analysis (R Script & R Markdown)
Run the R script (console or RStudio), from the project root so relative paths resolve:
```r
source("R Project Main.R")
```

Render the R Markdown report to HTML:
```r
rmarkdown::render("R Project Main Final.Rmd", output_format = "html_document")
```

Outputs will be written under `output/` and as an HTML report in the project root (if configured in the Rmd).

## Models
Saved models and resources (tracked via Git LFS):
- `models/naive_bayes_20k.rds`, `models/naive_bayes_model.rds`
- `models/random_forest_20k.rds`, `models/random_forest_model.rds`
- `models/svm_20k.rds`, `models/svm_model.rds`
- `models/training_dfm.rds` (document-feature matrix used during training)
- `models/hindi-hdtb-ud-2.5-191206.udpipe` (UDPipe model for NER)

To load a model in R:
```r
best_model <- readRDS("models/svm_20k.rds")
```

## Outputs
Key artifacts (examples):
- `output/accuracy_comparison.png`, `output/model_performance_comparison.png`
- `output/confusion_matrix_best_model.png`
- `output/ner_top_entities.csv`, `output/ner_complete_summary.rds`

## Troubleshooting
- Git LFS quota: If `git lfs pull` fails due to bandwidth/storage limits, clone without LFS and request increased quota or remove large files.
- R working directory: Run from the project root to avoid path issues. If `R Project Main.R` contains a `setwd(...)`, prefer removing or commenting it and using relative paths.
- Package installation: On Windows, you may need Rtools for packages requiring compilation.
- Memory: Building large DFMs can be memory-intensive; consider lowering vocabulary by increasing `min_docfreq` in `dfm_trim`.

## Contributing
Issues and PRs are welcome. For new features, please include a brief design note and, if applicable, updates to the README and Rmd.

## References
- See `R Basics/Project Code Explanation.md` for a detailed walkthrough of the analysis pipeline.
- The R Markdown `R Project Main Final.Rmd` contains executable steps and narrative.

---

Happy analyzing Nepali news data!
