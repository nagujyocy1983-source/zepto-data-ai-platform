# Zepto Data & AI Platform

An end-to-end Data, Analytics, and GenAI Support Assistant platform developed as part of the AI/ML Engineer Capstone Project.

---

## Project Overview

This project demonstrates an integrated AI/ML engineering workflow through three modules:

1. **Data Pipeline**
   - Web scraping
   - Data cleaning
   - Transformation
   - SQLite database creation
   - SQL querying
   - Pandas validation

2. **Analytics Pipeline**
   - Titanic dataset profiling
   - Missing-value analysis
   - Outlier analysis
   - Univariate and bivariate analysis
   - Multivariate visualizations
   - Correlation analysis
   - Standardization
   - Machine learning classification
   - Class-imbalance handling
   - SMOTE
   - Random Forest hyperparameter tuning
   - Fare regression
   - Model comparison
   - Model persistence using Joblib

3. **Support Assistant**
   - Eight Zepto policy documents
   - Local sentence-transformer embeddings
   - ChromaDB vector retrieval
   - LangGraph orchestration
   - Deterministic offline mock logic
   - Pydantic structured output
   - FastAPI `/ask` endpoint
   - Docker support

---

## Repository Structure

```text
zepto-data-ai-platform/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_pipeline/
│   ├── README.md
│   ├── scraper.py
│   ├── pipeline.py
│   ├── database.py
│   ├── queries.sql
│   ├── run_queries.py
│   ├── cleaned_books.csv
│   └── zepto_catalog.db
│
├── analytics/
│   ├── README.md
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── 03_multivariate.py
│   ├── titanic.csv
│   ├── class_imbalance_comparison.csv
│   ├── final_model_comparison.csv
│   ├── best_classification_pipeline.joblib
│   └── visualization outputs
│
└── support_assistant/
    ├── README.md
    ├── build_docs.py
    ├── build_index.py
    ├── main.py
    ├── Dockerfile
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    └── chroma_db/