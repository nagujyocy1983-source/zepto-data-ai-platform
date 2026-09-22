# Module 1 – Data Pipeline

## 1. Overview

This module implements the **Data Pipeline** component of the Zepto Data & AI Platform project.

The pipeline follows these steps:

1. Scrape book data from Books to Scrape
2. Clean and transform the scraped data
3. Convert GBP prices to INR using the fixed conversion rate
4. Save the cleaned dataset as CSV
5. Load the data into a normalized SQLite database
6. Execute SQL analysis queries
7. Read SQL results into pandas
8. Reproduce the SQL JOIN using pandas `merge()`
9. Verify that SQL JOIN and pandas JOIN produce equivalent results

---

## 2. Source Website

The source website used for this project is:

https://books.toscrape.com/

No login, API key, or paid service is required.

The scraper collects books from multiple categories and automatically follows category pagination.

### Scraping requirement achieved

- Minimum required books: 60
- Books collected: 69
- Minimum required categories: 3
- Categories collected:
  - Historical Fiction
  - Mystery
  - Travel

**Final dataset size: 69 books across 3 categories**

---

## 3. Project Files

```text
data_pipeline/
│
├── scraper.py
├── pipeline.py
├── database.py
├── queries.sql
├── run_queries.py
├── cleaned_books.csv
├── zepto_catalog.db
└── README.md