import sqlite3
from pathlib import Path

import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite database path
DB_PATH = PROJECT_ROOT / "data_pipeline" / "zepto_catalog.db"

# SQL queries file path
SQL_FILE = PROJECT_ROOT / "data_pipeline" / "queries.sql"


def load_queries():
    """Read SQL queries from queries.sql."""

    with open(SQL_FILE, "r", encoding="utf-8") as file:
        sql_text = file.read()

    # Split the file into individual SQL statements
    queries = [
        query.strip()
    for query in sql_text.split(";")
    if query.strip()
    ]

    return queries


def run_sql_queries(connection, queries):
    """Execute all SQL queries and display their results."""

    print("\n" + "=" * 70)
    print("SQL QUERY RESULTS")
    print("=" * 70)

    results = []

    for index, query in enumerate(queries, start=1):

        print(f"\n{'-' * 70}")
        print(f"QUERY {index}")
        print("-" * 70)
        print(query)

        result = pd.read_sql(query, connection)

        print("\nOutput:")
        print(result.to_string(index=False))

        results.append(result)

    return results


def demonstrate_pandas_operations(connection):
    """Demonstrate pd.read_sql and reproduce JOIN using pd.merge."""

    print("\n" + "=" * 70)
    print("PANDAS VERIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Demonstration 1: Read books query result using pd.read_sql
    # ---------------------------------------------------------

    price_query = """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr
    FROM books
    WHERE price_gbp BETWEEN 20 AND 40
    ORDER BY price_gbp DESC
    LIMIT 10;
    """

    price_result = pd.read_sql(price_query, connection)

    print("\n1. pd.read_sql() result:")
    print(price_result.to_string(index=False))

    # ---------------------------------------------------------
    # Demonstration 2: Read category JOIN result using pd.read_sql
    # ---------------------------------------------------------

    sql_join_query = """
    SELECT
        b.book_id,
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books AS b
    JOIN categories AS c
        ON b.category_id = c.category_id;
    """

    sql_join_result = pd.read_sql(sql_join_query, connection)

    print("\n2. SQL JOIN using pd.read_sql():")
    print(sql_join_result.head().to_string(index=False))

    # ---------------------------------------------------------
    # Load both database tables into pandas
    # ---------------------------------------------------------

    books_df = pd.read_sql(
        "SELECT * FROM books;",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories;",
        connection
    )

    # ---------------------------------------------------------
    # Reproduce the JOIN using pandas merge
    # ---------------------------------------------------------

    pandas_join_result = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join_result = pandas_join_result[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    print("\n3. Pandas JOIN using pd.merge():")
    print(pandas_join_result.head().to_string(index=False))

    # ---------------------------------------------------------
    # Verify SQL JOIN and pandas JOIN produce the same data
    # ---------------------------------------------------------

    sql_sorted = sql_join_result.sort_values(
        by="book_id"
    ).reset_index(drop=True)

    pandas_sorted = pandas_join_result.sort_values(
        by="book_id"
    ).reset_index(drop=True)

    joins_match = sql_sorted.equals(pandas_sorted)

    print("\n4. JOIN verification:")
    print(f"SQL JOIN matches pandas JOIN: {joins_match}")

    if joins_match:
        print("JOIN verification PASSED.")
    else:
        print("JOIN verification FAILED.")


def main():
    """Run SQL queries and pandas verification."""

    print("=" * 70)
    print("ZEPTO DATA & AI PLATFORM")
    print("MODULE 1 - SQL QUERY EXECUTION")
    print("=" * 70)

    print(f"\nDatabase: {DB_PATH}")
    print(f"SQL file: {SQL_FILE}")

    # Connect to SQLite database
    connection = sqlite3.connect(DB_PATH)

    try:
        # Load SQL queries
        queries = load_queries()

        print(f"\nTotal SQL queries found: {len(queries)}")

        # Execute SQL queries
        run_sql_queries(connection, queries)

        # Demonstrate pandas operations
        demonstrate_pandas_operations(connection)

        print("\n" + "=" * 70)
        print("QUERY EXECUTION COMPLETED")
        print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()