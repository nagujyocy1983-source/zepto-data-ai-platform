import sqlite3
import pandas as pd
from pathlib import Path


# File paths
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "cleaned_books.csv"
DB_FILE = BASE_DIR / "zepto_catalog.db"


def create_database():
    """Create normalized SQLite database and load cleaned book data."""

    # Check that the cleaned CSV exists
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {CSV_FILE}\n"
            "Run pipeline.py first."
        )

    # Read cleaned data
    df = pd.read_csv(CSV_FILE)

    print(f"Cleaned records found: {len(df)}")

    # Connect to SQLite database
    connection = sqlite3.connect(DB_FILE)

    try:
        # Enable foreign-key enforcement
        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.cursor()

        # ---------------------------------------------------------
        # 1. Remove old tables so the database is recreated cleanly
        # ---------------------------------------------------------
        cursor.execute("DROP TABLE IF EXISTS books")
        cursor.execute("DROP TABLE IF EXISTS categories")

        # ---------------------------------------------------------
        # 2. Create categories table
        # ---------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
            """
        )

        # ---------------------------------------------------------
        # 3. Create books table
        # ---------------------------------------------------------
        cursor.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL,
                price_inr REAL,
                rating INTEGER,
                in_stock INTEGER,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            )
            """
        )

        # ---------------------------------------------------------
        # 4. Insert unique categories
        # ---------------------------------------------------------
        categories = sorted(df["category"].dropna().unique())

        cursor.executemany(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            [(category,) for category in categories]
        )

        # ---------------------------------------------------------
        # 5. Get category IDs
        # ---------------------------------------------------------
        category_rows = cursor.execute(
            """
            SELECT category_id, category_name
            FROM categories
            """
        ).fetchall()

        category_map = {
            category_name: category_id
            for category_id, category_name in category_rows
        }

        # ---------------------------------------------------------
        # 6. Prepare book records
        # ---------------------------------------------------------
        book_records = []

        for _, row in df.iterrows():

            category_id = category_map[row["category"]]

            book_records.append(
                (
                    row["title"],
                    float(row["price_gbp"]),
                    float(row["price_inr"]),
                    int(row["star_rating"]),
                    int(bool(row["in_stock"])),
                    category_id,
                )
            )

        # ---------------------------------------------------------
        # 7. Insert books
        # ---------------------------------------------------------
        cursor.executemany(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            book_records
        )

        # Save changes
        connection.commit()

        print("\nSQLite database created successfully.")
        print(f"Database: {DB_FILE}")
        print("Tables created: categories, books")
        print(f"Categories loaded: {len(categories)}")
        print(f"Books loaded: {len(book_records)}")

    finally:
        connection.close()


def verify_database():
    """Verify tables, record counts, relationships, and sample JOIN."""

    connection = sqlite3.connect(DB_FILE)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        print("\n" + "=" * 60)
        print("DATABASE VERIFICATION")
        print("=" * 60)

        # ---------------------------------------------------------
        # 1. Check tables
        # ---------------------------------------------------------
        tables_query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """

        tables = pd.read_sql_query(tables_query, connection)

        print("\nTables in database:")
        print(tables)

        # ---------------------------------------------------------
        # 2. Count categories
        # ---------------------------------------------------------
        category_count_query = """
        SELECT COUNT(*) AS total_categories
        FROM categories;
        """

        category_count = pd.read_sql_query(
            category_count_query,
            connection
        )

        print("\nTotal categories:")
        print(category_count)

        # ---------------------------------------------------------
        # 3. Count books
        # ---------------------------------------------------------
        book_count_query = """
        SELECT COUNT(*) AS total_books
        FROM books;
        """

        book_count = pd.read_sql_query(
            book_count_query,
            connection
        )

        print("\nTotal books:")
        print(book_count)

        # ---------------------------------------------------------
        # 4. Check books by category
        # ---------------------------------------------------------
        category_summary_query = """
        SELECT
            c.category_name,
            COUNT(b.book_id) AS book_count
        FROM categories c
        JOIN books b
            ON c.category_id = b.category_id
        GROUP BY c.category_name
        ORDER BY c.category_name;
        """

        category_summary = pd.read_sql_query(
            category_summary_query,
            connection
        )

        print("\nBooks by category:")
        print(category_summary)

        # ---------------------------------------------------------
        # 5. Test JOIN between books and categories
        # ---------------------------------------------------------
        join_query = """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
        LIMIT 5;
        """

        join_result = pd.read_sql_query(
            join_query,
            connection
        )

        print("\nSample JOIN result:")
        print(join_result)

        # ---------------------------------------------------------
        # 6. Verify foreign-key violations
        # ---------------------------------------------------------
        fk_check = pd.read_sql_query(
            "PRAGMA foreign_key_check;",
            connection
        )

        print("\nForeign-key check:")
        if fk_check.empty:
            print("No foreign-key violations found.")
        else:
            print(fk_check)

        print("\n" + "=" * 60)
        print("DATABASE VERIFICATION COMPLETED")
        print("=" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()
    verify_database()