import sqlite3
import pandas as pd


CSV_FILE = "data_pipeline/cleaned_books.csv"
DB_FILE = "data_pipeline/zepto_catalog.db"


def create_database():
    """Create and populate the normalized SQLite database."""

    # Read cleaned dataset
    df = pd.read_csv(CSV_FILE)

    # Connect to SQLite database
    connection = sqlite3.connect(DB_FILE)

    # Enable foreign-key enforcement
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    # Start fresh so the database always matches the current CSV
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Create categories table
    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # Create books table
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

    # Insert unique categories
    categories = sorted(df["category"].dropna().unique())

    cursor.executemany(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        [(category,) for category in categories],
    )

    # Create category name -> category ID mapping
    category_map = {}

    cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    )

    for category_id, category_name in cursor.fetchall():
        category_map[category_name] = category_id

    # Prepare book records
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

    # Insert books
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
        book_records,
    )

    connection.commit()

    # Verify category count
    cursor.execute(
        """
        SELECT COUNT(*) FROM categories
        """
    )

    category_count = cursor.fetchone()[0]

    # Verify book count
    cursor.execute(
        """
        SELECT COUNT(*) FROM books
        """
    )

    book_count = cursor.fetchone()[0]

    # Verify JOIN
    join_query = """
        SELECT
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
        LIMIT 5
    """

    join_result = pd.read_sql_query(join_query, connection)

    print("\nDatabase created successfully.")
    print(f"Categories loaded: {category_count}")
    print(f"Books loaded: {book_count}")

    print("\nCategories table:")
    print(
        pd.read_sql_query(
            """
            SELECT *
            FROM categories
            ORDER BY category_id
            """,
            connection,
        )
    )

    print("\nSample JOIN result:")
    print(join_result)

    # Verify expected relationship
    if category_count >= 3 and book_count >= 60:
        print("\nDatabase verification PASSED.")
    else:
        print("\nDatabase verification FAILED.")

    connection.close()


if __name__ == "__main__":
    create_database()