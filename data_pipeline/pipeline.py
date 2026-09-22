import pandas as pd

from scraper import scrape_books


GBP_TO_INR = 105.50


def clean_price(price):
    """Convert a scraped GBP price into a numeric value."""
    try:
        return float(
            str(price)
            .replace("£", "")
            .replace("Â", "")
            .strip()
        )
    except (ValueError, TypeError):
        return None


def clean_rating(rating):
    """Convert word-based star ratings into integers from 1 to 5."""
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    return rating_map.get(str(rating).strip())


def clean_availability(value):
    """Convert availability text into a Boolean value."""
    if value is None:
        return False

    return "in stock" in str(value).lower()


def clean_data(books):
    """Clean scraped book records and return a pandas DataFrame."""

    df = pd.DataFrame(books)

    # Clean price
    df["price_gbp"] = df["price_gbp"].apply(clean_price)

    # Convert star rating words to integers
    df["star_rating"] = df["star_rating"].apply(clean_rating)

    # Convert availability text to Boolean
    df["in_stock"] = df["availability"].apply(clean_availability)

    # Convert GBP price to INR using the fixed project conversion rate
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    # Remove the original availability text column
    df.drop(columns=["availability"], inplace=True)

    # Handle numeric parsing failures using median imputation
    for column in ["price_gbp", "star_rating", "price_inr"]:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    # Ensure star rating is an integer from 1 to 5
    df["star_rating"] = df["star_rating"].round().astype(int)

    return df


def run_pipeline():
    """Scrape and clean the book data."""

    # Scrape at least 60 books from at least 3 categories
    books = scrape_books(min_books=60, min_categories=3)

    # Clean the scraped data
    df = clean_data(books)

    # Save the cleaned dataset
    output_file = "data_pipeline/cleaned_books.csv"
    df.to_csv(output_file, index=False)

    print("\nPipeline completed successfully.")
    print(f"Total records: {len(df)}")
    print(f"Saved to: {output_file}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 cleaned records:")
    print(df.head())


if __name__ == "__main__":
    run_pipeline()