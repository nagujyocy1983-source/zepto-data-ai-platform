import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"


def get_soup(url):
    """Fetch a webpage and return its BeautifulSoup object."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    """Return all available book categories."""
    soup = get_soup(BASE_URL)

    categories = {}

    for link in soup.select(".side_categories ul li ul li a"):
        category_name = link.get_text(strip=True)
        category_url = urljoin(BASE_URL, link.get("href"))
        categories[category_name] = category_url

    return categories


def scrape_category(category_name, category_url):
    """Scrape all books from one category across its pagination."""
    books = []
    page_url = category_url

    while page_url:
        soup = get_soup(page_url)

        for book in soup.select("article.product_pod"):
            title = book.select_one("h3 a")["title"]
            price_text = book.select_one("p.price_color").get_text(strip=True)
            rating_text = book.select_one("p.star-rating").get("class")[1]
            availability = book.select_one(".availability").get_text(
                " ", strip=True
            )

            books.append(
                {
                    "title": title,
                    "price_gbp": price_text,
                    "star_rating": rating_text,
                    "availability": availability,
                    "category": category_name,
                }
            )

        next_link = soup.select_one("li.next a")

        if next_link:
            page_url = urljoin(page_url, next_link["href"])
        else:
            page_url = None

    return books


def scrape_books(min_books=60, min_categories=3):
    """
    Scrape books from categories until both requirements are satisfied:
    - At least min_books books
    - At least min_categories categories
    """
    categories = get_categories()
    all_books = []
    scraped_categories = set()

    for category_name, category_url in categories.items():
        print(f"Scraping category: {category_name}")

        category_books = scrape_category(category_name, category_url)

        all_books.extend(category_books)
        scraped_categories.add(category_name)

        print(f"Collected so far: {len(all_books)}")
        print(f"Categories scraped: {len(scraped_categories)}")

        if (
            len(all_books) >= min_books
            and len(scraped_categories) >= min_categories
        ):
            break

    return all_books


if __name__ == "__main__":
    books = scrape_books(min_books=60, min_categories=3)

    print(f"\nTotal books collected: {len(books)}")
    print(f"Total categories scraped: {len(set(book['category'] for book in books))}")

    print("\nCategories collected:")

    for category in sorted(set(book["category"] for book in books)):
        count = sum(1 for book in books if book["category"] == category)
        print(f"- {category}: {count} books")

    print("\nFirst 5 records:")

    for book in books[:5]:
        print(book)