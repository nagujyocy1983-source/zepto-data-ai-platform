-- Zepto Data & AI Platform
-- Module 1: Data Pipeline
-- SQL Analysis Queries


-- ============================================================
-- Query 1: List all distinct categories
-- Demonstrates: SELECT, DISTINCT, ORDER BY
-- ============================================================

SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;


-- ============================================================
-- Query 2: Books priced between GBP 20 and GBP 40
-- Demonstrates: SELECT, WHERE, BETWEEN, ORDER BY, LIMIT
-- ============================================================

SELECT
    book_id,
    title,
    price_gbp,
    price_inr,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp DESC
LIMIT 10;


-- ============================================================
-- Query 3: Books with 4 or 5 star ratings
-- Demonstrates: SELECT, WHERE, IN, ORDER BY, LIMIT
-- ============================================================

SELECT
    book_id,
    title,
    rating,
    price_gbp,
    price_inr
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC
LIMIT 10;


-- ============================================================
-- Query 4: Book count and average price by category
-- Demonstrates: JOIN, GROUP BY, ORDER BY
-- ============================================================

SELECT
    c.category_name,
    COUNT(b.book_id) AS book_count,
    ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
FROM categories AS c
JOIN books AS b
    ON c.category_id = b.category_id
GROUP BY c.category_name
ORDER BY average_price_gbp DESC;


-- ============================================================
-- Query 5: Top 10 most expensive books with category
-- Demonstrates: JOIN, ORDER BY, LIMIT
-- ============================================================

SELECT
    b.book_id,
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.price_inr DESC
LIMIT 10;