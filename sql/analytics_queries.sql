-- Advanced SQL Analytics Queries for SmartHouse AI

-- Query 1: Average Price and Stats by City
-- Summarizes overall pricing metrics across the primary cities
-- SQL_QUERY_1_START
SELECT l.city, l.state, 
       COUNT(p.property_id) AS total_properties,
       ROUND(AVG(hs.sale_price), 2) AS avg_sale_price,
       ROUND(MIN(hs.sale_price), 2) AS min_sale_price,
       ROUND(MAX(hs.sale_price), 2) AS max_sale_price
FROM properties p
JOIN locations l ON p.location_id = l.location_id
JOIN historical_sales hs ON p.property_id = hs.property_id
GROUP BY l.city, l.state
ORDER BY avg_sale_price DESC;
-- SQL_QUERY_1_END


-- Query 2: Top Appreciating Areas (Zip Codes)
-- Compares average pricing in 2022 vs 2026 to rank appreciation using Dense Rank
-- SQL_QUERY_2_START
WITH YearlyZipPrices AS (
    SELECT l.zip_code, l.city,
           strftime('%Y', hs.sale_date) AS sale_year,
           AVG(hs.sale_price) AS avg_price
    FROM historical_sales hs
    JOIN properties p ON hs.property_id = p.property_id
    JOIN locations l ON p.location_id = l.location_id
    GROUP BY l.zip_code, l.city, sale_year
),
AppreciationStats AS (
    SELECT y1.zip_code, y1.city,
           y1.avg_price AS price_2022,
           y2.avg_price AS price_2026,
           ((y2.avg_price - y1.avg_price) / y1.avg_price) * 100 AS appreciation_pct
    FROM YearlyZipPrices y1
    JOIN YearlyZipPrices y2 ON y1.zip_code = y2.zip_code AND y1.sale_year = '2022' AND y2.sale_year = '2026'
)
SELECT zip_code, city, 
       ROUND(price_2022, 2) AS avg_price_2022, 
       ROUND(price_2026, 2) AS avg_price_2026, 
       ROUND(appreciation_pct, 2) AS growth_percentage,
       DENSE_RANK() OVER (ORDER BY appreciation_pct DESC) AS appreciation_rank
FROM AppreciationStats
ORDER BY growth_percentage DESC;
-- SQL_QUERY_2_END


-- Query 3: Market-wide Price Trends (Month-over-Month Change)
-- Uses Lag window function to calculate month-over-month growth metrics
-- SQL_QUERY_3_START
WITH MonthlyMetrics AS (
    SELECT strftime('%Y-%m', hs.sale_date) AS sale_month,
           COUNT(hs.sale_id) AS sales_volume,
           AVG(hs.sale_price) AS avg_price
    FROM historical_sales hs
    GROUP BY sale_month
)
SELECT sale_month,
       sales_volume,
       ROUND(avg_price, 2) AS average_price,
       ROUND(avg_price - LAG(avg_price, 1) OVER (ORDER BY sale_month), 2) AS price_change_mom,
       ROUND(((avg_price - LAG(avg_price, 1) OVER (ORDER BY sale_month)) / LAG(avg_price, 1) OVER (ORDER BY sale_month)) * 100, 2) AS pct_change_mom
FROM MonthlyMetrics
ORDER BY sale_month DESC;
-- SQL_QUERY_3_END


-- Query 4: Property Type Analysis
-- Breaks down square footage and listing volume metrics by property type
-- SQL_QUERY_4_START
SELECT p.property_type,
       COUNT(p.property_id) AS listing_count,
       ROUND(AVG(p.area_sqft), 1) AS avg_area_sqft,
       ROUND(AVG(hs.sale_price), 2) AS avg_price,
       ROUND(AVG(hs.sale_price / p.area_sqft), 2) AS avg_price_per_sqft
FROM properties p
JOIN historical_sales hs ON p.property_id = hs.property_id
GROUP BY p.property_type
ORDER BY avg_price DESC;
-- SQL_QUERY_4_END


-- Query 5: Neighborhood Infrastructure & Performance
-- Analyzes city metrics by local school ratings and crime index
-- SQL_QUERY_5_START
SELECT l.city,
       p.school_rating,
       p.crime_rate,
       COUNT(p.property_id) AS property_count,
       ROUND(AVG(hs.sale_price), 2) AS avg_price,
       ROUND(AVG(hs.sale_price / p.area_sqft), 2) AS avg_price_per_sqft
FROM properties p
JOIN locations l ON p.location_id = l.location_id
JOIN historical_sales hs ON p.property_id = hs.property_id
GROUP BY l.city, p.school_rating, p.crime_rate
HAVING COUNT(p.property_id) > 10
ORDER BY avg_price_per_sqft DESC;
-- SQL_QUERY_5_END


-- Query 6: Investment Opportunity Detection (Zip code Undervaluation)
-- Detects properties priced 15% lower than their neighborhood averages
-- SQL_QUERY_6_START
WITH ZipAvgPrices AS (
    SELECT p.property_id, p.property_type, p.area_sqft, l.zip_code, l.city,
           hs.sale_price,
           AVG(hs.sale_price) OVER (PARTITION BY l.zip_code, p.property_type) AS zip_type_avg_price,
           ROW_NUMBER() OVER (PARTITION BY hs.property_id ORDER BY hs.sale_date DESC) as rn
    FROM properties p
    JOIN locations l ON p.location_id = l.location_id
    JOIN historical_sales hs ON p.property_id = hs.property_id
)
SELECT property_id, property_type, zip_code, city, 
       ROUND(sale_price, 2) AS current_price,
       ROUND(zip_type_avg_price, 2) AS avg_zip_type_price,
       ROUND(zip_type_avg_price - sale_price, 2) AS discount_value,
       ROUND(((zip_type_avg_price - sale_price) / zip_type_avg_price) * 100, 2) AS discount_percentage
FROM ZipAvgPrices
WHERE rn = 1 AND sale_price < zip_type_avg_price * 0.85
ORDER BY discount_percentage DESC
LIMIT 100;
-- SQL_QUERY_6_END
