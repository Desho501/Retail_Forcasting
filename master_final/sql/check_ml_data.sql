-- Run this in DBeaver before running the Python model.
-- It checks whether the main Power BI/modeling view has usable data.

SELECT COUNT(*) AS total_rows
FROM m5.sales_modeling_view;

SELECT
    MIN(calendar_date) AS first_date,
    MAX(calendar_date) AS last_date,
    COUNT(DISTINCT store_id) AS stores,
    COUNT(DISTINCT cat_id) AS categories,
    SUM(units_sold) AS total_units_sold
FROM m5.sales_modeling_view;

SELECT
    DATE_TRUNC('week', calendar_date)::date AS week_start,
    store_id,
    cat_id,
    SUM(units_sold) AS weekly_units_sold,
    SUM(COALESCE(revenue, 0)) AS weekly_revenue
FROM m5.sales_modeling_view
WHERE calendar_date IS NOT NULL
GROUP BY
    DATE_TRUNC('week', calendar_date)::date,
    store_id,
    cat_id
ORDER BY week_start, store_id, cat_id
LIMIT 50;
