--1
SELECT churn, COUNT(*), 
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM customers
GROUP BY churn;


--2
SELECT contract,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


--3
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    ROUND(100.0 * COUNT(*) FILTER (WHERE churn) / COUNT(*), 2) AS churn_rate_pct,
    COUNT(*) AS total
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);



--4
SELECT payment_method,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;



--5
SELECT internet_service,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

--6
SELECT churn, 
       ROUND(AVG(monthly_charges), 2) AS avg_monthly,
       ROUND(AVG(total_charges), 2) AS avg_total
FROM customers
GROUP BY churn;