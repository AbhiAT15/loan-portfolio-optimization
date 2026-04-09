SELECT is_default, COUNT(*) 
FROM loan_analytics 
GROUP BY is_default
ORDER BY is_default;