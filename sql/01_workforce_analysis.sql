-- Reproducible workforce business views for SQLite.
-- Attrition is an observed association, not a causal conclusion.

DROP VIEW IF EXISTS overall_attrition;
CREATE VIEW overall_attrition AS
SELECT COUNT(*) AS employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate
FROM employees;

DROP VIEW IF EXISTS attrition_by_department;
CREATE VIEW attrition_by_department AS
SELECT Department, COUNT(*) AS employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate
FROM employees GROUP BY Department ORDER BY attrition_rate DESC;

DROP VIEW IF EXISTS attrition_by_overtime;
CREATE VIEW attrition_by_overtime AS
SELECT OverTime, COUNT(*) AS employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate
FROM employees GROUP BY OverTime ORDER BY attrition_rate DESC;

DROP VIEW IF EXISTS attrition_by_job_role;
CREATE VIEW attrition_by_job_role AS
SELECT JobRole, COUNT(*) AS employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate
FROM employees GROUP BY JobRole ORDER BY attrition_rate DESC;

DROP VIEW IF EXISTS income_by_attrition;
CREATE VIEW income_by_attrition AS
SELECT Attrition, COUNT(*) AS employees,
       ROUND(AVG(MonthlyIncome), 2) AS average_monthly_income,
       MIN(MonthlyIncome) AS minimum_monthly_income,
       MAX(MonthlyIncome) AS maximum_monthly_income
FROM employees GROUP BY Attrition ORDER BY Attrition DESC;

DROP VIEW IF EXISTS tenure_by_attrition;
CREATE VIEW tenure_by_attrition AS
SELECT Attrition, COUNT(*) AS employees,
       ROUND(AVG(YearsAtCompany), 2) AS average_years_at_company,
       ROUND(AVG(TotalWorkingYears), 2) AS average_total_working_years
FROM employees GROUP BY Attrition ORDER BY Attrition DESC;
