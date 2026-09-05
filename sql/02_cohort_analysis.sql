-- Query 1: Readmission rate by primary diagnosis category
SELECT
    diag_1_cat,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY diag_1_cat
ORDER BY readmission_rate_pct DESC;

-- Query 2: Readmission rate by payer code (insurance type)
SELECT
    payer_code,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY payer_code
ORDER BY readmission_rate_pct DESC;

-- Query 3: Readmission rate by length of stay (time_in_hospital buckets)
SELECT
    CASE
        WHEN time_in_hospital <= 2  THEN '1-2 days'
        WHEN time_in_hospital <= 5  THEN '3-5 days'
        WHEN time_in_hospital <= 9  THEN '6-9 days'
        ELSE '10+ days'
    END                                                  AS los_bucket,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
GROUP BY los_bucket
ORDER BY readmission_rate_pct DESC;

-- Query 4: Prior utilization impact — inpatient visits in past year vs readmission
SELECT
    number_inpatient,
    COUNT(*)                                             AS total_encounters,
    SUM(readmitted_binary)                               AS readmissions,
    ROUND(100.0 * SUM(readmitted_binary) / COUNT(*), 2) AS readmission_rate_pct
FROM patient_encounters
WHERE number_inpatient <= 10
GROUP BY number_inpatient
ORDER BY number_inpatient;
