-- Query: Design matrix with center-point replication statistics
-- Extracts all runs with factor levels, responses, and run type annotations.

SELECT
    r.run_id,
    r.run_type,
    r.A AS carbomer_pct,
    r.B AS naoh_ratio,
    r.viscosity,
    r.ph,
    COUNT(*) OVER (PARTITION BY r.A, r.B) AS cell_replicate_count,
    ROUND(AVG(r.viscosity) OVER (PARTITION BY r.A, r.B), 1) AS cell_mean_viscosity,
    ROUND(AVG(r.ph) OVER (PARTITION BY r.A, r.B), 2) AS cell_mean_ph
FROM runs r
ORDER BY r.run_id;
