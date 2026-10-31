-- Query: ANOVA inputs (factor level means, cell statistics, and pure error)
-- Part 1: Factor A main effect level means
-- Part 2: Factor B main effect level means
-- Part 3: Pure error calculation from center-point replicates

WITH center_replicates AS (
    SELECT
        run_id,
        A,
        B,
        viscosity,
        ph,
        AVG(viscosity) OVER () AS mean_center_viscosity,
        AVG(ph) OVER () AS mean_center_ph
    FROM runs
    WHERE run_type = 'center'
),
pure_error_summary AS (
    SELECT
        'Pure Error' AS source,
        COUNT(*) - 1 AS df_pure_error,
        ROUND(SUM((viscosity - mean_center_viscosity) * (viscosity - mean_center_viscosity)), 2) AS ss_pure_error_viscosity,
        ROUND(SUM((ph - mean_center_ph) * (ph - mean_center_ph)), 4) AS ss_pure_error_ph
    FROM center_replicates
)
SELECT * FROM pure_error_summary;
