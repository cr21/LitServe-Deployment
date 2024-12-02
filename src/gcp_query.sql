CREATE OR REPLACE PROCEDURE join_tables(dst_flag BOOL)
BEGIN
  -- Declare variables for the loop
  DECLARE done BOOL DEFAULT FALSE;
  DECLARE current_date DATE;
  DECLARE current_hour INT64;
  
  -- Create a temporary table to store results
  CREATE TEMP TABLE IF NOT EXISTS joined_results AS (
    SELECT 
      trans_dt,
      start_date,
      start_hr,
      subscriber_msisdn,
      imsi,
      AVG_5G_RSRP,
      timezone,
      DEVICE_RSRP,
      qos_trans_dt,
      qos_utc_hour
    FROM fwa_mdn_hc_ext
    WHERE 1=0  -- Empty template table
  );

  -- Loop through distinct dates and hours
  FOR date_hour_record IN (
    SELECT DISTINCT start_date, start_hr 
    FROM fwa_mdn_hc_ext 
    ORDER BY start_date, start_hr
  ) DO
    -- Insert results for current date and hour
    INSERT INTO joined_results
    WITH fwa_utc_converted AS (
      SELECT 
        trans_dt,
        start_date,
        start_hr,
        subscriber_msisdn,
        imsi,
        AVG_5G_RSRP,
        timezone,
        -- Calculate offset based on dst_flag
        CASE timezone
          WHEN 'MST' THEN IF(dst_flag, 6, 7)
          WHEN 'PST' THEN IF(dst_flag, 7, 8)
          WHEN 'CST' THEN IF(dst_flag, 5, 6)
          WHEN 'EST' THEN IF(dst_flag, 4, 5)
        END AS timezone_offset,
        -- Convert local time to UTC
        MOD(start_hr + 
          CASE timezone
            WHEN 'MST' THEN IF(dst_flag, 6, 7)
            WHEN 'PST' THEN IF(dst_flag, 7, 8)
            WHEN 'CST' THEN IF(dst_flag, 5, 6)
            WHEN 'EST' THEN IF(dst_flag, 4, 5)
          END + 24, 24) AS utc_hour,
        -- Handle date rollover
        DATE_ADD(start_date, 
          INTERVAL CASE WHEN start_hr + 
            CASE timezone
              WHEN 'MST' THEN IF(dst_flag, 6, 7)
              WHEN 'PST' THEN IF(dst_flag, 7, 8)
              WHEN 'CST' THEN IF(dst_flag, 5, 6)
              WHEN 'EST' THEN IF(dst_flag, 4, 5)
            END >= 24 THEN 1 ELSE 0 END DAY
        ) AS utc_date
      FROM 
        fwa_mdn_hc_ext
      WHERE 
        start_date = date_hour_record.start_date
        AND start_hr = date_hour_record.start_hr
    )
    
    SELECT 
      fwa.trans_dt,
      fwa.start_date,
      fwa.start_hr,
      fwa.subscriber_msisdn,
      fwa.imsi,
      fwa.AVG_5G_RSRP,
      fwa.timezone,
      qos.DEVICE_RSRP,
      qos.trans_dt AS qos_trans_dt,
      EXTRACT(HOUR FROM qos.event_time) AS qos_utc_hour
    FROM 
      fwa_utc_converted fwa
    JOIN 
      qos_performance_hrly qos
    ON 
      fwa.imsi = qos.imsi
      AND fwa.subscriber_msisdn = qos.mtn
      AND fwa.utc_date = qos.trans_dt
      AND fwa.utc_hour = EXTRACT(HOUR FROM qos.event_time)
    WHERE
      qos.trans_dt BETWEEN 
        DATE_SUB(date_hour_record.start_date, INTERVAL 1 DAY) 
        AND DATE_ADD(date_hour_record.start_date, INTERVAL 1 DAY);
  END FOR;

  -- Return final results
  SELECT * FROM joined_results
  ORDER BY trans_dt, start_date, start_hr;

  -- Clean up
  DROP TABLE joined_results;
END;

-- fwa_mdn_hc_ext 
--     trans_dt,
--     trans_hr,
--     start_date
--     start_hr,
--     subscriber_msisdn,
--     imsi,
--     AVG_5G_RSRP,
--     timezone



    
