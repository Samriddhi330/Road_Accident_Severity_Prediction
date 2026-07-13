================================================================================
ROAD ACCIDENT SEVERITY PREDICTION - DATA PREPROCESSING LOGS
================================================================================

1. DATA COLLECTION & UNDERSTANDING
--------------------------------------------------------------------------------
- Initial Dataset Size: 100,000 rows, 46 structural columns.
- Primary Target Variable: 'Severity' (Traffic disruption scale rated 1 to 4).
- Key Insight: Dataset is highly skewed towards California (CA), which accounts 
  for 99.2% of all accidents in this sample profile.

2. MISSING VALUES & DUPLICATES LOG
--------------------------------------------------------------------------------
- Action Taken: Dropped structural columns 'End_Lat' and 'End_Lng' due to 
  containing zero (0) valid non-null entries.
- Action Taken: Dropped low-information features 'Wind_Chill(F)' and 
  'Precipitation(in)' because over 90% of cells were missing.
- Action Taken: Dropped tracking variables 'ID' and unique strings 'Description' 
  to eliminate pure textual background noise.
- Imputation Rule: Missing numbers in operational weather logs were safely 
  patched using column median index values to resist average inflation. Missing 
  text labels were patched using column mode arrays (Value: "Fair").
- Deduplication: Identified and permanently removed 17 exact carbon-copy duplicate 
  rows, leaving a clean master set of 99,983 records.

3. OUTLIER LOG & TREATMENT
--------------------------------------------------------------------------------
- Anomaly Detected: 'Wind_Speed(mph)' displayed a broken tracking max reading 
  of 241.7 mph, representing unrealistic hardware telemetry errors.
- Resolution Rule: Applied a 99th percentile statistical cutoff boundary cap. All 
  values exceeding this threshold were squeezed down to the boundary limits 
  without deleting entire rows.

4. CATEGORICAL VARIABLE ENCODING
--------------------------------------------------------------------------------
- High-Volume Features ('Street', 'City', 'Zipcode', 'Airport_Code'): Converted 
  using simple category-index integer codes (.cat.codes) to keep file matrices 
  small and fast.
- Low-Volume Features ('Timezone', 'State'): Unpacked into transparent binary 
  columns using pd.get_dummies to prevent sequence value assumptions.
- Road Flags (True/False): Encoded as pure binary digits (True -> 1, False -> 0).

5. DATA SCALING & TRAIN-TEST PARTITION
--------------------------------------------------------------------------------
- Partition Ratio: 80% assigned to model training sets, 20% assigned to a 
  completely isolated validation test set.
- Strategy: Configured 'stratify=y' to distribute equal proportions of rare 
  Severity 1 and 4 incidents across both study piles.
- Feature Normalization: Continuous weather metrics scaled using StandardScaler. 
  The function calculated parameters strictly from training sets and transformed 
  test sets directly to lock out data leakage.

================================================================================
STATUS: FINAL PROCESSED FILES DEPLOYED TO DATA/PROCESSED/ DIRECTORY
================================================================================