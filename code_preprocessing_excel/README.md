# Excel Clinical Preprocessing + Diagnosis Normalization

Clean folder for converting raw clinical Excel notes into structured ML-ready clinical features.

## Goal

This folder handles:

1. Raw Excel clinical preprocessing.
2. Rule-based clinical feature extraction.
3. Diagnosis spelling/variant normalization.
4. Binary `diag_*` feature creation.
5. Optional merge with the existing TMD clinical table.

## Main scripts

- `preprocess_clinical_features.py`
  - Converts raw clinical text columns into structured features.
  - Extracts pain areas, click, crepitus, tenderness, mouth opening, lateral movements, dental history, medical history groups, diagnosis fields and severity.

- `normalize_diagnosis_features.py`
  - Normalizes `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`.
  - Groups variants and typos into canonical diagnoses.
  - Creates binary ML columns such as `diag_impacted_tooth`, `diag_periodontitis`, `diag_crowding`, etc.

- `run_full_clinical_pipeline.py`
  - Runs the complete workflow:
    - preprocess `normal_clinical(1).xlsx`
    - align `clinicaldataa.csv`
    - merge Normal + Mild + Severe cases
    - normalize diagnoses
    - save final structured datasets

## Recommended full run

From the project root:

```powershell
python code_preprocessing_excel\run_full_clinical_pipeline.py `
  --normal-xlsx "C:\Users\sadmin\Downloads\normal_clinical(1).xlsx" `
  --tmd-csv "C:\Users\sadmin\Desktop\mozo\clinicaldataa.csv" `
  --output-dir "C:\Users\sadmin\Desktop\mozo\outputs\clinical_pipeline"
```

Expected outputs:

- `normal_clinical_structured.csv`
- `tmd_clinical_structured_aligned.csv`
- `clinical_merged_structured.csv`
- `clinical_merged_with_diag_features.csv`
- `clinical_merged_diagnosis_normalization_table.csv`

## Single-step preprocessing only

```powershell
python code_preprocessing_excel\preprocess_clinical_features.py `
  --input "C:\Users\sadmin\Downloads\normal_clinical(1).xlsx" `
  --output "C:\Users\sadmin\Desktop\mozo\outputs\clinical_pipeline\normal_clinical_structured.csv"
```

## Diagnosis normalization only

```powershell
python code_preprocessing_excel\normalize_diagnosis_features.py `
  --input "C:\Users\sadmin\Desktop\mozo\outputs\clinical_pipeline\clinical_merged_structured.csv" `
  --output "C:\Users\sadmin\Desktop\mozo\outputs\clinical_pipeline\clinical_merged_with_diag_features.csv" `
  --mapping-output "C:\Users\sadmin\Desktop\mozo\outputs\clinical_pipeline\clinical_merged_diagnosis_normalization_table.csv"
```

## Leakage note

For the final non-leaky RF model, do not use `diagnosis`, `diagnosis1`, `diagnosis2`, `diagnosis3`, `diag_*`, or `condyle`.

These diagnosis features are still produced here because they are useful for audit, diagnosis normalization, and alternative clinical analyses.

