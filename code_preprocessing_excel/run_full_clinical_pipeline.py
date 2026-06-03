"""
Full clinical preprocessing pipeline.

This script:
1. Reads the raw normal control Excel file.
2. Extracts structured clinical features with the rule-based NLP pipeline.
3. Reads the already structured TMD clinical dataset.
4. Aligns both datasets to one common schema.
5. Merges Normal + TMD rows.
6. Normalizes diagnosis columns and creates binary diagnosis features.
7. Saves the merged datasets and the diagnosis normalization table.

Usage:
    python run_full_clinical_pipeline.py

Optional:
    python run_full_clinical_pipeline.py \
        --normal-xlsx "C:/Users/sadmin/Downloads/normal_clinical(1).xlsx" \
        --tmd-csv "./clinicaldataa.csv" \
        --output-dir "./outputs"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from normalize_diagnosis_features import add_diagnosis_features, collect_original_mapping
from preprocess_clinical_features import OUTPUT_COLUMNS, transform_clinical_dataframe


PROJECT_DIR = Path(__file__).resolve().parent


TMD_TO_COMMON_COLUMNS = {
    "Case ID No.": "Case ID",
    "DATE OF VISIT": "DATE OF VISIT",
    "Birth Date": "Birth Date",
    "age": "age",
    "age_visit": "age_visit",
    "age_range": "age_range",
    "gender": "gender",
    "race": "race",
    "pain_score": "pain_scor",
    "painarea_PA": "painarea_PA",
    "painarea_P": "painarea_P",
    "painarea_T": "painarea_T",
    "ttpp_TMJ": "ttpp_TMJ",
    "ttpp_M": "ttpp_M",
    "ttpp_T": "ttpp_T",
    "ttpp_P": "ttpp_P",
    "crep": "crep",
    "click": "click",
    "dev": "dev",
    "disl": "disl",
    "mmo": "mmo",
    "mio": "mio",
    "lat_R": "lat_R",
    "lat_L": "lat_L",
    "maxprot_OJ": "maxprot",
    "teeth_wear": "teeth_wea",
    "condyle_WD": "condyle",
    "teeth_grind": "teeth_grin",
    "hard_foods": "hard_food",
    "MD_autoimmune": "MD_autoi",
    "MD_CnR": "MD_CnR",
    "MD_CP": "MD_CP",
    "MD_mental": "MD_ment",
    "MD_gastro": "MD_gastr",
    "MD_neuro": "MD_neuro",
    "MD_infect": "MD_infect",
    "MD_derm": "MD_derm",
    "severity": "severity",
    "diagnosis": "diagnosis",
}


COMMON_COLUMNS = [
    "source_dataset",
    "Case ID",
    "DATE OF VISIT",
    *[col for col in OUTPUT_COLUMNS if col != "Case ID"],
    # Extra original TMD features kept to avoid dropping clinical signal.
    "painarea_M",
    "painarea_TMJ",
]


def clean_severity_series(series: pd.Series) -> pd.Series:
    mapping = {
        "normal": "Normal",
        "mild": "Mild",
        "severe": "Severe",
    }
    return series.astype("string").str.strip().str.lower().map(mapping).fillna(series)


def load_normal_structured(normal_xlsx: Path, normal_sheet: str) -> pd.DataFrame:
    normal_raw = pd.read_excel(normal_xlsx, sheet_name=normal_sheet)
    normal_structured = transform_clinical_dataframe(normal_raw)
    normal_structured.insert(0, "DATE OF VISIT", normal_raw.get("DATE OF VISIT", np.nan))
    normal_structured.insert(0, "source_dataset", "normal_clinical")
    return normal_structured


def load_tmd_structured(tmd_csv: Path) -> pd.DataFrame:
    tmd_raw = pd.read_csv(tmd_csv)
    tmd = pd.DataFrame()

    for old_col, new_col in TMD_TO_COMMON_COLUMNS.items():
        if old_col in tmd_raw.columns:
            tmd[new_col] = tmd_raw[old_col]

    # TMD CSV stores secondary/tertiary diagnoses as diagnosis2/diagnosis3.
    # In the common schema they become diagnosis1/diagnosis2, so diagnosis3
    # stays available for a possible fourth diagnosis.
    tmd["diagnosis1"] = tmd_raw["diagnosis2"] if "diagnosis2" in tmd_raw.columns else np.nan
    tmd["diagnosis2"] = tmd_raw["diagnosis3"] if "diagnosis3" in tmd_raw.columns else np.nan
    tmd["diagnosis3"] = np.nan

    # The normal-control schema has side-level painarea_L/R. For the existing
    # TMD table these are inferred from any side-coded pain area feature.
    side_cols = [col for col in ["painarea_PA", "painarea_P", "painarea_T", "painarea_M", "painarea_TMJ"] if col in tmd_raw.columns]
    if side_cols:
        has_left = tmd_raw[side_cols].astype(str).apply(lambda row: any(v in {"L", "B"} for v in row), axis=1)
        has_right = tmd_raw[side_cols].astype(str).apply(lambda row: any(v in {"R", "B"} for v in row), axis=1)
        tmd["painarea_L"] = pd.Series(np.nan, index=tmd_raw.index, dtype="object")
        tmd.loc[has_left, "painarea_L"] = "L"
        tmd["painarea_R"] = pd.Series(np.nan, index=tmd_raw.index, dtype="object")
        tmd.loc[has_right, "painarea_R"] = "R"
    else:
        tmd["painarea_L"] = np.nan
        tmd["painarea_R"] = np.nan

    # Preserve these TMD-specific pain area columns in the merged output.
    tmd["painarea_M"] = tmd_raw["painarea_M"] if "painarea_M" in tmd_raw.columns else np.nan
    tmd["painarea_TMJ"] = tmd_raw["painarea_TMJ"] if "painarea_TMJ" in tmd_raw.columns else np.nan

    tmd.insert(0, "source_dataset", "tmd_clinical")
    return tmd


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for col in COMMON_COLUMNS:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[COMMON_COLUMNS]
    aligned["severity"] = clean_severity_series(aligned["severity"])
    return aligned


def run_pipeline(
    normal_xlsx: Path,
    tmd_csv: Path,
    output_dir: Path,
    normal_sheet: str = "Control group",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_structured = align_columns(load_normal_structured(normal_xlsx, normal_sheet))
    tmd_structured = align_columns(load_tmd_structured(tmd_csv))

    merged = pd.concat([tmd_structured, normal_structured], ignore_index=True)
    merged = merged.sort_values(["source_dataset", "Case ID"], kind="stable").reset_index(drop=True)

    featured = add_diagnosis_features(merged)
    mapping = collect_original_mapping(merged)

    paths = {
        "normal_structured": output_dir / "normal_clinical_structured.csv",
        "tmd_structured": output_dir / "tmd_clinical_structured_aligned.csv",
        "merged_structured": output_dir / "clinical_merged_structured.csv",
        "merged_featured": output_dir / "clinical_merged_with_diag_features.csv",
        "diagnosis_mapping": output_dir / "clinical_merged_diagnosis_normalization_table.csv",
    }

    normal_structured.to_csv(paths["normal_structured"], index=False)
    tmd_structured.to_csv(paths["tmd_structured"], index=False)
    merged.to_csv(paths["merged_structured"], index=False)
    featured.to_csv(paths["merged_featured"], index=False)
    mapping.to_csv(paths["diagnosis_mapping"], index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Normal + TMD clinical preprocessing pipeline.")
    parser.add_argument(
        "--normal-xlsx",
        default=r"C:\Users\sadmin\Downloads\normal_clinical(1).xlsx",
        help="Raw normal clinical Excel file.",
    )
    parser.add_argument(
        "--normal-sheet",
        default="Control group",
        help="Sheet name in the normal clinical Excel file.",
    )
    parser.add_argument(
        "--tmd-csv",
        default=str(PROJECT_DIR / "clinicaldataa.csv"),
        help="Structured TMD clinical CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs" / "clinical_pipeline"),
        help="Output directory.",
    )
    args = parser.parse_args()

    paths = run_pipeline(
        normal_xlsx=Path(args.normal_xlsx),
        tmd_csv=Path(args.tmd_csv),
        output_dir=Path(args.output_dir),
        normal_sheet=args.normal_sheet,
    )

    featured = pd.read_csv(paths["merged_featured"])
    print("Full clinical pipeline completed.")
    print(f"Rows: {len(featured)}")
    print(f"Columns: {len(featured.columns)}")
    print("Severity counts:")
    print(featured["severity"].value_counts(dropna=False).to_string())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
