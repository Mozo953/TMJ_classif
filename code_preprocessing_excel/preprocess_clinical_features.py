"""
Rule-based preprocessing for raw TMD clinical notes.

The script converts free-text clinical columns into the structured tabular
features used by the clinical Decision Tree pipeline.

Usage:
    python preprocess_clinical_features.py --input raw_clinical.csv --output structured_clinical.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "Case ID",
    "Birth Date",
    "age",
    "age_visit",
    "age_range",
    "gender",
    "race",
    "pain_scor",
    "painarea_PA",
    "painarea_P",
    "painarea_T",
    "painarea_L",
    "painarea_R",
    "ttpp_TMJ",
    "ttpp_M",
    "ttpp_T",
    "ttpp_P",
    "crep",
    "click",
    "dev",
    "disl",
    "mmo",
    "mio",
    "lat_R",
    "lat_L",
    "maxprot",
    "teeth_wea",
    "condyle",
    "teeth_grin",
    "hard_food",
    "MD_autoi",
    "MD_CnR",
    "MD_CP",
    "MD_ment",
    "MD_gastr",
    "MD_neuro",
    "MD_infect",
    "MD_derm",
    "severity",
    "diagnosis",
    "diagnosis1",
    "diagnosis2",
    "diagnosis3",
]


TEXT_COLUMNS = [
    "CHIEF COMPLAINT",
    "PAIN AREA (joint or muscle)",
    "PAIN AREA (JOINT OR MUSCLE)",
    "SYMPTOMS (TMD)",
    "SYMPTOMS (CLICK/LOCK/SWELLING/CREPITATIONS/DISLOCATION/PAINFUL OPEN/PAINFUL CLOSE/EXAGG OCCLUSAL SENSE) (RIGHT OR LEFT)",
    "MEDICAL HISTORY",
    "DENTAL HISTORY",
    "MEDICAL TREATMENT",
    "DENTAL TREATMENT",
    "SOCIAL HISTORY",
    "CLINICAL FINDINGS",
    "CLINICAL DIAGNOSIS",
    "CLINICAL DIAGNOSIS ",
    "RADIOGRAPHIC FINDINGS",
    "RADIOGRAPHIC FINDINGS ",
]


def first_existing(row: pd.Series, candidates: Iterable[str], default=np.nan):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def concat_text(row: pd.Series, cols: Iterable[str]) -> str:
    parts = []
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return normalize_text(" ".join(parts))


def has_negation_near(text: str, keyword_start: int, window: int = 35) -> bool:
    prefix = text[max(0, keyword_start - window):keyword_start]
    return bool(re.search(r"\b(no|not|nil|none|denies|denied|without|absence of|nad)\b", prefix))


def contains_positive(text: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.I):
            if not has_negation_near(text, match.start()):
                return True
    return False


def side_from_context(text: str, patterns: Iterable[str]) -> float | str:
    hits = []
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.I):
            if has_negation_near(text, match.start()):
                continue
            context = text[max(0, match.start() - 80): match.end() + 80]
            left = bool(re.search(r"\b(left|lhs|lt|l tmj|l/|/l|\bl\b)\b", context))
            right = bool(re.search(r"\b(right|rhs|rt|r tmj|r/|/r|\br\b)\b", context))
            bilateral = bool(re.search(r"\b(bilateral|bilaterally|both|b/l|b/l|l and r|r and l|left and right|right and left)\b", context))
            if bilateral or (left and right):
                hits.append("B")
            elif left:
                hits.append("L")
            elif right:
                hits.append("R")
            else:
                hits.append("Y")
    if not hits:
        return np.nan
    if "B" in hits or ("L" in hits and "R" in hits):
        return "B"
    if "L" in hits:
        return "L"
    if "R" in hits:
        return "R"
    return "Y"


def extract_number_near(text: str, patterns: Iterable[str]) -> float:
    for pat in patterns:
        # keyword before number: "mmo 34", "mouth opening: 40mm"
        m = re.search(rf"(?:{pat})\D{{0,25}}(\d{{1,2}}(?:\.\d+)?)\s*(?:mm)?", text, flags=re.I)
        if m:
            return float(m.group(1))
        # number before keyword: "40mm mouth opening"
        m = re.search(rf"(\d{{1,2}}(?:\.\d+)?)\s*(?:mm)?\D{{0,25}}(?:{pat})", text, flags=re.I)
        if m:
            return float(m.group(1))
    if re.search(r"\bwnl\b|within normal limit", text, flags=re.I):
        return "WNL"
    return np.nan


def parse_measurement_value(value) -> float | str:
    if pd.isna(value):
        return np.nan
    text = normalize_text(value)
    if not text:
        return np.nan
    if re.search(r"\bwnl\b|within normal limit", text, flags=re.I):
        return "WNL"
    if re.search(r"finger|fingers|fingerbreadth|finger width", text, flags=re.I):
        return np.nan
    match = re.search(r"\d{1,2}(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return np.nan


def extract_lateral(text: str, side: str) -> float:
    side_words = {
        "R": r"(?:right|rt|rhs|r)",
        "L": r"(?:left|lt|lhs|l)",
    }[side]
    patterns = [
        rf"{side_words}\s+(?:lateral(?:trusion)?|laterotrusion|excursion)\D{{0,20}}(\d{{1,2}}(?:\.\d+)?)",
        rf"(?:lateral(?:trusion)?|laterotrusion|excursion)\D{{0,20}}{side_words}\D{{0,20}}(\d{{1,2}}(?:\.\d+)?)",
        rf"{side_words}\s+(\d{{1,2}}(?:\.\d+)?)\s*(?:mm)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return float(m.group(1))
    if re.search(r"\bwnl\b|within normal limit", text, flags=re.I):
        return "WNL"
    return np.nan


def age_range(age_visit) -> float | str:
    if pd.isna(age_visit):
        return np.nan
    try:
        age = float(age_visit)
    except ValueError:
        return np.nan
    if age <= 12:
        return "<12"
    if age <= 19:
        return "13-19"
    if age <= 39:
        return "20-39"
    if age <= 64:
        return "40-64"
    return ">65"


def clean_gender(value) -> float | str:
    text = normalize_text(value)
    if re.fullmatch(r"m|male", text):
        return "M"
    if re.fullmatch(r"f|female", text):
        return "F"
    return np.nan if not text else str(value).strip()


def clean_severity(value) -> float | str:
    text = normalize_text(value)
    if not text:
        return np.nan
    if text == "normal":
        return "Normal"
    if text == "mild":
        return "Mild"
    if text == "severe":
        return "Severe"
    return str(value).strip()


def split_diagnoses(value) -> list[float | str]:
    if pd.isna(value):
        return [np.nan, np.nan, np.nan, np.nan]
    text = str(value)
    parts = [p.strip(" -;/\n\t") for p in re.split(r",|;|\n| / |\s-\s", text) if p.strip(" -;/\n\t")]
    parts = parts[:4]
    while len(parts) < 4:
        parts.append(np.nan)
    return parts


@dataclass(frozen=True)
class FeatureRule:
    output: str
    source: str
    rule: str
    keywords: tuple[str, ...]


FEATURE_RULES = [
    FeatureRule("click", "SYMPTOMS/CHIEF COMPLAINT/CLINICAL FINDINGS", "side_from_context", ("click", "clicking", "reciprocal click")),
    FeatureRule("crep", "SYMPTOMS/CLINICAL FINDINGS", "side_from_context", ("crepitus", "crepitation", "creps")),
    FeatureRule("dev", "CLINICAL FINDINGS/SYMPTOMS", "side_from_context", ("deviation", "mandibular deviation", "chin point deviation")),
    FeatureRule("disl", "SYMPTOMS/DIAGNOSIS", "side_from_context", ("dislocation", "dislocat")),
    FeatureRule("mmo", "CLINICAL FINDINGS/SYMPTOMS", "numeric extraction", ("mmo", "maximum mouth opening", "mouth opening")),
    FeatureRule("mio", "CLINICAL FINDINGS/SYMPTOMS", "numeric extraction", ("mio", "active mouth opening", "interincisal opening")),
    FeatureRule("lat_R", "CLINICAL FINDINGS/SYMPTOMS", "right laterotrusion numeric extraction", ("right lateraltrusion", "right laterotrusion", "right excursion")),
    FeatureRule("lat_L", "CLINICAL FINDINGS/SYMPTOMS", "left laterotrusion numeric extraction", ("left lateraltrusion", "left laterotrusion", "left excursion")),
    FeatureRule("maxprot", "CLINICAL FINDINGS/SYMPTOMS", "numeric extraction", ("protrusion", "maximum protrusion", "max protrusion")),
    FeatureRule("teeth_grin", "DENTAL HISTORY/SOCIAL HISTORY/CLINICAL DIAGNOSIS", "Y if positive", ("bruxism", "grinding", "grinds", "clenching")),
    FeatureRule("teeth_wea", "DENTAL HISTORY/CLINICAL FINDINGS/RADIOGRAPHIC FINDINGS", "Y if positive", ("attrition", "wear facet", "teeth wear", "tooth wear")),
    FeatureRule("hard_food", "CHIEF COMPLAINT/SOCIAL HISTORY/DENTAL HISTORY", "Y if positive", ("hard food", "chewy food", "nuts", "hamburger")),
    FeatureRule("condyle", "RADIOGRAPHIC FINDINGS", "side_from_context", ("condyle worn", "condylar wear", "flattening", "erosion", "osteoarth")),
]


MEDICAL_GROUPS = {
    "MD_autoi": [
        r"\bra\b", r"rheumatoid", r"arthritis", r"autoimmune", r"lupus", r"sle",
        r"sjogren", r"thyroid", r"hypothyroid", r"hyperthyroid",
    ],
    "MD_CnR": [
        r"\bht\b", r"\bhtn\b", r"hypertension", r"\bhld\b", r"hyperlipid",
        r"heart", r"cardiac", r"coronary", r"asthma", r"copd", r"respiratory",
        r"diabetes", r"\bdm\b",
    ],
    "MD_CP": [
        r"chronic pain", r"fibromyalgia", r"back pain", r"neck pain", r"migraine",
        r"headache", r"carpal tunnel",
    ],
    "MD_ment": [
        r"anxiety", r"depression", r"psychi", r"adjustment disorder", r"fluoxetine",
        r"stress", r"mental",
    ],
    "MD_gastr": [
        r"gastric", r"gastritis", r"gastro", r"gerd", r"reflux", r"\bbph\b",
        r"stomach", r"ulcer",
    ],
    "MD_neuro": [
        r"stroke", r"\btia\b", r"epilep", r"seizure", r"neuro", r"trigeminal",
        r"neuralgia", r"neuropath", r"gabapentin",
    ],
    "MD_infect": [
        r"dengue", r"hepatitis", r"\bhep\b", r"tb\b", r"tuberculosis",
        r"hiv", r"infection", r"infectious",
    ],
    "MD_derm": [
        r"eczema", r"dermat", r"psoriasis", r"skin", r"rash",
    ],
}


def transform_row(row: pd.Series) -> dict:
    all_text = concat_text(row, TEXT_COLUMNS)
    symptom_text = concat_text(row, [
        "SYMPTOMS (TMD)",
        "SYMPTOMS (CLICK/LOCK/SWELLING/CREPITATIONS/DISLOCATION/PAINFUL OPEN/PAINFUL CLOSE/EXAGG OCCLUSAL SENSE) (RIGHT OR LEFT)",
        "CHIEF COMPLAINT",
        "CLINICAL FINDINGS",
        "PAIN AREA (joint or muscle)",
        "PAIN AREA (JOINT OR MUSCLE)",
    ])
    medical_text = concat_text(row, ["MEDICAL HISTORY"])
    dental_text = concat_text(row, ["DENTAL HISTORY", "SOCIAL HISTORY", "CLINICAL FINDINGS", "CLINICAL DIAGNOSIS", "CLINICAL DIAGNOSIS "])
    radiographic_text = concat_text(row, ["RADIOGRAPHIC FINDINGS", "RADIOGRAPHIC FINDINGS "])

    diagnosis_raw = first_existing(row, ["CLINICAL DIAGNOSIS", "CLINICAL DIAGNOSIS ", "diagnosis"])
    diagnoses = split_diagnoses(diagnosis_raw)

    age_visit_value = first_existing(row, ["age_visit", "AGE (On Visit) (year)", "AGE (ON VISIT) (year)"])
    age_value = first_existing(row, ["age", "Present age (year)", "AGE"])

    out = {
        "Case ID": first_existing(row, ["Case ID", "Case ID No.", "Case ID No"]),
        "Birth Date": first_existing(row, ["Birth Date", "BIRTH DATE"]),
        "age": age_value,
        "age_visit": age_visit_value,
        "age_range": age_range(age_visit_value),
        "gender": clean_gender(first_existing(row, ["gender", "SEX", "sex"])),
        "race": first_existing(row, ["race", "RACE"]),
        "pain_scor": first_existing(row, ["pain_scor", "pain_score", "PAIN SCORE"]),
        "severity": clean_severity(first_existing(row, ["severity", "Severity"])),
        "diagnosis": diagnoses[0],
        "diagnosis1": diagnoses[1],
        "diagnosis2": diagnoses[2],
        "diagnosis3": diagnoses[3],
    }

    out["painarea_PA"] = side_from_context(symptom_text, [r"preauricular|pre-auricular|\bpa\b"])
    out["painarea_P"] = side_from_context(symptom_text, [r"pterygoid|\bp\b"])
    out["painarea_T"] = side_from_context(symptom_text, [r"temporalis|temporal|\bt\b"])
    out["painarea_L"] = "L" if contains_positive(symptom_text, [r"\bleft\b|\blhs\b|\blt\b"]) else np.nan
    out["painarea_R"] = "R" if contains_positive(symptom_text, [r"\bright\b|\brhs\b|\brt\b"]) else np.nan

    out["ttpp_TMJ"] = side_from_context(symptom_text, [r"tmj.{0,35}(?:tender|tpp|ttp|palpation)|(?:tender|tpp|ttp).{0,35}tmj"])
    out["ttpp_M"] = side_from_context(symptom_text, [r"masseter.{0,35}(?:tender|tpp|ttp|palpation)|(?:tender|tpp|ttp).{0,35}masseter"])
    out["ttpp_T"] = side_from_context(symptom_text, [r"temporalis.{0,35}(?:tender|tpp|ttp|palpation)|(?:tender|tpp|ttp).{0,35}temporalis"])
    out["ttpp_P"] = side_from_context(symptom_text, [r"pterygoid.{0,35}(?:tender|tpp|ttp|palpation)|(?:tender|tpp|ttp).{0,35}pterygoid"])

    out["crep"] = side_from_context(symptom_text, [r"crepitus|crepitation|crepitations|creps"])
    out["click"] = side_from_context(symptom_text, [r"click(?:ing)?|clicking sound|reciprocal click"])
    out["dev"] = side_from_context(symptom_text, [r"deviation|deviat(?:es|ion)?|chin point deviation"])
    out["disl"] = side_from_context(all_text, [r"dislocation|dislocat(?:ed|ion)|lock jaw|closed lock"])

    movement_text = concat_text(row, [
        "CLINICAL FINDINGS",
        "SYMPTOMS (TMD)",
        "PAIN AREA (joint or muscle)",
        "PAIN AREA (JOINT OR MUSCLE)",
        "PASSIVE MOUTH OPENING(mm)",
        "ACTIVE MOUTHOPENING (mm)",
        "Right lateraltrusion (mm)",
        "Left lateraltrusion(mm)",
        "MAXIMUM PROTRUSION (MM)",
    ])
    out["mmo"] = parse_measurement_value(first_existing(row, ["mmo", "PASSIVE MOUTH OPENING(mm)"]))
    if pd.isna(out["mmo"]):
        out["mmo"] = extract_number_near(movement_text, [r"mmo", r"maximum mouth opening", r"mouth opening", r"passive mouth opening"])
    out["mio"] = parse_measurement_value(first_existing(row, ["mio", "ACTIVE MOUTHOPENING (mm)"]))
    if pd.isna(out["mio"]):
        out["mio"] = extract_number_near(movement_text, [r"mio", r"active mouth ?opening", r"interincisal opening"])
    out["lat_R"] = parse_measurement_value(first_existing(row, ["lat_R", "Right lateraltrusion (mm)"]))
    if pd.isna(out["lat_R"]):
        out["lat_R"] = extract_lateral(movement_text, "R")
    out["lat_L"] = parse_measurement_value(first_existing(row, ["lat_L", "Left lateraltrusion(mm)"]))
    if pd.isna(out["lat_L"]):
        out["lat_L"] = extract_lateral(movement_text, "L")
    out["maxprot"] = parse_measurement_value(first_existing(row, ["maxprot", "maxprot_OJ", "MAXIMUM PROTRUSION (MM)"]))
    if pd.isna(out["maxprot"]):
        out["maxprot"] = extract_number_near(movement_text, [r"maximum protrusion", r"max protrusion", r"protrusion"])

    out["teeth_wea"] = "Y" if contains_positive(dental_text + " " + radiographic_text, [r"teeth wear", r"tooth wear", r"wear facet", r"attrition", r"erosion"]) else np.nan
    out["condyle"] = side_from_context(radiographic_text, [r"condyl(?:e|ar).{0,60}(?:wear|worn|flatten|erosion|osteoarth|sclerosed|degenerative)|(?:wear|worn|flatten|erosion|osteoarth).{0,60}condyl"])
    out["teeth_grin"] = "Y" if contains_positive(dental_text, [r"bruxism", r"grind(?:s|ing)?", r"teeth grinding", r"clench(?:ing)?"]) else np.nan
    out["hard_food"] = "Y" if contains_positive(all_text, [r"hard food", r"hard or chewy", r"chewy food", r"nuts", r"hamburger"]) else np.nan

    for feature, patterns in MEDICAL_GROUPS.items():
        out[feature] = "Y" if contains_positive(medical_text, patterns) else np.nan

    return out


def transform_clinical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    case_col = next((col for col in ["Case ID", "Case ID No.", "Case ID No"] if col in df.columns), None)
    if case_col is not None:
        df = df[df[case_col].notna()].copy()
    rows = [transform_row(row) for _, row in df.iterrows()]
    final = pd.DataFrame(rows)
    return final.reindex(columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw clinical TMD notes into structured features.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file containing raw clinical columns.")
    parser.add_argument("--output", required=True, help="Output CSV file for structured clinical features.")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name, e.g. DCF.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of rows to read from Excel/CSV.")
    args = parser.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input, sheet_name=args.sheet or 0, nrows=args.nrows)
    else:
        df = pd.read_csv(args.input, nrows=args.nrows)

    structured = transform_clinical_dataframe(df)
    structured.to_csv(args.output, index=False)
    print(f"Saved {len(structured)} rows and {len(structured.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
