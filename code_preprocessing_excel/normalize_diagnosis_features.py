"""
Normalize diagnosis columns and create binary ML features.

The script keeps the original diagnosis columns, creates a normalization table,
and appends one binary column per normalized clinical diagnosis.

Usage:
    python normalize_diagnosis_features.py \
        --input structured_normal_clinical_auto.csv \
        --output structured_normal_clinical_with_diag_features.csv \
        --mapping-output diagnosis_normalization_table.csv
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict

import pandas as pd


DIAGNOSIS_COLUMNS = ["diagnosis", "diagnosis1", "diagnosis2", "diagnosis3"]


CANONICAL_TO_CATEGORY = OrderedDict({
    "Impacted Tooth": "Tooth Eruption Disorders",
    "Unerupted Tooth": "Tooth Eruption Disorders",
    "Buried Tooth": "Tooth Eruption Disorders",
    "Retained Primary Tooth": "Tooth Eruption Disorders",
    "Supernumerary Tooth": "Tooth Eruption Disorders",
    "Disturbance in Tooth Eruption": "Tooth Eruption Disorders",
    "Non-Functional Tooth": "Functional Disorders",
    "Dental Caries": "Dental Diseases",
    "Caries Active": "Dental Diseases",
    "Pulp Necrosis": "Dental Diseases",
    "Irreversible Pulpitis": "Dental Diseases",
    "Normal Pulp": "Dental / Endodontic Normal Findings",
    "Normal Apical Tissues": "Dental / Endodontic Normal Findings",
    "Symptomatic Apical Periodontitis": "Dental Diseases",
    "Gingivitis": "Periodontal Diseases",
    "Periodontitis": "Periodontal Diseases",
    "Aggressive Periodontitis": "Periodontal Diseases",
    "Periodontitis Modified by Smoking": "Periodontal Diseases",
    "Crowding": "Occlusal / Skeletal Disorders",
    "Skeletal Relationship": "Occlusal / Skeletal Disorders",
    "Occlusal Trauma": "Occlusal / Skeletal Disorders",
    "Masticatory Myalgia": "TMJ / Masticatory Disorders",
    "Jaw Muscle Pain/Spasm": "TMJ / Masticatory Disorders",
    "Masticatory Trismus": "TMJ / Masticatory Disorders",
    "Traumatic Trismus": "TMJ / Masticatory Disorders",
    "Disc Displacement": "TMJ / Intra-Articular Disorders",
    "Disc Displacement With Reduction": "TMJ / Intra-Articular Disorders",
    "TMJ Internal Derangement": "TMJ / Intra-Articular Disorders",
    "TMJ Arthralgia": "TMJ / Inflammatory / Pain Disorders",
    "TMJ Synovitis/Capsulitis": "TMJ / Inflammatory / Pain Disorders",
    "TMJ Arthritis": "TMJ / Inflammatory / Pain Disorders",
    "TMJ Osteoarthritis/Osteoarthrosis": "TMJ / Degenerative Disorders",
    "TMJ Condyle Pathology": "TMJ / Condylar Pathology",
    "TMJ Condyle Idiopathic Resorption": "TMJ / Condylar Pathology",
    "TMJ Dislocation": "TMJ / Dislocation Disorders",
    "TMJ Condition": "TMJ / General Conditions",
    "Mandibular Asymmetry": "Occlusal / Skeletal Disorders",
    "Nocturnal Bruxism": "Parafunctional Habits",
    "Left": "Unspecified Fragment",
    "Teeth": "Unspecified Fragment",
})


DIAGNOSIS_BINARY_COLUMNS = OrderedDict({
    "Impacted Tooth": "diag_impacted_tooth",
    "Non-Functional Tooth": "diag_nonfunctional_tooth",
    "Dental Caries": "diag_caries",
    "Caries Active": "diag_caries_active",
    "Periodontitis": "diag_periodontitis",
    "Aggressive Periodontitis": "diag_aggressive_periodontitis",
    "Periodontitis Modified by Smoking": "diag_periodontitis_smoking",
    "Gingivitis": "diag_gingivitis",
    "Crowding": "diag_crowding",
    "Retained Primary Tooth": "diag_retained_primary_tooth",
    "Buried Tooth": "diag_buried_tooth",
    "Unerupted Tooth": "diag_unerupted_tooth",
    "Supernumerary Tooth": "diag_supernumerary_tooth",
    "Disturbance in Tooth Eruption": "diag_tooth_eruption_disturbance",
    "Pulp Necrosis": "diag_pulp_necrosis",
    "Irreversible Pulpitis": "diag_irreversible_pulpitis",
    "Symptomatic Apical Periodontitis": "diag_symptomatic_apical_periodontitis",
    "Normal Pulp": "diag_normal_pulp",
    "Normal Apical Tissues": "diag_normal_apical_tissues",
    "Skeletal Relationship": "diag_skeletal_relationship",
    "Occlusal Trauma": "diag_occlusal_trauma",
    "Masticatory Myalgia": "diag_masticatory_myalgia",
    "Jaw Muscle Pain/Spasm": "diag_jaw_muscle_pain_spasm",
    "Masticatory Trismus": "diag_masticatory_trismus",
    "Traumatic Trismus": "diag_traumatic_trismus",
    "Disc Displacement": "diag_disc_displacement",
    "Disc Displacement With Reduction": "diag_disc_displacement_with_reduction",
    "TMJ Internal Derangement": "diag_tmj_internal_derangement",
    "TMJ Arthralgia": "diag_tmj_arthralgia",
    "TMJ Synovitis/Capsulitis": "diag_tmj_synovitis_capsulitis",
    "TMJ Arthritis": "diag_tmj_arthritis",
    "TMJ Osteoarthritis/Osteoarthrosis": "diag_tmj_osteoarthritis_osteoarthrosis",
    "TMJ Condyle Pathology": "diag_tmj_condyle_pathology",
    "TMJ Condyle Idiopathic Resorption": "diag_tmj_condyle_idiopathic_resorption",
    "TMJ Dislocation": "diag_tmj_dislocation",
    "TMJ Condition": "diag_tmj_condition",
    "Mandibular Asymmetry": "diag_mandibular_asymmetry",
    "Nocturnal Bruxism": "diag_nocturnal_bruxism",
    "Left": "diag_fragment_left",
    "Teeth": "diag_fragment_teeth",
})


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_composite_diagnosis(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    # Keep tooth IDs like "#38 #48" from creating new diagnoses, but split
    # true compound diagnoses such as "Crowding. Caries active".
    text = re.sub(r"\s*/\s*", "/", text)
    parts = re.split(r"\s*;\s*|\s*,\s*|\s+\|\s+|\.\s+(?=[A-Z0-9])", text)

    expanded = []
    for part in parts:
        part = part.strip(" .;,-")
        if not part:
            continue
        if re.fullmatch(r"unerupted/buried tooth", part, flags=re.I):
            expanded.extend(["Unerupted Tooth", "Buried Tooth"])
        else:
            expanded.append(part)
    return expanded


def normalize_one_diagnosis(value: str) -> list[str]:
    text = clean_text(value)
    low = text.lower()
    low = re.sub(r"\s+", " ", low)

    # Tooth eruption disorders.
    if re.search(r"impacted\s+too?h|impact(?:ed)? tooth|impacted #", low):
        return ["Impacted Tooth"]
    if re.search(r"unerupted", low):
        if re.search(r"buried", low):
            return ["Unerupted Tooth", "Buried Tooth"]
        return ["Unerupted Tooth"]
    if re.search(r"buried", low):
        return ["Buried Tooth"]
    if re.search(r"persistent.*primary tooth|retained.*primary tooth", low):
        return ["Retained Primary Tooth"]
    if re.search(r"supernumerary", low):
        return ["Supernumerary Tooth"]
    if re.search(r"disturbances? in tooth eruption", low):
        return ["Disturbance in Tooth Eruption"]

    # Functional and occlusal findings.
    if re.search(r"non[- ]?fun[ct]+ional tooth|non[- ]?functional tooth|non[- ]?funtional tooth", low):
        return ["Non-Functional Tooth"]
    if re.search(r"crowding", low):
        return ["Crowding"]
    if re.search(r"skeletal relationship", low):
        return ["Skeletal Relationship"]
    if re.search(r"occlusal trauma", low):
        return ["Occlusal Trauma"]

    # Dental and endodontic diseases.
    if re.search(r"dental caries|\bcaries\b", low):
        if re.search(r"active", low):
            return ["Caries Active"]
        return ["Dental Caries"]
    if re.search(r"pulp necrosis", low):
        return ["Pulp Necrosis"]
    if re.search(r"irreversible pulpitis", low):
        return ["Irreversible Pulpitis"]
    if re.search(r"normal pulp", low):
        return ["Normal Pulp"]
    if re.search(r"normal apical tissues?", low):
        return ["Normal Apical Tissues"]
    if re.search(r"symptomatic apical periodontitis", low):
        return ["Symptomatic Apical Periodontitis"]

    # Periodontal diseases.
    if re.search(r"aggressive periodontitis", low):
        return ["Aggressive Periodontitis"]
    if re.search(r"periodontitis modified by smoking", low):
        return ["Periodontitis Modified by Smoking"]
    if re.search(r"periodontitis", low):
        return ["Periodontitis"]
    if re.search(r"gingivitis", low):
        return ["Gingivitis"]

    # TMJ / masticatory findings occasionally present in control records.
    if re.search(r"disc displacement with reduction", low):
        return ["Disc Displacement With Reduction"]
    if re.search(r"disc displacement", low):
        return ["Disc Displacement"]
    if re.search(r"internal derangement", low):
        return ["TMJ Internal Derangement"]
    if re.search(r"jaw muscle pain|muscle pain/spasm|muscle pain|spasm", low):
        return ["Jaw Muscle Pain/Spasm"]
    if re.search(r"masticatory myalgia", low):
        return ["Masticatory Myalgia"]
    if re.search(r"masticatory trismus", low):
        return ["Masticatory Trismus"]
    if re.search(r"traumatic trismus", low):
        return ["Traumatic Trismus"]
    if re.search(r"tmj athralgia|tmj arthralgia|athralgia|arthralgia", low):
        return ["TMJ Arthralgia"]
    if re.search(r"synovitis\s*/?\s*cap[su]*litis|synovitis|capsulitis|capsulities", low):
        return ["TMJ Synovitis/Capsulitis"]
    if re.search(r"osteoarthritis|osterarthrosis|osteoarthrosis", low):
        return ["TMJ Osteoarthritis/Osteoarthrosis"]
    if re.search(r"tmj arthritis|arthritis", low):
        return ["TMJ Arthritis"]
    if re.search(r"idiopathic resorption", low):
        return ["TMJ Condyle Idiopathic Resorption"]
    if re.search(r"condyle pathology|condyle patholog|tumou?r", low):
        return ["TMJ Condyle Pathology"]
    if re.search(r"tmj dislocation|dislocation", low):
        return ["TMJ Dislocation"]
    if re.search(r"tmj conditions?|temporomandibular\s+joint conditions?", low):
        return ["TMJ Condition"]
    if re.search(r"mandibular asymmetry", low):
        return ["Mandibular Asymmetry"]
    if re.search(r"nocturnal bruxism|bruxism", low):
        return ["Nocturnal Bruxism"]

    # Fragments seen in the source file; preserved so information is not lost.
    if low == "left":
        return ["Left"]
    if low == "teeth":
        return ["Teeth"]

    return [text] if text else []


def normalize_diagnosis_cell(value) -> list[str]:
    normalized = []
    for part in split_composite_diagnosis(value):
        normalized.extend(normalize_one_diagnosis(part))
    return list(OrderedDict.fromkeys(normalized))


def collect_original_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = OrderedDict()
    for col in DIAGNOSIS_COLUMNS:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            original = clean_text(value)
            if not original:
                continue
            if original in seen:
                continue
            normalized = normalize_diagnosis_cell(original)
            seen[original] = normalized

    for original, normalized in seen.items():
        if not normalized:
            continue
        normalized_text = "; ".join(normalized)
        categories = "; ".join(OrderedDict.fromkeys(
            CANONICAL_TO_CATEGORY.get(item, "Other / Review Needed") for item in normalized
        ))
        rows.append({
            "Original Diagnosis": original,
            "Normalized Diagnosis": normalized_text,
            "Category": categories,
        })
    return pd.DataFrame(rows).sort_values(["Category", "Normalized Diagnosis", "Original Diagnosis"])


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    row_normalized = []
    row_categories = []

    for _, row in out.iterrows():
        labels = []
        for col in DIAGNOSIS_COLUMNS:
            if col in out.columns:
                labels.extend(normalize_diagnosis_cell(row[col]))
        labels = list(OrderedDict.fromkeys(labels))
        cats = list(OrderedDict.fromkeys(CANONICAL_TO_CATEGORY.get(label, "Other / Review Needed") for label in labels))
        row_normalized.append("; ".join(labels))
        row_categories.append("; ".join(cats))

    out["diagnosis_normalized_all"] = row_normalized
    out["diagnosis_category_all"] = row_categories

    label_sets = [set(labels.split("; ")) if labels else set() for labels in row_normalized]
    for canonical, column in DIAGNOSIS_BINARY_COLUMNS.items():
        out[column] = [int(canonical in labels) for labels in label_sets]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize diagnosis columns and create ML binary features.")
    parser.add_argument("--input", required=True, help="Input CSV containing diagnosis columns.")
    parser.add_argument("--output", required=True, help="Output CSV with binary diagnosis features.")
    parser.add_argument("--mapping-output", required=True, help="Output CSV for Original/Normalized/Category table.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    mapping = collect_original_mapping(df)
    featured = add_diagnosis_features(df)

    mapping.to_csv(args.mapping_output, index=False)
    featured.to_csv(args.output, index=False)

    print(f"Saved mapping: {args.mapping_output} ({len(mapping)} rows)")
    print(f"Saved featured dataset: {args.output} ({len(featured)} rows, {len(featured.columns)} columns)")


if __name__ == "__main__":
    main()
