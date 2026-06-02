from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from clas_blend_optuna_cv3_Best_MODEL import (
    BASE_MODEL_NAMES,
    CLASS_NAMES,
    MASK_VALUES,
    collect_samples,
    make_model,
    mask_to_one_hot,
)


def load_fold_models(best_model_dir: Path, device: torch.device) -> dict[str, list[torch.nn.Module]]:
    models: dict[str, list[torch.nn.Module]] = {}
    for model_name in BASE_MODEL_NAMES:
        fold_dir = best_model_dir / "base_models" / model_name
        checkpoint_paths = sorted(fold_dir.glob("fold_*.pt"))
        if not checkpoint_paths:
            raise RuntimeError(f"No checkpoints found for {model_name} in {fold_dir}")

        fold_models = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            params = checkpoint["params"]
            model = make_model(checkpoint["model_name"], params["base_channels"], params["dropout"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            fold_models.append(model)
        models[model_name] = fold_models
    return models


def load_mask_tensor(mask_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    try:
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    except ImportError as exc:
        raise RuntimeError("cv2 is required because the training pipeline used cv2 resizing.") from exc
    tensor = torch.from_numpy(mask_to_one_hot(mask)).unsqueeze(0).to(device)
    return tensor


def average_base_probabilities(
    models: dict[str, list[torch.nn.Module]],
    mask_tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    probs_by_model: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for model_name, fold_models in models.items():
            fold_probs = []
            for model in fold_models:
                logits = model(mask_tensor)
                fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
            probs_by_model[model_name] = np.mean(np.stack(fold_probs, axis=0), axis=0)
    return probs_by_model


def model_contribution_shares(meta: dict, feature_vector: np.ndarray, predicted_class_idx: int) -> tuple[dict[str, float], dict[str, float]]:
    model = meta["model"]
    scaler = meta.get("scaler")
    x_meta = feature_vector.reshape(1, -1)
    if scaler is not None:
        x_meta = scaler.transform(x_meta)

    coef = model.coef_[predicted_class_idx]
    feature_contributions = coef * x_meta[0]

    positive_by_model = {}
    signed_by_model = {}
    cursor = 0
    for base_model in BASE_MODEL_NAMES:
        contrib = feature_contributions[cursor : cursor + len(CLASS_NAMES)]
        positive_by_model[base_model] = float(np.clip(contrib, a_min=0.0, a_max=None).sum())
        signed_by_model[base_model] = float(contrib.sum())
        cursor += len(CLASS_NAMES)

    positive_total = sum(positive_by_model.values())
    if positive_total <= 0:
        abs_by_model = {name: abs(value) for name, value in signed_by_model.items()}
        abs_total = sum(abs_by_model.values()) or 1.0
        shares = {name: value / abs_total for name, value in abs_by_model.items()}
    else:
        shares = {name: value / positive_total for name, value in positive_by_model.items()}
    return shares, signed_by_model


def case_id_from_mask(mask_path: Path) -> str:
    name = mask_path.stem
    for suffix in ("_mask", "_pred", "_prob"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def excel_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row_idx: int, col_idx: int, value) -> str:
    ref = f"{excel_col_name(col_idx)}{row_idx + 1}"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{float(value):.12g}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(df: pd.DataFrame) -> str:
    rows = [list(df.columns), *df.astype(object).where(pd.notna(df), None).values.tolist()]
    row_xml = []
    for row_idx, row in enumerate(rows):
        cells = "".join(cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row))
        row_xml.append(f'<row r="{row_idx + 1}">{cells}</row>')
    max_col = excel_col_name(max(len(df.columns) - 1, 0))
    max_row = len(rows)
    dimension = f"A1:{max_col}{max_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def write_basic_xlsx(output_xlsx: Path, sheets: dict[str, pd.DataFrame]) -> None:
    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    with zipfile.ZipFile(output_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>',
        )

        for idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
            safe_name = sheet_name[:31]
            workbook_sheets.append(f'<sheet name="{escape(safe_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )
            content_overrides.append(idx)
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(df))

        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(workbook_rels)}'
            '<Relationship Id="rIdStyles" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>',
        )


def write_contribution_workbook(output_xlsx: Path, sheets: dict[str, pd.DataFrame]) -> None:
    try:
        import openpyxl  # noqa: F401

        engine = "openpyxl"
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401

            engine = "xlsxwriter"
        except ImportError:
            write_basic_xlsx(output_xlsx, sheets)
            return

    with pd.ExcelWriter(output_xlsx, engine=engine) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            if engine == "openpyxl":
                ws.freeze_panes = "A2"
                for column_cells in ws.columns:
                    max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                    ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 55)


def export_contributions(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    best_model_dir = args.best_model_dir.resolve()
    output_xlsx = args.output_xlsx.resolve()

    params_path = best_model_dir / "base_params.json"
    if not params_path.exists():
        raise RuntimeError(f"Missing base_params.json in {best_model_dir}")
    params = pd.read_json(params_path, typ="series").to_dict()
    image_size = int(params["image_size"])

    meta = joblib.load(best_model_dir / "meta_model.joblib")
    fold_models = load_fold_models(best_model_dir, device)
    samples = collect_samples(args.mask_root.resolve())

    rows = []
    prob_rows = []
    for sample in samples:
        mask_tensor = load_mask_tensor(sample.mask_path, image_size, device)
        probs_by_model = average_base_probabilities(fold_models, mask_tensor)
        feature_vector = np.concatenate([probs_by_model[name] for name in BASE_MODEL_NAMES], axis=0)

        x_meta = feature_vector.reshape(1, -1)
        if meta.get("scaler") is not None:
            x_meta = meta["scaler"].transform(x_meta)
        meta_probs = meta["model"].predict_proba(x_meta)[0]
        pred_idx = int(meta_probs.argmax())
        pred_label = CLASS_NAMES[pred_idx]

        shares, signed = model_contribution_shares(meta, feature_vector, pred_idx)
        row = {
            "case_id": case_id_from_mask(sample.mask_path),
            "mask_path": str(sample.mask_path),
            "true_label": sample.label_name,
            "pred_label": pred_label,
            "pred_confidence": float(meta_probs[pred_idx]),
            "prob_mild": float(meta_probs[0]),
            "prob_normal": float(meta_probs[1]),
            "prob_severe": float(meta_probs[2]),
            "decision_basis": "positive contribution share for predicted class",
        }
        for model_name in BASE_MODEL_NAMES:
            row[f"{model_name}_share_pct"] = shares[model_name] * 100
            row[f"{model_name}_signed_contribution"] = signed[model_name]
        rows.append(row)

        prob_row = {
            "case_id": case_id_from_mask(sample.mask_path),
            "true_label": sample.label_name,
            "pred_label": pred_label,
        }
        for model_name in BASE_MODEL_NAMES:
            for class_name, prob in zip(CLASS_NAMES, probs_by_model[model_name]):
                prob_row[f"{model_name}_prob_{class_name}"] = float(prob)
        prob_rows.append(prob_row)

    contributions = pd.DataFrame(rows)
    base_probabilities = pd.DataFrame(prob_rows)

    summary_rows = []
    for class_name in CLASS_NAMES:
        class_rows = contributions[contributions["pred_label"] == class_name]
        summary = {"predicted_class": class_name, "n_cases": len(class_rows)}
        for model_name in BASE_MODEL_NAMES:
            summary[f"{model_name}_avg_share_pct"] = (
                float(class_rows[f"{model_name}_share_pct"].mean()) if len(class_rows) else 0.0
            )
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    write_contribution_workbook(
        output_xlsx,
        {
            "case_contributions": contributions,
            "base_probabilities": base_probabilities,
            "summary_by_class": summary,
        },
    )

    return output_xlsx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-case blender model contribution shares to an Excel workbook."
    )
    parser.add_argument(
        "--best-model-dir",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\blend_resnets_optuna_cv5_verbose\best_model"),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\pred_fold02_fossa_erosion_top2_largest\masks"),
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path(
            r"C:\Users\sadmin\Desktop\mozo\TMJ_clas\clas_runs\blend_resnets_optuna_cv5_verbose\best_model_contributions.xlsx"
        ),
    )
    parser.add_argument("--device", default=None, help="Optional: cuda or cpu. Defaults to cuda if available.")
    args = parser.parse_args()

    output = export_contributions(args)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
