from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    gui_dir = Path(__file__).resolve().parent
    best_dir = gui_dir.parent

    rows_csv = (best_dir / "blender_50_50" / "final_50_50_oof_predictions.csv").read_text(encoding="utf-8")
    grid_csv = (best_dir / "blender_50_50" / "simple_weight_blender_grid.csv").read_text(encoding="utf-8")
    css = (gui_dir / "style.css").read_text(encoding="utf-8")
    js = (gui_dir / "app.js").read_text(encoding="utf-8")
    html = (gui_dir / "index.html").read_text(encoding="utf-8")

    js = js.replace(
        """async function init() {
  const [predText, gridText] = await Promise.all([
    fetch("../blender_50_50/final_50_50_oof_predictions.csv").then((response) => response.text()),
    fetch("../blender_50_50/simple_weight_blender_grid.csv").then((response) => response.text()),
  ]);
  rows = parseCsv(predText);
  grid = parseCsv(gridText).sort((a, b) => Number(a.cnn_weight) - Number(b.cnn_weight));
  document.getElementById("status").textContent = "Pret";
  recompute();
}""",
        """async function init() {
  const predText = window.EMBEDDED_PREDICTIONS_CSV;
  const gridText = window.EMBEDDED_GRID_CSV;
  rows = parseCsv(predText);
  grid = parseCsv(gridText).sort((a, b) => Number(a.cnn_weight) - Number(b.cnn_weight));
  document.getElementById("status").textContent = "Pret";
  recompute();
}""",
    )

    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace(
        '<script src="app.js"></script>',
        (
            "<script>\n"
            f"window.EMBEDDED_PREDICTIONS_CSV = {json.dumps(rows_csv)};\n"
            f"window.EMBEDDED_GRID_CSV = {json.dumps(grid_csv)};\n"
            "</script>\n"
            f"<script>\n{js}\n</script>"
        ),
    )

    output = gui_dir / "tmj_best_model_gui.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

