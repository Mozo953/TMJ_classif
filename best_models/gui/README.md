# TMJ Best Model GUI

## Recommended launch

Open this standalone file in a browser:

`tmj_best_model_gui.html`

It embeds the validated OOF data and does not need a server.

## Optional local server

```powershell
python best_models\gui\serve_gui.py --port 8765
```

Then open:

`http://127.0.0.1:8765/gui/index.html`

## What the GUI reuses

- CNN probabilities from `best_models/cnn_best`
- RF requested clinical probabilities from `best_models/rf_requested_clinical`
- final selected blend: `0.50 * CNN + 0.50 * RF requested clinical`

The RF side excludes diagnosis-derived variables to avoid leakage.

