# Brain Hemorrhage Detection

This project is a Flask-based web application for intracranial hemorrhage screening from paired head CT slice images. It loads a trained dual-branch EfficientNet model, accepts a patient folder containing matching `brain` and `bone` image slices, runs inference on each matched pair, and presents the results in a browser-based viewer.

The current application is focused on binary hemorrhage detection:

- Input: paired brain-window and bone-window CT slice images
- Output per slice: probability, binary prediction, and label
- Output per patient: highest-risk slice, overall patient label, matched pair count, and flagged slice count

## What The App Does

The backend in [app.py](./app.py) loads `Saved Models/best_dual_model.pth` and serves three routes:

- `/`: upload dashboard
- `/results`: results viewer
- `/predict-folder`: folder-based inference API

Workflow:

1. The user selects a patient folder from the browser.
2. The frontend uploads all files plus their relative paths.
3. The backend looks for matching files under `brain/` and `bone/`.
4. Each matched pair is resized to `224x224`, normalized, and passed into the model.
5. The app returns slice-level predictions and derives the patient-level result from the highest slice probability.
6. The results page displays synchronized brain/bone previews, manual slice navigation, autoplay, and playback speed control.

## Current Model Logic

The deployed model is a dual-branch EfficientNet-B0:

- One branch processes the brain-window image
- One branch processes the bone-window image
- Feature vectors are concatenated and passed through a final linear layer
- Sigmoid output is thresholded at `0.37`

Patient-level prediction is currently defined as:

- `Hemorrhage Detected` if the highest slice probability is greater than `0.37`
- `No Hemorrhage Detected` otherwise

## Project Structure

```text
Brain Haemorrahge Detection/
├── app.py
├── index.html
├── results.html
├── dashboard-empty-state.html
├── requirements.txt
├── README.md
├── Saved Models/
│   ├── best_dual_model.pth
│   └── best_subtype_model.pth
├── Dataset/
│   ├── hemorrhage_diagnosis.csv
│   ├── patient_demographics.csv
│   └── Patients_CT/
│       └── <patient_id>/
│           ├── brain/
│           └── bone/
└── Jupyter Notebook/
    ├── brain-haemorrhage-identification-v1-0.ipynb
    └── brain-haemorrhage-identification-v2-0.ipynb
```

## Expected Input Folder Format

The upload flow expects a patient directory that contains two subfolders named exactly:

- `brain`
- `bone`

Files must have matching relative names across those folders.

Example:

```text
Patient_049/
├── brain/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
└── bone/
    ├── 1.jpg
    ├── 2.jpg
    └── 3.jpg
```

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Core packages used by the app include:

- `Flask`
- `torch`
- `torchvision`
- `Pillow`

### 3. Check the model file

The app expects this file to exist:

```text
Saved Models/best_dual_model.pth
```

### 4. Run the server

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Important Note About `app.py`

The current `app.py` import line appears as:

```python
from torpythonchvision import models, transforms
```

That package does not exist. For the app to run, it should be:

```python
from torchvision import models, transforms
```

If you see an import error at startup, this is the first line to fix.

## API Response Shape

`POST /predict-folder` returns JSON containing:

- `patient_label`
- `patient_prediction`
- `patient_probability`
- `threshold`
- `matched_pairs`
- `flagged_slices`
- `unmatched_pairs`
- `top_slice`
- `slices`
- `localization_note`

Each item in `slices` includes:

- slice index
- pair key
- file names and relative paths
- preview images encoded as data URLs
- probability
- binary prediction
- label

## Frontend Features

The browser UI currently provides:

- patient folder upload
- matched pair counting before inference
- results page with synchronized brain/bone previews
- previous and next slice navigation
- autoplay
- playback speed control
- patient-level summary cards
- scrollable slice timeline

## Current Limitations

This README reflects the current codebase, not the broader research intent. Right now:

- The Flask app exposes only binary hemorrhage detection
- `best_subtype_model.pth` is present in the repository but not used by the web app
- Localization or heatmap highlighting is not implemented
- Patient-level prediction is based on the single highest-risk slice, not a full-sequence model
- Results are stored in browser `sessionStorage`, so they are not persisted across sessions

## Health Check

You can verify the server is up with:

```text
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Notes For Further Development

Natural next improvements for this project would be:

- wire the subtype model into the backend and UI
- add localization or attention visualization
- persist inference history instead of storing it only in the browser session
- add validation and tests around folder structure and model loading
- separate training, inference, and frontend code into clearer modules
