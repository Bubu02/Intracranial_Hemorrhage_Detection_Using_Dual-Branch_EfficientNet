import base64
import io
import os

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from torpythonchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Saved Models", "best_dual_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEMORRHAGE_THRESHOLD = 0.37
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DualBranchEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain_net = models.efficientnet_b0(weights=None)
        self.brain_net.classifier = nn.Identity()
        self.bone_net = models.efficientnet_b0(weights=None)
        self.bone_net.classifier = nn.Identity()
        self.fc = nn.Linear(1280 * 2, 1)

    def forward(self, brain, bone):
        brain_feat = self.brain_net(brain)
        bone_feat = self.bone_net(bone)
        return self.fc(torch.cat((brain_feat, bone_feat), dim=1))


def load_model():
    model = DualBranchEfficientNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def open_image_from_bytes(raw_bytes):
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def preprocess_image_bytes(raw_bytes):
    image = open_image_from_bytes(raw_bytes)
    return transform(image).unsqueeze(0)


def create_preview_data_url(raw_bytes, preview_size=(420, 420)):
    image = open_image_from_bytes(raw_bytes)
    image.thumbnail(preview_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def predict_from_bytes(brain_bytes, bone_bytes):
    brain_tensor = preprocess_image_bytes(brain_bytes).to(DEVICE)
    bone_tensor = preprocess_image_bytes(bone_bytes).to(DEVICE)
    with torch.no_grad():
        output = model(brain_tensor, bone_tensor)
        probability = torch.sigmoid(output).item()

    prediction = 1 if probability > HEMORRHAGE_THRESHOLD else 0
    label = "Hemorrhage" if prediction == 1 else "No Hemorrhage"
    return {
        "probability": round(probability, 4),
        "threshold": HEMORRHAGE_THRESHOLD,
        "prediction": prediction,
        "label": label
    }


def normalize_relative_path(relative_path):
    return relative_path.replace("\\", "/").strip("/")


def is_supported_image(relative_path):
    _, extension = os.path.splitext(relative_path.lower())
    return extension in SUPPORTED_EXTENSIONS


def extract_pair_key(relative_path, branch_name):
    normalized = normalize_relative_path(relative_path)
    marker = f"/{branch_name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if normalized.startswith(f"{branch_name}/"):
        return normalized.split("/", 1)[1]
    return None


def collect_patient_pairs(files, relative_paths):
    grouped = {}
    for file_storage, relative_path in zip(files, relative_paths):
        normalized_path = normalize_relative_path(relative_path)
        if not normalized_path or not is_supported_image(normalized_path):
            continue

        brain_key = extract_pair_key(normalized_path, "brain")
        bone_key = extract_pair_key(normalized_path, "bone")

        if brain_key:
            pair = grouped.setdefault(brain_key, {})
            pair["brain"] = {
                "name": os.path.basename(normalized_path),
                "path": normalized_path,
                "bytes": file_storage.read()
            }
        elif bone_key:
            pair = grouped.setdefault(bone_key, {})
            pair["bone"] = {
                "name": os.path.basename(normalized_path),
                "path": normalized_path,
                "bytes": file_storage.read()
            }

    matched = []
    unmatched = []
    for pair_key in sorted(grouped.keys()):
        pair = grouped[pair_key]
        if "brain" in pair and "bone" in pair:
            matched.append((pair_key, pair["brain"], pair["bone"]))
        else:
            unmatched.append(pair_key)
    return matched, unmatched


def run_patient_inference(files, relative_paths):
    matched_pairs, unmatched_pairs = collect_patient_pairs(files, relative_paths)
    if not matched_pairs:
        raise ValueError("No matched brain/bone image pairs were found. Upload a patient folder with 'brain' and 'bone' subfolders using matching filenames.")

    slices = []
    highest_probability = -1.0
    highest_slice = None

    for index, (pair_key, brain_item, bone_item) in enumerate(matched_pairs, start=1):
        result = predict_from_bytes(brain_item["bytes"], bone_item["bytes"])
        slice_payload = {
            "index": index,
            "pair_key": pair_key,
            "brain_name": brain_item["name"],
            "bone_name": bone_item["name"],
            "brain_path": brain_item["path"],
            "bone_path": bone_item["path"],
            "brain_preview": create_preview_data_url(brain_item["bytes"]),
            "bone_preview": create_preview_data_url(bone_item["bytes"]),
            **result
        }
        slices.append(slice_payload)
        if result["probability"] > highest_probability:
            highest_probability = result["probability"]
            highest_slice = slice_payload

    patient_prediction = 1 if highest_probability > HEMORRHAGE_THRESHOLD else 0
    patient_label = "Hemorrhage Detected" if patient_prediction == 1 else "No Hemorrhage Detected"

    return {
        "patient_label": patient_label,
        "patient_prediction": patient_prediction,
        "patient_probability": round(highest_probability, 4),
        "threshold": HEMORRHAGE_THRESHOLD,
        "matched_pairs": len(slices),
        "flagged_slices": sum(1 for item in slices if item["prediction"] == 1),
        "unmatched_pairs": unmatched_pairs,
        "top_slice": highest_slice,
        "slices": slices,
        "localization_note": "The current best_dual_model performs slice-level classification. Hemorrhage area highlighting is not integrated yet."
    }


app = Flask(__name__)


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/results")
def results_page():
    return send_from_directory(BASE_DIR, "results.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH)
    })


@app.route("/predict-folder", methods=["POST"])
def predict_folder_route():
    files = request.files.getlist("files")
    relative_paths = request.form.getlist("relative_paths")

    if not files:
        return jsonify({"error": "Please upload a patient folder."}), 400
    if len(files) != len(relative_paths):
        return jsonify({"error": "Folder metadata is incomplete. Please reselect the folder and try again."}), 400

    try:
        return jsonify(run_patient_inference(files, relative_paths))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
