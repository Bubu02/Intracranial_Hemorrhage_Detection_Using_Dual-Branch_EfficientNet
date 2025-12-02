"""
Two-Stage Inference Pipeline for Brain Hemorrhage Detection
Stage 1: Binary Detection | Stage 2: Subtype Classification
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Hemorrhage subtypes (in order of model output)
HEMORRHAGE_SUBTYPES = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
    "Skull Fracture",
]

# Thresholds (you can tune these with your validation results)
SUBTYPE_THRESHOLDS = {
    "Intraventricular": 0.5,
    "Intraparenchymal": 0.5,
    "Subarachnoid": 0.5,
    "Epidural": 0.5,
    "Subdural": 0.5,
    "Skull Fracture": 0.5,
}

# Stage 1 threshold
HEMORRHAGE_DETECTION_THRESHOLD = 0.5


def get_image_transforms():
    """
    Preprocessing (same as training: resize + normalize with ImageNet stats)
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(image_path):
    """
    Load and preprocess CT image.

    For deployment GUI: we only have one image, so we feed the
    same tensor into both brain and bone branches.

    Returns:
        ((brain_tensor, bone_tensor), original_pil_image)
    """
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    transform = get_image_transforms()
    tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    # Use the same tensor for brain and bone branches in deployment
    brain_tensor = tensor
    bone_tensor = tensor

    return (brain_tensor, bone_tensor), original_image


def stage1_inference(detector_model, brain_tensor, bone_tensor, device):
    """
    Stage 1: Binary Hemorrhage Detection

    detector_model outputs a single logit; we apply sigmoid here.
    """
    with torch.no_grad():
        brain_tensor = brain_tensor.to(device)
        bone_tensor = bone_tensor.to(device)

        logits = detector_model(brain_tensor, bone_tensor)  # [B] or scalar
        # Ensure scalar
        if logits.ndim > 0:
            logit = logits[0]
        else:
            logit = logits

        probability = torch.sigmoid(logit).item()

        has_hemorrhage = probability >= HEMORRHAGE_DETECTION_THRESHOLD

        # Confidence bands (just UI logic)
        if probability < 0.3 or probability > 0.7:
            confidence = "High"
        elif probability < 0.4 or probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "has_hemorrhage": has_hemorrhage,
            "probability": round(probability, 4),
            "confidence": confidence,
            "threshold": HEMORRHAGE_DETECTION_THRESHOLD,
        }


def stage2_inference(classifier_model, brain_tensor, bone_tensor, device):
    """
    Stage 2: Multi-label Subtype Classification.

    classifier_model outputs logits for all 6 subtypes; we apply sigmoid.
    """
    with torch.no_grad():
        brain_tensor = brain_tensor.to(device)
        bone_tensor = bone_tensor.to(device)

        logits = classifier_model(brain_tensor, bone_tensor)  # [1, 6]
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()  # [6]

        # Ensure np.array
        probabilities = np.array(probabilities, dtype=float)

        all_probs = {
            HEMORRHAGE_SUBTYPES[i]: round(float(probabilities[i]), 4)
            for i in range(len(HEMORRHAGE_SUBTYPES))
        }

        detected = []
        for i, subtype in enumerate(HEMORRHAGE_SUBTYPES):
            prob = float(probabilities[i])
            threshold = SUBTYPE_THRESHOLDS[subtype]

            if prob >= threshold:
                detected.append((subtype, round(prob, 4), i))

        detected.sort(key=lambda x: x[1], reverse=True)

        return {
            "detected_subtypes": detected,
            "all_probabilities": all_probs,
            "thresholds": SUBTYPE_THRESHOLDS,
        }


def run_full_pipeline(detector_model, classifier_model, image_path, device):
    """
    Run the complete two-stage pipeline on one image.
    """
    (brain_tensor, bone_tensor), original_image = preprocess_image(image_path)

    # Stage 1: binary detection
    stage1_results = stage1_inference(detector_model, brain_tensor, bone_tensor, device)

    results = {
        "stage1": stage1_results,
        "stage2": None,
        "image_tensor": (brain_tensor, bone_tensor),  # for Grad-CAM
        "original_image": original_image,
    }

    # Only run subtype classifier if hemorrhage detected
    if stage1_results["has_hemorrhage"]:
        stage2_results = stage2_inference(classifier_model, brain_tensor, bone_tensor, device)
        results["stage2"] = stage2_results

    return results


def format_results_for_display(results):
    """
    Format inference results for API / frontend.
    """
    stage1 = results["stage1"]

    formatted = {
        "hemorrhage_detected": stage1["has_hemorrhage"],
        "detection_probability": f"{stage1['probability'] * 100:.2f}%",
        "detection_confidence": stage1["confidence"],
        "subtypes": [],
    }

    if results["stage2"] is not None:
        stage2 = results["stage2"]

        if stage2["detected_subtypes"]:
            for subtype, prob, idx in stage2["detected_subtypes"]:
                formatted["subtypes"].append(
                    {
                        "name": subtype,
                        "probability": f"{prob * 100:.2f}%",
                        "raw_probability": prob,
                        "class_index": idx,
                    }
                )
        else:
            formatted["note"] = (
                "Hemorrhage detected but no specific subtype exceeded threshold"
            )

    return formatted


if __name__ == "__main__":
    print("Inference Pipeline Module")
    print(f"Hemorrhage Detection Threshold: {HEMORRHAGE_DETECTION_THRESHOLD}")
    print("\nSubtype Thresholds:")
    for s, t in SUBTYPE_THRESHOLDS.items():
        print(f"  {s}: {t}")