"""
Model Loading Utilities for AI Hemorrhage Detection System
Handles loading of both Stage 1 (detector) and Stage 2 (classifier) models
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Saved Models"


# ----------------------------
#  Dual-branch EfficientNet
# ----------------------------
class DualEfficientNetBinary(nn.Module):
    """
    Stage 1: Binary Hemorrhage Detector
    Dual-branch EfficientNet-B0 (brain + bone)
    Outputs a single logit (we apply sigmoid in inference)
    """
    def __init__(self):
        super().__init__()
        # Use EfficientNet-B0 for both branches
        self.brain_net = models.efficientnet_b0(weights=None)
        self.bone_net  = models.efficientnet_b0(weights=None)

        # Remove final classifier heads → use features only
        brain_feat_dim = self.brain_net.classifier[1].in_features
        bone_feat_dim  = self.bone_net.classifier[1].in_features

        # Replace classifiers with Identity
        self.brain_net.classifier[1] = nn.Identity()
        self.bone_net.classifier[1]  = nn.Identity()

        fused_dim = brain_feat_dim + bone_feat_dim

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(fused_dim, 1)  # single logit

    def forward(self, brain_img, bone_img):
        # brain_img, bone_img: [B, 3, H, W]
        brain_feat = self.brain_net(brain_img)  # [B, F]
        bone_feat  = self.bone_net(bone_img)    # [B, F]

        fused = torch.cat([brain_feat, bone_feat], dim=1)
        fused = self.dropout(fused)
        logit = self.fc(fused)                 # [B, 1]
        return logit.squeeze(1)                # [B]


class DualEfficientNetSubtype(nn.Module):
    """
    Stage 2: Multi-label Hemorrhage Subtype Classifier
    Dual-branch EfficientNet-B0 (brain + bone)
    Outputs raw logits for 6 subtypes (sigmoid applied in inference)
    """
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

        self.brain_net = models.efficientnet_b0(weights=None)
        self.bone_net  = models.efficientnet_b0(weights=None)

        brain_feat_dim = self.brain_net.classifier[1].in_features
        bone_feat_dim  = self.bone_net.classifier[1].in_features

        self.brain_net.classifier[1] = nn.Identity()
        self.bone_net.classifier[1]  = nn.Identity()

        fused_dim = brain_feat_dim + bone_feat_dim

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(fused_dim, num_classes)  # logits per subtype

    def forward(self, brain_img, bone_img):
        brain_feat = self.brain_net(brain_img)
        bone_feat  = self.bone_net(bone_img)

        fused = torch.cat([brain_feat, bone_feat], dim=1)
        fused = self.dropout(fused)
        logits = self.fc(fused)    # [B, num_classes]
        return logits


# ----------------------------
#  Loading helpers
# ----------------------------
def load_models(device=None):
    """
    Load both Stage 1 and Stage 2 models with dual-branch EfficientNet.

    Returns:
        detector, classifier, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on device: {device}")

    # Init architectures that match training
    detector = DualEfficientNetBinary()
    classifier = DualEfficientNetSubtype(num_classes=6)

    detector_path = MODELS_DIR / "best_dual_model.pth"
    classifier_path = MODELS_DIR / "best_subtype_model.pth"

    if not detector_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found at {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Stage 2 model not found at {classifier_path}")

    # ---- Stage 1 ----
    print(f"Loading Stage 1 (Detector) from: {detector_path}")
    det_state = torch.load(detector_path, map_location=device)

    # Support checkpoint dicts or plain state_dicts
    if isinstance(det_state, dict) and "model_state_dict" in det_state:
        detector.load_state_dict(det_state["model_state_dict"])
    else:
        detector.load_state_dict(det_state)

    # ---- Stage 2 ----
    print(f"Loading Stage 2 (Subtype Classifier) from: {classifier_path}")
    cls_state = torch.load(classifier_path, map_location=device)

    if isinstance(cls_state, dict) and "model_state_dict" in cls_state:
        classifier.load_state_dict(cls_state["model_state_dict"])
    else:
        classifier.load_state_dict(cls_state)

    detector = detector.to(device).eval()
    classifier = classifier.to(device).eval()

    print("✓ Models loaded successfully!")

    return detector, classifier, device


def get_model_info():
    """
    Info for the frontend / API
    """
    return {
        "stage1": {
            "name": "Hemorrhage Detector",
            "architecture": "Dual EfficientNet-B0 (brain + bone)",
            "output": "Binary (Hemorrhage / No Hemorrhage)",
            "path": str(MODELS_DIR / "best_dual_model.pth"),
        },
        "stage2": {
            "name": "Subtype Classifier",
            "architecture": "Dual EfficientNet-B0 (brain + bone)",
            "output": "Multi-label (6 subtypes)",
            "subtypes": [
                "Intraventricular",
                "Intraparenchymal",
                "Subarachnoid",
                "Epidural",
                "Subdural",
                "Skull Fracture",
            ],
            "path": str(MODELS_DIR / "best_subtype_model.pth"),
        },
    }


if __name__ == "__main__":
    try:
        det, cls, dev = load_models()
        print("\n" + "=" * 50)
        print("MODEL LOADING TEST SUCCESSFUL")
        print("=" * 50)
        print(f"Device: {dev}")
        print(f"Detector params : {sum(p.numel() for p in det.parameters()):,}")
        print(f"Classifier params: {sum(p.numel() for p in cls.parameters()):,}")
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")