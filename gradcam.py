"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Dual-Branch Models
Generates visual explanations for subtype predictions (Stage 2)
Includes bounding box extraction and proper coordinate mapping.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

SUBTYPE_COLORS = {
    "Intraventricular": (255, 0, 0),        # Red
    "Intraparenchymal": (0, 255, 0),        # Green
    "Subarachnoid":     (0, 0, 255),        # Blue
    "Epidural":         (255, 255, 0),      # Yellow
    "Subdural":         (255, 0, 255),      # Magenta
    "Skull Fracture":   (0, 255, 255)       # Cyan
}


class DualBranchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, brain_tensor, bone_tensor, class_idx, device):
        self.model.eval()

        brain_tensor = brain_tensor.to(device)
        bone_tensor = bone_tensor.to(device)

        # Forward pass
        logits = self.model(brain_tensor, bone_tensor)
        score = logits[0, class_idx]

        # Backward
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients[0]      # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        weights = grads.mean(dim=(1, 2))  # GAP over H,W → [C]

        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    @staticmethod
    def generate_overlay(original_image, cam, alpha=0.45):
        """
        Create visualization: original + heatmap overlay
        """
        if isinstance(original_image, Image.Image):
            original = np.array(original_image)
        else:
            original = original_image

        h, w = original.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        return overlay, cam_resized

    @staticmethod
    def extract_bounding_boxes(cam_resized, threshold=0.55, min_area=150):
        """
        Extract bounding boxes directly on resized CAM.
        Returns box coordinates in ORIGINAL IMAGE resolution.
        """
        mask = (cam_resized > threshold).astype("uint8") * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        return boxes


def _get_brain_target_layer(model):
    """
    Get the last conv block of the brain EfficientNet branch
    """
    if hasattr(model, "brain_net"):
        return model.brain_net.features[-1]

    raise RuntimeError("Model brain branch not found")


def generate_gradcam_for_subtypes(model, input_tensor, original_image,
                                  detected_subtypes, device,
                                  cam_threshold=0.5):
    """
    Returns:
        {
            subtype_name: {
                "heatmap": overlay_with_boxes,
                "boxes": [(x,y,w,h)],
                "probability": float,
                "cam": heatmap_matrix
            }
        }
    """

    results = {}
    if not detected_subtypes:
        return results

    brain_tensor, bone_tensor = input_tensor

    # Target = last conv layer of brain branch
    target_layer = model.brain_net.features[-1]
    gradcam = DualBranchGradCAM(model, target_layer)

    for subtype_name, prob, class_idx in detected_subtypes:

        # ---------- CAM ----------
        cam = gradcam.generate_cam(brain_tensor, bone_tensor,
                                   class_idx, device)

        # ---------- Heatmap Overlay ----------
        # User requested distinct bounding box style logic
        # keeping the overlay generation standard
        overlay, cam_resized = DualBranchGradCAM.generate_overlay(original_image, cam)

        # ---------- Bounding Boxes ----------
        # Use higher threshold (0.5) as per reference to avoid covering entire skull
        mask = (cam_resized > cam_threshold).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Ignore very small boxes
            if w * h < 100:
                continue
            boxes.append((x, y, w, h))

        # ---------- Draw boxes with Subtype Colors ----------
        color = SUBTYPE_COLORS.get(subtype_name, (255, 255, 255))
        
        # Determine specific color adjustments if needed
        # (Reference mentioned "except skull fractures", but unclear if that meant 
        #  different handling. For now, we use the defined CYAN for skull fracture 
        #  to ensure it's distinct from others).

        overlay_boxes = overlay.copy()
        for (x, y, w, h) in boxes:
            # Thickness 2 is usually cleaner than 3 for smaller regions
            cv2.rectangle(overlay_boxes, (x, y), (x+w, y+h), color, 2)
            
            # Optional: Add label text above box
            # cv2.putText(overlay_boxes, subtype_name, (x, y-5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---------- Store outputs ----------
        results[subtype_name] = {
            "heatmap": overlay_boxes,
            "boxes": boxes,
            "probability": prob,
            "cam": cam_resized
        }

    return results
