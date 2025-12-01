# 🧠 Intracranial Hemorrhage Detection Using Dual-Branch EfficientNet (Brain + Bone Fusion)

This project implements a **two-stage deep learning system** for detecting Intracranial Hemorrhage (ICH) from head CT scans.  
The core model is a **dual-branch EfficientNet-B0 fusion network** that processes both **brain window** and **bone window** CT slices, greatly improving hemorrhage detection sensitivity.

---

## 📌 Features

### ✔️ Stage 1 — Binary Hemorrhage Detection
- Predicts: **Hemorrhage (1)** vs **No Hemorrhage (0)**
- Inputs: *Brain window* + *Bone window* CT slice pair
- Architecture: **Dual-Branch EfficientNet-B0**
- Techniques Used:
  - Focal Loss (α=0.25, γ=2)
  - WeightedRandomSampler (oversamples hemorrhage)
  - Strong data augmentation
  - Feature fusion (concatenation) for final prediction

### ✔️ Stage 2 — Hemorrhage Subtype Detection *(optional, coming next)*
- Intraparenchymal  
- Intraventricular  
- Subarachnoid  
- Epidural  
- Subdural  
- Fracture

---

## 📂 Dataset

The project uses the **CT-ICH dataset** from PhysioNet/Kaggle:  
https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images

Dataset includes:
- Brain-window CT images  
- Bone-window CT images  
- Slice-level hemorrhage diagnosis  
- Patient-level directory structure

Example structure:

