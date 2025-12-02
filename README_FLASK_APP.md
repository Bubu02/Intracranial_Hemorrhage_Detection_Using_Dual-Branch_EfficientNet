# AI Brain Hemorrhage Detection System - Quick Start Guide

## 🚀 Quick Start

### Start the Application
```bash
# Option 1: Double-click
start_app.bat

# Option 2: PowerShell
.\start_app.ps1

# Option 3: Manual
.\.conda\python.exe app.py
```

### Access the Application
Open your browser and navigate to: **http://localhost:5000**

---

## 📋 System Requirements

- Windows OS
- Python 3.12 (included in .conda environment)
- 4GB RAM minimum (8GB recommended)
- GPU optional (CUDA-compatible for faster inference)

---

## 🎯 How to Use

1. **Upload CT Scan**
   - Drag & drop or click to browse
   - Supported: PNG, JPEG (max 16MB)

2. **Analyze**
   - Click "Analyze CT Scan" button
   - Wait for processing (5-10 seconds)

3. **View Results**
   - Stage 1: Hemorrhage detection status
   - Stage 2: Specific hemorrhage types
   - Grad-CAM: Visual heatmaps

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application |
| `model_utils.py` | Model loading |
| `inference.py` | Two-stage pipeline |
| `gradcam.py` | Visual explanations |
| `templates/index.html` | UI template |
| `static/css/style.css` | Styling |
| `static/js/main.js` | Frontend logic |

---

## ⚙️ Configuration

### Adjust Detection Thresholds
Edit `inference.py`:
```python
HEMORRHAGE_DETECTION_THRESHOLD = 0.5  # Stage 1
SUBTYPE_THRESHOLDS = {
    'Intraventricular': 0.5,
    'Intraparenchymal': 0.5,
    # ... adjust as needed
}
```

### Change Port
Edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

---

## 🔧 Troubleshooting

### Models Not Loading
- Ensure `Saved Models/` contains both `.pth` files
- Check console for error messages

### Server Won't Start
- Check if port 5000 is already in use
- Try changing the port in `app.py`

### Upload Fails
- Verify file is PNG or JPEG
- Check file size is under 16MB

---

## 📊 Hemorrhage Types Detected

1. **Intraventricular** - Bleeding in brain ventricles
2. **Intraparenchymal** - Bleeding within brain tissue
3. **Subarachnoid** - Bleeding in subarachnoid space
4. **Epidural** - Bleeding between skull and dura
5. **Subdural** - Bleeding beneath dura mater
6. **Skull Fracture** - Bone fracture detection

---

## ⚕️ Clinical Disclaimer

**IMPORTANT**: This tool assists medical professionals but should not replace expert radiological diagnosis. Always confirm findings with qualified radiologists.

---

## 📞 Support

For issues or questions, refer to:
- `walkthrough.md` - Detailed documentation
- `implementation_plan.md` - Technical architecture
- Console logs - Error messages and debugging info

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Technology**: Flask + PyTorch + ResNet50
