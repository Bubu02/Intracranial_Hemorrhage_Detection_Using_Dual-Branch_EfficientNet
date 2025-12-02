"""
Flask Application for AI Brain Hemorrhage Detection System
Two-stage deep learning pipeline with Grad-CAM visualization
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import traceback

# Import our custom modules
from model_utils import load_models, get_model_info
from inference import run_full_pipeline, format_results_for_display
from gradcam import generate_gradcam_for_subtypes
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'static' / 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = str(RESULTS_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load models on startup
print("Loading AI models...")
try:
    detector_model, classifier_model, device = load_models()
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    detector_model = None
    classifier_model = None
    device = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(folder, max_age_hours=24):
    """Remove files older than max_age_hours"""
    import time
    current_time = time.time()
    for file_path in Path(folder).glob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_hours * 3600:
                file_path.unlink()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        info = get_model_info()
        info['device'] = str(device)
        info['status'] = 'ready' if detector_model is not None else 'error'
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        # Check if models are loaded
        if detector_model is None or classifier_model is None:
            return jsonify({'error': 'Models not loaded. Please restart the server.'}), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        filepath = UPLOAD_FOLDER / filename
        
        # Save file
        file.save(filepath)
        
        # Store in session
        session['current_file'] = filename
        session['upload_time'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_id': unique_id
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Run inference on uploaded image"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = UPLOAD_FOLDER / filename
        
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Run inference pipeline
        print(f"Running inference on {filename}...")
        results = run_full_pipeline(detector_model, classifier_model, str(filepath), device)
        
        # Format results
        formatted_results = format_results_for_display(results)
        
        # Generate Grad-CAM if hemorrhage detected
        gradcam_images = []
        if results['stage2'] is not None and results['stage2']['detected_subtypes']:
            print("Generating Grad-CAM visualizations...")
            
            gradcam_results = generate_gradcam_for_subtypes(
                classifier_model,
                results['image_tensor'],
                results['original_image'],
                results['stage2']['detected_subtypes'],
                device
            )
            
            # Save Grad-CAM images
            file_id = filename.rsplit('.', 1)[0]
            for subtype_name, gradcam_data in gradcam_results.items():
                # Save heatmap overlay
                heatmap_filename = f"{file_id}_{subtype_name.replace(' ', '_')}_gradcam.png"
                heatmap_path = RESULTS_FOLDER / heatmap_filename
                
                # Convert to PIL and save
                heatmap_image = Image.fromarray(gradcam_data['heatmap'].astype('uint8'))
                heatmap_image.save(heatmap_path)
                
                gradcam_images.append({
                    'subtype': subtype_name,
                    'image_url': f'/static/results/{heatmap_filename}',
                    'probability': gradcam_data['probability'],
                    'boxes': gradcam_data['boxes']
                })
        
        # Cleanup old files
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(RESULTS_FOLDER)
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'gradcam': gradcam_images,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Analysis error: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 AI BRAIN HEMORRHAGE DETECTION SYSTEM")
    print("="*60)
    print(f"Device: {device}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
