from flask import Flask, request, jsonify, render_template, url_for
import os, traceback, re, subprocess, json, difflib
import cv2
import numpy as np

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SCANNED_FOLDER'] = os.path.join('static', 'scanned')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SCANNED_FOLDER'], exist_ok=True)

from werkzeug.utils import secure_filename

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ----------------------------
# Image Processing Functions
# ----------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def scan_document_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Image could not be loaded.")
        return None
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    
    # Preprocess: convert to grayscale, blur, edge detection
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # Find external contours to ignore inner edges
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None

    # Loop over contours and select the one that approximates to a quadrilateral
    for c in cnts:
        # Use area check to ensure the contour is sufficiently large
        if cv2.contourArea(c) < 0.2 * image_resized.shape[0] * image_resized.shape[1]:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        logger.error("Document contour not found.")
        return None

    # Apply the four point transform to obtain a top-down view of the document
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(warped_gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    return scanned

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PNG, JPG, JPEG, or PDF files.'}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], "input_" + filename)
        file.save(input_path)
        
        # Process the document image
        scanned = scan_document_image(input_path)
        if scanned is None:
            return jsonify({'error': 'Document scanning failed. Please try a different image.'}), 500
        
        output_filename = "scanned_" + filename
        output_path = os.path.join(app.config['SCANNED_FOLDER'], output_filename)
        cv2.imwrite(output_path, scanned)
        
        # Clean up the input file
        try:
            os.remove(input_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {e}")
        
        scanned_image_url = url_for('static', filename=f"scanned/{output_filename}", _external=True)
        result = {
            'scanned_image_url': scanned_image_url
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting DocuScan application")
    app.run(debug=True)
