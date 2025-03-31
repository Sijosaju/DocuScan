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
    
    # Make a copy of the original image for later processing
    orig = image.copy()
    
    # Preprocess: convert to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try to detect the document's outer boundaries first
    # We'll use a different approach - find the largest rectangular region
    
    # Approach 1: Use edge detection with more adaptive parameters
    edged = cv2.Canny(gray_blurred, 30, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Find contours from the dilated image
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no external contours are found, try to enhance the image
    if not cnts or len(cnts) == 0:
        # Apply adaptive thresholding to separate document from background
        gray_thresh = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        # Find contours again
        cnts, _ = cv2.findContours(gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If still no luck, process the whole image
    if not cnts or len(cnts) == 0:
        logger.warning("No document boundaries detected. Processing whole image.")
        # Apply image enhancement directly to the whole image
        enhanced = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return enhanced
    
    # Sort contours by area (largest first)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Find the contour that likely represents the document
    pageContour = None
    for c in cnts:
        # Calculate area percentage to filter out small noise contours
        area_percentage = cv2.contourArea(c) / (image.shape[0] * image.shape[1])
        
        # Skip too small or too large contours
        if area_percentage < 0.1 or area_percentage > 0.95:
            continue
            
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If we have a quadrilateral, it's likely our document
        if len(approx) == 4:
            pageContour = approx
            break
    
    # If we found a good document contour, transform it
    if pageContour is not None:
        warped = four_point_transform(orig, pageContour.reshape(4, 2))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to enhance readability
        scanned = cv2.adaptiveThreshold(warped_gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        return scanned
    else:
        # If we couldn't find a good quadrilateral, use the largest contour
        x, y, w, h = cv2.boundingRect(cnts[0])
        
        # Check if the bounding rectangle is reasonable
        area_percentage = (w * h) / (image.shape[0] * image.shape[1])
        
        if area_percentage > 0.3:  # Reasonable size
            # Crop the image to the bounding rectangle
            cropped = image[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Enhance the cropped image
            enhanced = cv2.adaptiveThreshold(cropped_gray, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            return enhanced
        else:
            # Just process the whole image as a fallback
            enhanced = cv2.adaptiveThreshold(gray, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            return enhanced

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
