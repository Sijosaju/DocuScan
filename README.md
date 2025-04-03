# DocuScan - Document Scanning Web Application

DocuScan is a Flask-based web application designed to scan and enhance document images. It allows users to upload images (PNG, JPG, JPEG) or PDFs, and it processes them to produce clear, scanned-looking documents. The application uses advanced image processing techniques to detect document boundaries, correct perspective, and enhance readability.

## Key Features

* **Document Boundary Detection:** Automatically detects the edges of a document within an image.
* **Perspective Correction:** Corrects the perspective of angled or skewed document photos, ensuring a straight, top-down view.
* **Image Enhancement:** Applies adaptive thresholding and other image processing techniques to improve document clarity and readability.
* **File Upload Support:** Accepts image files (PNG, JPG, JPEG) and PDFs as input.
* **Web-Based Interface:** Provides a user-friendly interface for easy document scanning.
* **Output Image Generation:** Generates a processed, scanned-looking image that can be downloaded or viewed directly in the browser.

## Technologies Used

* Python
* Flask
* OpenCV (cv2)
* NumPy
* Werkzeug

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sijosaju/DocuScan.git
    cd DocuScan
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install Flask opencv-python numpy Werkzeug
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```

## Usage

1.  Access the application through your web browser (usually at `http://127.0.0.1:5000/`).
2.  Upload an image or PDF file containing a document.
3.  Click the "Scan" button to process the document.
4.  The processed, scanned-looking image will be displayed on the page.



