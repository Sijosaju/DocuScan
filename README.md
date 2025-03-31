# DocuScan

## Overview

DocuScan is a Flask-based document scanning application that leverages OpenCV and NumPy to process images of documents. It automatically detects the document's outer boundary, applies a perspective transformation to obtain a top-down view, and uses adaptive thresholding to produce a scanned-like output.

## Features

- **Document Detection:** Automatically detects the document’s edges using contour detection.
- **Perspective Transformation:** Corrects the document's orientation to provide a flat, top-down view.
- **Adaptive Thresholding:** Enhances the scanned image quality by applying adaptive thresholding.
- **File Upload Support:** Allows uploads of PNG, JPG, JPEG, and PDF files.
- **Logging:** Uses Python’s logging module to track and debug application events.



