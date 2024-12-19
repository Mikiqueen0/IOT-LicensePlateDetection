
# IOT-LicensePlateDetection

This project provides an API for detecting and recognizing Thai license plate numbers and provinces from images using [YOLO11](https://docs.ultralytics.com/models/yolo11/) and [thai-trocr](https://huggingface.co/openthaigpt/thai-trocr) models.

## Features
- Detects license plates and provinces from input images URL.
- Performs text recognition on detected license plates using the TrOCR model.
- Matches recognized text with predefined Thai province names using Levenshtein distance.

---

## Installation

### Prerequisites
- Python 3.9 or later
- Docker (optional for containerized deployment)

### Clone the Repository
```bash
git clone https://github.com/Mikiqueen0/IOT-LicensePlateDetection.git
cd IOT-LicensePlateDetection
```
### Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using Docker:
1. Build the Docker image:
	```bash
	docker build -t license-plate-detection .
	```
2. Run the container:
	```bash
	docker run -p 8000:8000 license-plate-detection
	```

## Usage

### Running the Application

Run the application locally using:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Root Endpoint

**URL**: `/`  
**Method**: `GET`  
**Response**:
```bash
{
  "details": "OCR and YOLO processing app"
}
```

#### 2. Process Image

**URL**: `/process-image/`  
**Method**: `POST`  
**Input**:
```bash
{
  "image_path": "https://example.com/path/to/image.jpg"
}
```

**Response**:

-   On success:
```bash
{
  "plate_number": "XYZ1234",
  "province": "กรุงเทพมหานคร",
  "raw_province": "บางกอก"
}
```
- On failure:
```bash
{
  "error": "Error message"
}
```
## Project Structure

### Running the Application
Project Structure
```bash
.
├── main.py               # ไฟล์หลักสำหรับแอปพลิเคชัน FastAPI
├── Dockerfile            # Dockerfile สำหรับ containerizing แอปพลิเคชัน
├── requirements.txt      # รายการ dependencies
├── best.pt               # YOLO model weights
```

## Models Used

1.  [**YOLO11**](https://docs.ultralytics.com/models/yolo11/): Used for object detection (detects license plates and province areas in the image).
2.  [**thai-trocr**](https://huggingface.co/openthaigpt/thai-trocr): Used for text recognition on cropped license plate images.

## How It Works

1.  Input image URL is provided via the API.
2.  The image is processed with YOLO to detect bounding boxes for license plates and provinces.
3.  The detected regions are passed to the TrOCR model for text recognition.
4.  The recognized text is matched against a list of Thai provinces using Levenshtein distance for the best match.

## Dependencies

-   [FastAPI](https://fastapi.tiangolo.com)
-   [Ultralytics YOLO](https://docs.ultralytics.com)
-   Transformers
-   [OpenCV](https://opencv.org/)
-   [Levenshtein](https://github.com/ztane/python-Levenshtein)

Install them using:
```bash
pip install -r requirements.txt
```
