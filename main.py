from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import Levenshtein
import requests
from PIL import Image
import io

app = FastAPI()

# Define a Pydantic model to handle input
class ImageRequest(BaseModel):
    image_path: str

# Load TrOCR models
processor_plate = TrOCRProcessor.from_pretrained('openthaigpt/thai-trocr')
model_plate = VisionEncoderDecoderModel.from_pretrained('openthaigpt/thai-trocr')

thai_provinces = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา",
    "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นราธิวาส", "น่าน", "บึงกาฬ",
    "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พะเยา", "พังงา", "พัทลุง",
    "พิจิตร", "พิษณุโลก", "เพชรบูรณ์", "เพชรบุรี", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อำนาจเจริญ",
    "อุดรธานี", "อุทัยธานี", "อุบลราชธานี", "อ่างทอง"
]

# Load YOLO model (ensure correct path to YOLO weights)
yolo_model = YOLO('best.pt')

CONF_THRESHOLD = 0.5

def get_closest_province(input_text, provinces):
    min_distance = float('inf')
    closest_province = None

    for province in provinces:
        distance = Levenshtein.distance(input_text, province)
        if distance < min_distance:
            min_distance = distance
            closest_province = province

    return closest_province, min_distance

@app.get("/")
async def index():
    return {"details": "OCR and YOLO processing app"}

@app.post("/process-image/")
async def process_image(request: ImageRequest):
    try:
        image_path = request.image_path
        
        # Step 1: Download the image
        response = requests.get(image_path, stream=True, timeout=10)
        if response.status_code != 200:
            return {"error": f"Failed to fetch the image: HTTP {response.status_code}"}
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return {"error": f"Invalid content type: {content_type}"}
        
        # Step 2: Open the image
        try:
            image = Image.open(io.BytesIO(response.content))
        except Exception as e:
            return {"error": f"Failed to open image: {e}"}
        
        # Step 3: Convert to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Perform YOLO processing
        results = yolo_model(image)
        data = {"plate_number": "", "province": "", "raw_province": ""}

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.flatten())

                # Crop the image based on the bounding box
                cropped_image = image[y1:y2, x1:x2]
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Enhance the cropped image
                equalized_image = cv2.equalizeHist(cropped_image_gray)
                _, thresh_image = cv2.threshold(equalized_image, 65, 255, cv2.THRESH_BINARY_INV)
                cropped_image_3d = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
                resized_image = cv2.resize(cropped_image_3d, (128, 32))
                
                pixel_values = processor_plate(resized_image, return_tensors="pt").pixel_values
                generated_ids = model_plate.generate(pixel_values)
                generated_text = processor_plate.batch_decode(generated_ids, skip_special_tokens=True)[0]

                if int(box.cls.item()) == 1:
                    generated_province, distance = get_closest_province(generated_text, thai_provinces)
                
                if int(box.cls.item()) == 0:
                    data["plate_number"] = generated_text
                else:
                    data["raw_province"] = generated_text
                    data["province"] = generated_province

        if not data["plate_number"]:
            return {"error": "No license plate number detected."}
        return data

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
