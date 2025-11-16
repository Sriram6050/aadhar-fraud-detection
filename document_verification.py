# document_verification.py
import os
import cv2
import pytesseract
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------
# 1️⃣ Tesseract Path Setup
# ---------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Y SRIRAM\tesseract.exe"

# ---------------------------------------------------
# 2️⃣ Load Pre-Trained CNN Model
# ---------------------------------------------------
MODEL_PATH = "document_authentication_model_v2.h5"
model = load_model(MODEL_PATH)

# ---------------------------------------------------
# 3️⃣ Test Image Path (change this for each test)
# ---------------------------------------------------
img_path = r"C:\Users\Y SRIRAM\OneDrive\Desktop\reproject_documents\processed_dataset\test\genuine\108front_scaled_up.jpg"

# ---------------------------------------------------
# STEP 1: Document Authenticity Prediction (FIXED)
# ---------------------------------------------------
def predict_authenticity(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # ✅ FIXED: Corrected label mapping (genuine=0, tampered=1)
    label = "Tampered" if confidence > 0.5 else "Genuine"
    confidence = confidence if label == "Tampered" else 1 - confidence

    return label, round(confidence, 3)

# ---------------------------------------------------
# STEP 2: OCR Extraction with Preprocessing
# ---------------------------------------------------
def extract_text(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Error: Image not found at {img_path}")
        return ""

    # Convert to grayscale and denoise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply adaptive threshold (black & white)
    bw = cv2.adaptiveThreshold(
        gray_filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Resize for better OCR accuracy
    bw = cv2.resize(bw, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Optional: Save for debugging
    cv2.imwrite("preprocessed_for_ocr.jpg", bw)

    # Tesseract OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(bw, config=custom_config)

    # Clean extra spaces and newlines
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# ---------------------------------------------------
# STEP 3: Extract Key Fields (Name, DOB, Aadhaar)
# ---------------------------------------------------
def extract_fields(text):
    lines = text.splitlines()
    clean_lines = [l.strip() for l in lines if l.strip()]
    text_clean = " ".join(clean_lines)

    # Extract Name
    name_match = re.search(r"(?:Name|Narne|Nee|नाम|नाम:)\s*[:\-]?\s*([A-Za-z ]+)", text_clean, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Not Found"

    # Extract DOB (formats like DD/MM/YYYY or DD-MM-YYYY)
    dob_match = re.search(r"(\d{2}[-/]\d{2}[-/]\d{4})", text_clean)
    dob = dob_match.group(1) if dob_match else "Not Found"

    # Extract Aadhaar (1234 5678 9012)
    aadhaar_match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text_clean)
    aadhaar = aadhaar_match.group(0) if aadhaar_match else "Not Found"

    return name, dob, aadhaar

# ---------------------------------------------------
# STEP 4: Validate Extracted Fields
# ---------------------------------------------------
def validate_fields(name, dob, aadhaar):
    print("--------------------------------")
    print(f"Name: {name if name != 'Not Found' else '❌ Missing'}")
    print(f"DOB: {dob if dob != 'Not Found' else '❌ Missing'}")
    print(f"Aadhaar: {aadhaar if aadhaar != 'Not Found' else '❌ Missing'}")
    print("--------------------------------")

    # Validation passes only if all details found
    return name != "Not Found" and dob != "Not Found" and aadhaar != "Not Found"

# ---------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n📄 Document Verification Started")
    print("--------------------------------")

    # Step 1: Authenticity Prediction
    label, confidence = predict_authenticity(img_path)
    print(f"🧠 Prediction: {label} (Confidence: {confidence})")

    # Step 2: OCR Extraction
    print("\n🔍 Extracting Text via OCR...")
    text = extract_text(img_path)
    print("--------------------------------")
    print(text)
    print("--------------------------------")

    # Step 3: Extract Fields
    name, dob, aadhaar = extract_fields(text)

    # Step 4: Validate Fields
    is_valid = validate_fields(name, dob, aadhaar)

    # Step 5: Final Result Summary
    print("\n✅ Final Verification Summary")
    print("--------------------------------")
    print(f"Authenticity: {label} (Confidence: {confidence})")
    print(f"Name: {name}")
    print(f"DOB: {dob}")
    print(f"Aadhaar: {aadhaar}")
    print(f"Validation: {'🟩 Passed' if is_valid else '🟥 Failed'}")
    print("--------------------------------")
    print("🏁 Document Verification Completed.")


