import streamlit as st
import numpy as np
import cv2
import pytesseract
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------------
st.set_page_config(page_title="Aadhaar Fraud Detection", layout="centered")
st.title("🧠 AI-Powered Aadhaar Verification & Fraud Detection")
st.write("Upload Aadhaar card to detect tampering and extract details automatically.")

# -----------------------------------------------------
# Tesseract OCR Path (Update if needed)
# -----------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Y SRIRAM\tesseract.exe"

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("document_authentication_model_v2.h5")

with st.spinner("🔄 Loading AI Model..."):
    model = load_cnn_model()
st.success("✨ Model Loaded Successfully!")


# -----------------------------------------------------
# Image Preprocessing for OCR
# -----------------------------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    bw = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,25,10)

    bw = cv2.resize(bw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("debug_preprocessed_for_ocr.png", bw)
    return bw


# -----------------------------------------------------
# OCR Text
# -----------------------------------------------------
def extract_text(img):
    config = r'--oem 3 --psm 3'
    return pytesseract.image_to_string(img, config=config, lang='eng')


# -----------------------------------------------------
# Extract Aadhaar Name, DOB, Number
# -----------------------------------------------------
def extract_fields(text):

    text = text.replace("\n", " ")
    text = re.sub(r'[^A-Za-z0-9\s:/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    st.text_area("📌 Cleaned OCR Text", text, height=140)

    # Name extraction (more flexible)
    name_pattern = r"(?:Name|नाम|a\/w|S\/O)[:\s\-]([A-Za-z]{3,}(?:\s[A-Za-z]{3,}))"
    name = re.search(name_pattern, text, re.IGNORECASE)

    # DOB extraction
    dob_pattern = r"(?:DOB|DoB|Date of Birth|YOB)[^\d]*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})"
    dob = re.search(dob_pattern, text, re.IGNORECASE)

    # Aadhaar digits fallback
    digits = re.findall(r'\d', text)
    aadhaar = ''.join(digits[-12:]) if len(digits) >= 12 else None

    # Final cleanup
    result = {
        "Name": name.group(1).strip() if name else None,
        "DOB": dob.group(1).strip() if dob else None,
        "Aadhaar": aadhaar
    }

    return result



# -----------------------------------------------------
# Model Prediction
# -----------------------------------------------------
def predict_authenticity(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    confidence = float(pred)

    label = "Tampered" if confidence > 0.5 else "Genuine"
    confidence = confidence if label == "Tampered" else 1 - confidence

    return label, round(confidence, 3)


# -----------------------------------------------------
# Upload & Process
# -----------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Aadhaar Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file)
    img = np.array(img_pil)

    img = cv2.resize(img, None, fx=2.3, fy=2.3, interpolation=cv2.INTER_LINEAR)

    st.image(img, caption="📌 Upscaled Image", use_container_width=True)

    with st.spinner("⚙ Processing Image..."):
        preprocessed = preprocess_image(img)
    st.image(preprocessed, caption="🧾 Preprocessed for OCR", use_container_width=True)

    with st.spinner("🤖 Checking Authenticity..."):
        label, conf = predict_authenticity(img)

    st.subheader("🎯 Prediction Result")
    st.write(f"*Status:* {label}")
    st.write(f"*Confidence:* {conf}")

    with st.spinner("🔍 Extracting Text Fields..."):
        text = extract_text(preprocessed)
    st.text_area("📄 Raw OCR Text", text, height=150)

    fields = extract_fields(text)
    
    st.subheader("📋 Extracted Aadhaar Fields")
    st.write(f"👤 *Name:* {fields['Name'] or '❌ Not Detected'}")
    st.write(f"🎂 *DOB:* {fields['DOB'] or '❌ Not Detected'}")
    st.write(f"🔢 *Aadhaar No.:* {fields['Aadhaar'] or '❌ Not Detected'}")

    st.write("---")
    valid = label == "Genuine" and fields['Aadhaar']

    st.subheader("📌 Final Verification Status")
    if valid:
        st.success("✔ Aadhaar Verified Successfully — Looks Real!")
    else:
        st.error("⚠ Verification Failed — Missing details or tampered!")


else:
    st.info("📎 Upload Aadhaar Image to Start Verification")






