Aadhaar Fraud Detection using AI & OCR

This project verifies the authenticity of Aadhaar cards using:

✔ Convolutional Neural Network (CNN)
✔ Image Preprocessing (OpenCV)
✔ Optical Character Recognition (Tesseract)
✔ Streamlit User Interface

It detects whether an Aadhaar card is Genuine or Tampered and extracts key fields such as:

Name

Date of Birth

Aadhaar Number


📦 Download Project Files (Dataset + Model)

Due to large size, dataset and trained model are provided via Google Drive:

🔗 Google Drive Download:
https://drive.google.com/file/d/1zZIqUwtVRsPixrcXvaAOIc8-lHM4lupP/view?usp=sharing

Includes:

📌 Synthetic Aadhaar Dataset (Train & Test)

🧾 Preprocessed images

🧠 Trained CNN model .h5

📁 Full project ZIP for execution

⚠️ Download and extract this ZIP, then place model file in the same folder as app.py.

🚀 How to Run the Project

1️⃣ Install Dependencies 

pip install -r requirements.txt

2️⃣ Install Tesseract OCR

Download from → https://github.com/UB-Mannheim/tesseract/wiki

Update this line in app.py with your Tesseract installation path:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

3️⃣ Run Streamlit App
streamlit run app.py


Upload Aadhaar image → Get result instantly!



→ Prediction (Genuine / Tampered)
→ Confidence Score
→ Extracted Aadhaar Fields
→ Automatic Fraud Validation
