import pytesseract
from PIL import Image
import re

# Set the path to tesseract.exe (change if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Y SRIRAM\tesseract.exe"

# Path to your test document
img_path = r"C:\Users\Y SRIRAM\OneDrive\Desktop\reproject_documents\processed_dataset\test\genuine\107front_scaled_down.jpg"


# Extract text from image
img = Image.open(img_path)
text = pytesseract.image_to_string(img)

print("🔍 Extracted Text:")
print("--------------------------------")
print(text)
print("--------------------------------")

# Basic Aadhaar field validations
name_match = re.search(r"Name[:\s]*([A-Z][a-z]+\s[A-Z][a-z]+)", text)
aadhaar_match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
dob_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)

print("\n✅ Field Extraction Results:")
print("Name:", name_match.group(1) if name_match else "Not Found")
print("DOB:", dob_match.group(0) if dob_match else "Not Found")
print("Aadhaar No:", aadhaar_match.group(0) if aadhaar_match else "Not Found")

# Validation logic
if aadhaar_match and len(aadhaar_match.group(0)) == 14:
    print("\n🟩 Document format looks valid")
else:
    print("\n🟥 Invalid or tampered document format")
