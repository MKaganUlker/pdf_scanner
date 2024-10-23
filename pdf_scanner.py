import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import os

# Configure tesseract path (if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_pdf_to_image(pdf_path):
    # Load PDF and convert each page to an image
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images.append(img)
    return images

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def detect_table(image):
    # Detect horizontal and vertical lines using morphology to identify table structure
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)

    # Combine lines to detect intersection points (tables have both horizontal and vertical lines)
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    # Find contours of the table
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return True, table_structure
    return False, None

def save_image_as_png(image, output_path):
    cv2.imwrite(output_path, image)

def scan_pdf_for_tables(pdf_path, output_dir):
    images = convert_pdf_to_image(pdf_path)
    for i, image in enumerate(images):
        processed_image = preprocess_image(image)
        has_table, table_img = detect_table(processed_image)
        
        if has_table:
            output_file = os.path.join(output_dir, f"page_{i+1}_table.png")
            save_image_as_png(image, output_file)
            print(f"Table found on page {i+1}, saved as {output_file}")
        else:
            print(f"No table found on page {i+1}.")

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ")
    output_dir = input("Enter the directory to save PNG files: ")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scan_pdf_for_tables(pdf_path, output_dir)
