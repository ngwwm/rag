import os
import docx  # For .docx files (install with `pip install python-docx`)
from PyPDF2 import PdfReader  # For .pdf files (install with `pip install PyPDF2`)

def convert_docx_to_txt(docx_path, txt_path):
    """
    Convert a .docx file to a .txt file.
    """
    try:
        doc = docx.Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                text.append(paragraph.text)
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(text))
        print(f"Converted {docx_path} to {txt_path}")
        return True
    except Exception as e:
        print(f"Error converting {docx_path}: {str(e)}")
        return False

def convert_pdf_to_txt(pdf_path, txt_path):
    """
    Convert a .pdf file to a .txt file.
    """
    try:
        pdf_reader = PdfReader(pdf_path)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(text))
        print(f"Converted {pdf_path} to {txt_path}")
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False

def batch_convert(directory_path):
    """
    Batch convert all .docx and .pdf files in the specified directory to .txt files.
    Track successful and failed conversions.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    success_count = 0
    failure_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it's a .docx file
        if filename.endswith('.docx'):
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(directory_path, txt_filename)
            if convert_docx_to_txt(file_path, txt_path):
                success_count += 1
            else:
                failure_count += 1
        
        # Check if it's a .pdf file
        elif filename.endswith('.pdf'):
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(directory_path, txt_filename)
            if convert_pdf_to_txt(file_path, txt_path):
                success_count += 1
            else:
                failure_count += 1

    # Print summary after conversion
    print("\nBatch Conversion Summary:")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed to convert: {failure_count} files")

if __name__ == "__main__":
    # Specify the directory containing your .docx and .pdf files
    directory = input("Enter the directory path where your .docx or .pdf files are located: ")
    batch_convert(directory)