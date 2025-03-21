import zipfile
import os
from docx import Document

class ClauseLibraryLoader:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.clause_dict = {}

    def load_library(self):
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall("clause_library")
        
        # Use os.walk() to recursively walk through all subdirectories and files
        for root, dirs, files in os.walk("clause_library"):
            for filename in files:
                file_path = os.path.join(root, filename)

                # Process only .docx files
                if filename.endswith(".docx"):
                    try:
                        self.clause_dict[filename] = self.read_docx(file_path)
                        print(self.clause_dict[filename])
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
                else:
                    print(f"Skipping non-docx file: {filename}")

    def read_docx(self, file_path):
        # Open and read the docx file using python-docx
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)

    def get_clauses(self):
        return list(self.clause_dict.values())

