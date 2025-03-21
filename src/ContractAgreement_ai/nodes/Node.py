import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from  langchain_community.vectorstores import FAISS 
from  langchain_community.embeddings import OpenAIEmbeddings
import requests
from docx import Document
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
import os
from openai import OpenAI
import streamlit as st

class ClauseExtractor:
    def __init__(self, file_path):
        st.secrets["GROQ"]["API_KEY"]
        self.file_path = file_path
        #self.ollama_api_url = "http://localhost:11434/api"  # URL for Ollama Omega 3 LLM API
        self.groq_api_key = st.secrets["GROQ"]["API_KEY"]

        # Set the API URL here, no need to pass it as a parameter
        #self.groq_api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/inference")  # Default Groq API URL if not found in environment

    def extract_clauses(self):
        if not os.path.exists(self.file_path):
                raise ValueError("File not exists.")

        if self.file_path.endswith('.pdf'):
            text = self._extract_text_from_pdf(self.file_path)
        elif self.file_path.endswith('.docx'):
            text = self._extract_text_from_word(self.file_path)
        else:
            raise ValueError("Unsupported file format")

        # Once the text is extracted, we send it to Ollama Omega 3 for processing
        #print(len(text))

        return self._process_with_groq(text)

    def _extract_text_from_pdf(self, file_path):
        """
        Extracts text from a PDF file, handling both text-based and image-based PDFs.
        """
        # Check if it's a text-based PDF first
        text = self._extract_text_from_pdf_text(file_path)
        if text:
            return text
        
        # If the PDF is image-based, use OCR (Tesseract) to extract text
        return self._extract_text_from_pdf_image(file_path)

    def _extract_text_from_pdf_text(self, file_path):
        """
        Extracts text from a text-based PDF.
        """
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_text_from_pdf_image(self, file_path):
        """
        Extracts text from an image-based PDF using OCR (Tesseract).
        """
        try:
            pages = convert_from_path(file_path, 300)  # 300 DPI for OCR
            text = ""
            for page in pages:
                text += pytesseract.image_to_string(page)
            return text
        except Exception as e:
            print(f"Error extracting text from image-based PDF: {e}")
            return ""

    def _extract_text_from_word(self, file_path):
        """
        Extracts text from a Word document (.docx).
        """
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from Word document: {e}")
            return ""
        

    def _process_with_groq(self, text):
        """
        Processes text using Groq's OpenAI-compatible API with error handling.
        Returns: A list of clauses extracted from the response, or an empty list on failure.
        """
        try:
            from openai import OpenAI
            
            # Initialize client with Groq's endpoint
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_api_key
            )

            # System message to guide the AI model's behavior
            system_message = """You are an AI assistant to help in clause identification in contract agreements.
                                Do not change or strip the input text in the clause."""

            # Create chat completion with error handling
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Use the model you wish to query
                messages=[{"role": "system", "content": system_message},
                        {"role": "user", "content": text}],
                temperature=0.7,
                max_tokens=1024
            )

            # Extract the response content
            response_content = response.choices[0].message.content.strip()
            
            print("Extraction response: " + response_content)
            
            # Assuming clauses are separated by new lines or another delimiter, split the response into clauses
            lines = response_content.split("\n")  # Adjust delimiter if necessary
            
            # Filter out empty lines and join remaining lines as paragraphs
            clauses = []
            current_paragraph = []
            
            for line in lines[1:]:
                # If the line is not empty, add it to the current paragraph
                if line.strip():
                    current_paragraph.append(line.strip())
                else:
                    # If an empty line is encountered, it marks the end of a paragraph
                    if len(current_paragraph) >= 2:
                        clauses.append(" ".join(current_paragraph))
                        current_paragraph = []
            
            # Add any leftover paragraph if the text doesn't end with an empty line
            if len(current_paragraph) >= 2:
                clauses.append(" ".join(current_paragraph))
            
            return clauses

        except Exception as e:
            # Handle any exceptions that may occur during the API call
            print(f"Error occurred while processing with Groq: {e}")
            return []

        except ImportError:
                print("Error: Install openai package - 'pip install openai'")
                return ""
        except Exception as e:
                print(f"Groq API Error: {str(e)}")
                return ""

class EmbeddingManager:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.embeddings = OpenAIEmbeddings()

    def create_embeddings(self, clauses):
        embeddings = []
        for clause in clauses:
            embedding = self.embeddings.embed(clause)
            embeddings.append(embedding)
        return np.array(embeddings)

    def compare_embeddings(self, query_embedding, clauses_embeddings):
        similarities = cosine_similarity([query_embedding], clauses_embeddings)
        return similarities


class ClauseComparator:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
   
    def compare_clauses(self,uploaded_clause, standard_clause):
        """
        Compare text os standard clause using Groq's OpenAI-compatible API with error handling.Do not ommit any text from clause.
        Returns: Generated response (str) or empty string on failure.
        """
        try:
            # Initialize client with Groq's endpoint
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_api_key
            )

            # System message to set the task context for the AI model
            system_message = """You are an AI assistant helping to:
            1. Identify the type of the uploaded clause list at the top.
            2. Identify the type of the standard clause only once at the top.
            3. Compare the uploaded clause with standard clauses to check for matching types.
            4. If there is no matching clause for a uploaded clause in standard library then do not add Key Differences and Detailed Analysis for such clause to response.
            5. Do a complete legal term analysis if the clause match or even if partial match with standard library.
            """

            # User message that provides the uploaded clause and the standard clause
            user_message = f"""User uploaded clause: {uploaded_clause}
            Standard clause: {standard_clause}
            Please compare these clauses and identify if they are of the same type or provide a detailed analysis of their differences.
            """

            # Create chat completion with error handling
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Use the model you wish to query
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": user_message}],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()

        except ImportError:
            print("Error: Install openai package - 'pip install openai'")
            return ""
        except Exception as e:
            print(f"Groq API Error: {str(e)}")
            return ""


