import streamlit as st
import os
from src.ContractAgreement_ai.llms.LLM import LLM
from src.ContractAgreement_ai.graph.Graph import Graph
from src.ContractAgreement_ai.nodes.Node import ClauseExtractor, EmbeddingManager, ClauseComparator
from src.ContractAgreement_ai.state.State import UserState
from src.ContractAgreement_ai.tools.Tools import ClauseLibraryLoader


def main():

    # Initialize the LLM and Graph
    llm = LLM(api_key=os.getenv("GROQ_API_KEY"))
    graph = Graph()

    # Function to handle file uploads
    def upload_file():
        file = st.file_uploader("Upload your contract (PDF/Word)", type=["pdf", "docx"])
        if file is not None:
            upload_dir = "uploads"  # Directory to store the uploaded files
            # Ensure the 'uploads' directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)  # Create the directory if it doesn't exist
            
            # Define the file path to store the uploaded file
            file_path = os.path.join(upload_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())  # Write the uploaded file to disk
            
            return file_path
        return None

    # Function to load and extract clauses
    def extract_clauses_from_file(file_path):
        extractor = ClauseExtractor(file_path)
        clauses = extractor.extract_clauses()
        st.write(clauses)
        #print(clauses)
        return clauses

    # Function to load clause library
    def load_clause_library(zip_file):
        loader = ClauseLibraryLoader(zip_file)
        loader.load_library()
        return loader.get_clauses()

    # Function to compare and redline clauses
    def compare_clauses(user_clause, standard_clause):
        ClauseComp = ClauseComparator()
        response = ClauseComp.compare_clauses(user_clause,standard_clause)
        # Implement clause comparison and provide redlining suggestions
        return f"{response}"

    # Streamlit UI
    st.title("AI Assistant for Contract Clause Negotiation")

    st.write("This is app has limitation due to free tier token usage! Please use smaller contract file for comparision:")
    st.write("Sample contract and clause library can be found @ https://tinyurl.com/ClausesLibrary")
    user_type = st.radio("Select User Type", ("Buyer", "Supplier"))

    # Upload clause library and contract file
    zip_file = st.file_uploader("Upload Clause Library (ZIP file)", type=["zip"])
    if zip_file:
        clauses = load_clause_library(zip_file)
        st.write("Loaded Clause Library:", clauses)

    file_path = upload_file()
    if file_path:
        # Extract clauses from the uploaded contract
        contract_clauses = extract_clauses_from_file(file_path)
        st.write("Extracted Clauses from Contract:", contract_clauses)

        # Handle interaction based on user type
        if user_type == "Buyer":
            # Buyer vetting of clauses
            for clause in contract_clauses:
                print("clause " + clause)
                redline_suggestion = compare_clauses(clause, clauses)  # Compare with first standard clause
                st.write(f"Analysis: {redline_suggestion}")

        elif user_type == "Supplier":
            # Supplier negotiation
            negotiate_clause = st.text_input("Propose new clause language:")
            if negotiate_clause:
                st.write(f"Negotiated Clause: {negotiate_clause}")
                # Add negotiation logic here
                response = llm.query(f"Negotiate the following clause: {negotiate_clause}")
                st.write("Negotiation Response:", response)

    # Interaction handling (e.g., chat)
    user_input = st.text_input("Chat with the system:")
    if user_input:
        response = llm.query(user_input)
        st.write("Response from LLM:", response)

#if __name__=="__main__":
#    main()
