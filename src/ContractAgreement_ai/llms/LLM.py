from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('/Users/prasadkarani/Documents/AgenticAIWorkspace/.env')
load_dotenv()  #load all the environment variables
from openai import OpenAI

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

class LLM:
    def __init__(self, api_key):
        self.api_key = api_key
        # System message to guide the AI model's behavior
        self.system_message = """You are an AI assistant to help in supplier draft a mutually acceptable clause for contract agreements. 
                                Do not strip any text in the clause."""

    def query(self, prompt):
        
        # Initialize client with Groq's endpoint
        client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )
        response = client.chat.completions.create( model="llama3-70b-8192",  # Use the model you wish to query
                messages=[{"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}],
                temperature=0.7
                #max_tokens=1024
                )
        response_content = response.choices[0].message.content.strip()
        return response_content