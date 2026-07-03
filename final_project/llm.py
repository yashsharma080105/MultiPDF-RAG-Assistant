import os
from dotenv import load_dotenv
import google.generativeai as genai


# Load Environment Variables

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)



# Load Gemini Model

def load_llm():

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    return model


# Load once
model = load_llm()



# Generate Answer


def generate_answer(question, context):

    prompt = f"""
You are an AI assistant that answers questions ONLY using the provided document context.

Instructions:
1. Use only the given context.
2. Do not make up information.
3. If the answer is not found, reply:
   "I couldn't find the answer in the uploaded documents."
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 512
        }
    )

    return response.text
