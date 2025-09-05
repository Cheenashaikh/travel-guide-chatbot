from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatOpenAI(
     model="mistralai/mistral-7b-instruct:free",
     temperature=0.5,
     openai_api_key=os.getenv("OPENAI_API_KEY"),
     openai_api_base=os.getenv("OPENAI_API_BASE"),
)
