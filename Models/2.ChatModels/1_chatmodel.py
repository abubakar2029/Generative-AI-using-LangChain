from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = model.invoke("What are the signs that a university student is on the right track?")
print(result.content)
