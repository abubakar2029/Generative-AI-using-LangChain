from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace(HuggingFaceEndpoint(repo_id="",task="chat"))

result = model.invoke("What are the signs that a university student is on the right track?")
print(result.content) 
