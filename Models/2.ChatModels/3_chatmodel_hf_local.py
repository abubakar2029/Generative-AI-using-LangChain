from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace(HuggingFacePipeline.from_model_id(model_id="model_id",task="chat",pipeline_kwargs={"max_length":512,"temperature":0.3}))

result = model.invoke("What are the signs that a university student is on the right track?")
print(result.content) 
