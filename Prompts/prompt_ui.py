 
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# model = ChatHuggingFace(llm=
#     HuggingFacePipeline.from_model_id(
#         model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         task="text-generation",
#         pipeline_kwargs={
#             "max_new_tokens": 512,
#             "temperature": 0.3,
#         }
#     )
# )

st.header("Research Tool")

prompt = load_prompt("summary_prompt.json")
paper_name = st.text_input("Enter the research paper name:")
style_choice = st.selectbox("Select the summary style:", ["Bullet Points","Code Oriented", "Paragraph", "Key Takeaways"])
output_length = st.slider("Select the output length:", min_value=50, max_value=500, value=150, step=10)


if st.button("Summarize"):
    formatted_prompt = prompt.format(
        paper_name=paper_name,
        style=style_choice,
        length=output_length
    )
    result = model.invoke(formatted_prompt)
    st.write(result.content)
