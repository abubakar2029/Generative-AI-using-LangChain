from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 3 pointer summary of:\n\n{report}",
    input_variables=["report"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({"topic": "Problems in Agriculture Sector of Faisalabad Pakistan than can be solved by tech students"})
print(result) 

chain.get_graph().print_ascii()