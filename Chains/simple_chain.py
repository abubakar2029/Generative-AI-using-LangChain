from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Briefly generate 5 facts about {topic}.",
    input_variables=["topic"]
    )

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"topic": "Student life of a University student"})

print(result)

chain.get_graph().print_ascii()