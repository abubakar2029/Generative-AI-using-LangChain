from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a 3 pointer summary of:\n\n{report}",
    input_variables=["report"]
)

prompt2 = PromptTemplate(
    template="Generate a 3 question quiz based on the following report:\n\n{report}",
    input_variables=["report"]
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)
merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

document_sample = """
Climate change is one of the most significant environmental challenges of the 21st century.
It refers to long-term shifts in temperatures and weather patterns, primarily caused by human
activities such as burning fossil fuels, deforestation, and industrial emissions. These actions
increase greenhouse gases in the atmosphere, trapping heat and raising global temperatures.

The effects of climate change are visible worldwide. Sea levels are rising due to melting ice
caps, extreme weather events such as floods and droughts are becoming more frequent, and many
animal species are losing their natural habitats. These environmental disruptions also threaten
human life, affecting agriculture, health, and access to clean water.

To address climate change, both governments and individuals must take action. Governments can
implement policies promoting renewable energy, regulate emissions, and invest in sustainable
infrastructure. Meanwhile, individuals can reduce waste, conserve energy, and support
environmentally responsible initiatives. Collective effort is necessary to slow down the
negative impacts and protect the planet for future generations.
"""

result = chain.invoke({"report": document_sample})
print(result)

chain.get_graph().print_ascii()