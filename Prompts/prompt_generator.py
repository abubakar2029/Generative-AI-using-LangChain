from langchain_core.prompts import PromptTemplate

template = """You are an expert research assistant. Summarize the research paper titled '{paper_name}' in the style of '{style}' with a length of approximately {length} words."""

prompt = PromptTemplate(
    input_variables=["paper_name", "style", "length"],
    template=template
)

prompt.save("summary_prompt.json")