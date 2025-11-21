from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, language="en")

query = "Successful traits of tier-3 university students?"
results = retriever.invoke(query)


for i, result in enumerate(results, start=1):
    print(f"Result {i}: {result}")