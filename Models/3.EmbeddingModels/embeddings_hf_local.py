from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "The Eiffel Tower is located in Paris and is one of the most visited monuments in the world.",
    "Healthy eating habits include consuming more vegetables, fruits, and whole grains.",
    "Machine learning allows computers to learn patterns from data without being explicitly programmed.",
    "Football is one of the most popular sports played and watched globally.",
    "Electric vehicles help reduce carbon emissions and support environmental sustainability.",
    "A savings plan helps people manage their finances and achieve future goals."
]

embeddings = model.embed_documents(texts)

query = "Why are electric cars good?"

query_embedding = model.embed_query(query)

scores = cosine_similarity(query_embedding, embeddings[4])[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Most similar text: {texts[index]} with score {score}")
# hmm cosine similarity find krean gan 
# similarities = model.similarity_search(query_embedding, embeddings)
