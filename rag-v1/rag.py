
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
file = open("companypolicy.txt","r")
text = file.read()

chunks = []
chunk_size = 20
overlap_size = 5
step = chunk_size-overlap_size

for i in range(0,len(text),step):
    chunks.append(text[i:i+chunk_size])
embeddings = model.encode(chunks)
question = "Can customers return products?"
question_embeddings = model.encode(question)
#print(embeddings.shape)
#print(embeddings[0])
#question_magnitude = np.linalg.norm(question_embeddings)
#chunks_magnitude = np.linalg.norm(embeddings[0])
#magnitude_product = question_magnitude * chunks_magnitude
#similarity = np.dot(question_embeddings,embeddings[0])/magnitude_product
#print(similarity)
results = []
for i in range(len(embeddings)):
    chunk_vector = embeddings[i]
    chunk_text = chunks[i]
    dot_product = np.dot(question_embeddings,chunk_vector)
    question_magnitude = np.linalg.norm(question_embeddings)
    chunk_magnitude = np.linalg.norm(chunk_vector)
    magnitude_product = question_magnitude * chunk_magnitude
    similarity = dot_product / magnitude_product
    results.append((similarity,chunk_text))
    #print("chunk",i+1,similarity) 
results.sort(reverse = True)   
top_results = results[:3]
#print(top_results) #this shows the similarity and chunks

retrevied_chunks = [] #gives only top chunks
for (similarity,chunk_text) in top_results:
    retrevied_chunks.append(chunk_text)
#print(retrevied_chunks)

context = "\n".join(retrevied_chunks)

prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the context provided.
"""

output = generator(prompt, max_new_tokens=100) 

print(output)

