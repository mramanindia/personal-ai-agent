import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import re
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Step 1: Extract data from PDF
def extract_text_with_headings(pdf_path):
    reader = PdfReader(pdf_path)
    data = []
    for page in reader.pages:
        text = page.extract_text()
        lines = text.split("\n")
        for line in lines:
            if re.match(r"^[A-Z][A-Z\s]+$", line.strip()):  # Heading (uppercase assumption)
                data.append({"heading": line.strip(), "content": ""})
            elif data:
                data[-1]["content"] += line.strip() + " "
    return data

# Step 2: Process text into meaningful chunks
def chunk_by_headings(data, max_chunk_size=500):
    chunks = []
    for section in data:
        heading = section["heading"]
        content = section["content"]
        words = content.split()
        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i:i + max_chunk_size])
            chunks.append(f"{heading}\n{chunk}")
    return chunks

# Step 3: Embed and store the data
def store_in_faiss_advanced(chunks, faiss_index_path="faiss_index"):
    model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, normalize_embeddings=True)  # Normalize embeddings

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(np.array(embeddings))

    faiss.write_index(index, faiss_index_path)
    return faiss_index_path, chunks

def re_rank_results(chunks, query, top_k=3):
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Compute cosine similarity
    embeddings = model.encode(chunks, normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Rank and retrieve top-k
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices], similarities[ranked_indices]
    
def query_faiss_with_reranking(index, query, chunks, top_k=5):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    reranked_chunks, reranked_scores = re_rank_results(retrieved_chunks, query, top_k=3)
    return reranked_chunks
    

# Step 3: Use OpenAI GPT for generating answers
def get_answer_with_openai(context, query, openai_api_key):
    print("Context provided for this query is: ", context)
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
)

    # Construct the conversation messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context only."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
    ]

    # Call OpenAI Chat Completion API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" for a faster, cheaper model
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    
    # Extract and return the assistant's reply
    return response.choices[0].message.content.strip()


    

# Main function as per your request
def main(pdf_path, openai_api_key):
    # Step 1: Extract and store data
    data = extract_text_with_headings(pdf_path)
    chunks = chunk_by_headings(data)
    faiss_index_path, formatted_chunks = store_in_faiss_advanced(chunks)

    # Step 2: Load FAISS and query
    # (dummy index for now, replace with actual FAISS index loading)
    index = None
    user_query = "What is this about?"  # You can get this input via a separate field
    relevant_chunks = query_faiss_with_reranking(index, user_query, formatted_chunks)

    # Step 3: Use OpenAI to get the answer
    context = "\n".join(relevant_chunks)
    answer = get_answer_with_openai(context, user_query, openai_api_key)

    print(f"\nAnswer: {answer}")
    messagebox.showinfo("Answer", f"Answer: {answer}")

# GUI Setup
def open_pdf_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        pdf_file_entry.delete(0, tk.END)
        pdf_file_entry.insert(0, file_path)

def on_run_button_click():
    pdf_path = pdf_file_entry.get()
    openai_api_key = openai_key_entry.get()
    
    if not pdf_path or not openai_api_key:
        messagebox.showerror("Error", "Please provide both the PDF file and OpenAI API key.")
        return
    
    # Run the main function with the selected PDF and OpenAI key
    main(pdf_path, openai_api_key)

# Creating the main window
root = tk.Tk()
root.title("PDF & OpenAI Query Tool")

# PDF File Selector
pdf_file_label = tk.Label(root, text="Select PDF File:")
pdf_file_label.pack(pady=10)

pdf_file_entry = tk.Entry(root, width=50)
pdf_file_entry.pack(pady=5)

pdf_file_button = tk.Button(root, text="Browse", command=open_pdf_file)
pdf_file_button.pack(pady=5)

# OpenAI API Key input
openai_key_label = tk.Label(root, text="Enter OpenAI API Key:")
openai_key_label.pack(pady=10)

openai_key_entry = tk.Entry(root, width=50)
openai_key_entry.pack(pady=5)

# Run button
run_button = tk.Button(root, text="Run", command=on_run_button_click)
run_button.pack(pady=20)

# Run the GUI loop
root.mainloop()
