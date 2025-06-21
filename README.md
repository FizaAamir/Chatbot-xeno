📄 ## Xeno – Chat with Your Documents
Xeno is an AI-powered assistant that lets you chat with your uploaded PDF and Word documents using natural language. Backed by Google's Gemini (via LangChain), it retrieves, understands, and answers your queries with reference to the source content. Whether you're working with research papers, contracts, or reports, Xeno simplifies document interaction through intelligent retrieval and summarization.

🚀 Features
📥 Multi-file Upload: Supports PDFs and Word (.docx) files.
🤖 Contextual Q&A: Uses Gemini 1.5 Flash to answer questions based on uploaded content.
🔍 Dynamic Retrieval: Automatically adjusts the number of document chunks (k) retrieved based on the type of user query.
🧠 History-Aware Conversation: Retains chat context to improve follow-up question understanding.
📚 Source-Aware Responses: Associates answers with the document they were retrieved from.
🧩 Chunking & Embedding: Splits documents intelligently and embeds them using Google’s embedding model for accurate semantic search.

🛠️ Tech Stack
Tool/Library	Purpose
Streamlit	UI framework
LangChain	RAG pipeline & retrieval chains
Google Gemini	LLM for reasoning and Q&A
Google Embeddings	Text embeddings for retrieval
PyMuPDF (fitz)	PDF parsing
python-docx	Word document parsing
InMemoryVectorStore	Temporary vector database
SentenceTransformerEmbeddings	Optional alt embedding model

⚙️ Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/yourusername/xeno-doc-chat.git
cd xeno-doc-chat

2. Install Dependencies
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
```source venv/bin/activate  # or .\venv\Scripts\activate on Windows```
Install the requirements:

bash
Copy
Edit
pip install -r requirements.txt
You’ll need Python 3.8+

3. Set Google API Key
Add your Google API key (for Gemini & embeddings) as an environment variable:

bash
Copy
Edit
```export GOOGLE_API_KEY="your-gemini-api-key"```
Or edit it directly in the script (not recommended for production):

python
Copy
Edit
```os.environ['GOOGLE_API_KEY'] = 'your-gemini-api-key'```
4. Run the App
bash
Copy
Edit
streamlit run app.py
Open the URL provided in your terminal to interact with Xeno.

🧪 Example Use Cases
“What are the main differences between the uploaded reports?”

“Summarize the conclusions from Marketing_Report.pdf.”

“List all KPIs mentioned in both documents.”

“What does Contract_A.docx say about payment terms?”

📁 Project Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit application
├── README.md               # You are here
├── requirements.txt        # Python dependencies
🧠 How It Works
Upload Docs: PDFs and Word files are parsed into text.

Chunking & Embedding: Text is split into overlapping chunks and vectorized using Google embeddings.

Query Processing:

Chat history-aware question rewriting

Dynamically calculates optimal k for retrieval

Retrieves relevant chunks

LLM Response: Gemini answers the question using the retrieved context.

Response Display: Chat UI shows the answer and tracks history.

## 📝 Requirements
Python ≥ 3.8

Streamlit

LangChain

PyMuPDF

python-docx

Google Generative AI SDK (via LangChain)

✅ To-Do / Future Features
 Persistent vector store (e.g., FAISS or ChromaDB)
 Highlighting exact answer sources
 Support for tables/images extraction
 Export chat as PDF or DOCX

📜 License
MIT License. See LICENSE file.

🙋‍♀️ Author
Built by Fiza Aamir – AI Engineer passionate about LLM applications and document intelligence.
