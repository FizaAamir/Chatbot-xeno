# Xeno - Your AI File Assistant

Xeno is a smart chatbot that answers your questions based on the content of PDF and Word documents you upload. It uses a powerful AI model (Google's Gemini) and a Retrieval-Augmented Generation (RAG) system to find the most relevant information in your files and give you accurate answers.

## 🚀 Key Features

- **Chat with your documents:** Interact with your files in a conversational way.
- **Supports multiple formats:** Upload and process both PDF (`.pdf`) and Word (`.docx`) files.
- **Multi-document understanding:** Ask questions that require information from several documents at once.
- **Context-aware conversations:** The chatbot remembers your previous questions to provide better follow-up answers.
- **Smart retrieval:** Xeno dynamically adjusts how it searches your documents based on your query to find the best possible answer.
- **Powered by Gemini:** Utilizes Google's state-of-the-art `gemini-1.5-flash` model for high-quality responses.
- **Easy-to-use interface:** A clean and simple web interface built with Streamlit.

## 🛠️ How It Works

The application is built on a Retrieval-Augmented Generation (RAG) architecture:

1.  **File Upload & Text Extraction:** You upload your PDF or DOCX files through the Streamlit interface. The app uses `PyMuPDF` and `python-docx` to extract the text.
2.  **Text Splitting & Embedding:** The extracted text is divided into smaller chunks. These chunks are then converted into numerical representations (embeddings) using Google's embedding model.
3.  **Vector Storage:** The embeddings are stored in an in-memory vector store for fast retrieval.
4.  **Dynamic Retrieval:** When you ask a question, the app first uses the Gemini LLM to determine the optimal number of text chunks (`k`) to retrieve. This makes the search more efficient and relevant.
5.  **Context-Aware Retrieval:** The system considers your chat history to understand the context of your question before retrieving relevant chunks from the vector store.
6.  **Answer Generation:** The retrieved text chunks and your question are passed to the `gemini-1.5-flash` model, which generates a comprehensive answer.

## ⚙️ Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8 or higher

### 2. Set up the Environment

First, clone the repository (or download the code) and navigate into the project directory.

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate
```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Google API Key

The application uses the Google Gemini API. You need to get a free API key from the [Google AI Studio](https://aistudio.google.com/).

Once you have your key, open the Python script (`app.py`) and replace `'Gemini-api-key'` with your actual key:

```python
# In your python script
os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_API_KEY_HERE'
```

> **Note:** For better security, it is recommended to load your API key from a `.env` file or system environment variables rather than hardcoding it directly in the script.

## ▶️ Usage

To run the application, execute the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

The application will open in your web browser. You can then upload your documents via the sidebar and start asking questions! 