# C_WYSWYG
# LangChain RAG Setup

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain and Ollama embeddings to answer user queries based on the context from PDFs. The solution involves processing and splitting PDFs, storing the content in a vector database, and retrieving relevant answers using language models.

## Prerequisites

Ensure you have the following installed on your machine:

- Python 3.7+ 
- Pip (Python package manager)

### Dependencies

1. **LangChain**: A framework for building language model-powered applications.
2. **Ollama Embeddings**: For generating embeddings using Ollama models.
3. **Chroma**: A vector store used to store document embeddings.
4. **Flask**: For handling HTTP requests (if you're deploying this as a Slack bot or web service).
5. **PyPDFLoader**: For loading and splitting PDFs into documents.

### Installation

To set up the environment, follow the steps below:

1. Install the necessary Python packages:

langchain==0.0.168
langchain-community==0.0.3
langchain-ollama==0.0.5
langchain-text-splitters==0.0.6
langchain-core==0.0.9
slack-sdk==3.19.1
flask==2.2.2


## Setting Up the Project

Before running the script, ensure the following:

### 1. **PDF Files**

Place your PDF files (e.g., `2021.pdf`, `2023.pdf`, `2024.pdf`) in the project directory or specify the path in the script.

### 2. **Configure Environment Variables**

Set the environment variable for the protobuf implementation by adding the following in the script or setting it manually:

```python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
