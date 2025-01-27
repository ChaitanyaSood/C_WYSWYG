import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.rtm_v2 import RTMClient
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Set environment variable for protobuf implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Slackbot credentials and setup
slack_token = 'xoxb-8373902922944-8352858469539-4k4LDyFgVvZ6Ml6J9pSjmG4k'  # Replace with your Slack bot token
client = WebClient(token=slack_token)
app = Flask(__name__)

# Function to load and split PDFs
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Load and process multiple PDFs
pdf_files = ["2021.pdf", "2023.pdf", "2024.pdf"]
all_pages = []
for pdf in pdf_files:
    all_pages.extend(load_and_split_pdf(pdf))

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(all_pages)

# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

# Define retrieval logic
local_model = "llama3.2"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# Define RAG chain
rag_prompt_template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Slack event listener for messages
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    event = data.get("event", {})

    if "subtype" in event and event["subtype"] == "bot_message":
        return jsonify({"status": "ok"})

    user_message = event.get("text", "")
    channel = event.get("channel")

    if user_message:
        response = chain.invoke(user_message)

        try:
            # Send response to Slack
            client.chat_postMessage(
                channel=channel,
                text=response
            )
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    return jsonify({"status": "ok"})


# Start the Flask app
if __name__ == "__main__":
    app.run(port=3000)