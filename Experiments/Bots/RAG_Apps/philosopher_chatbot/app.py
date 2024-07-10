import os
import cassio
from langchain_astradb import AstraDBVectorStore

# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

from datasets import load_dataset
from langchain.schema import Document

from langchain.prompts import ChatPromptTemplate

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


load_dotenv()


# Enter your settings for Astra DB and OpenAI:
os.environ["ASTRA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


# RAG Pipeline

# Embedding Model and Vector Store ===================

# Configure your embedding model and vector store
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embedding,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
print("Astra vector store configured")


# Load a sample dataset
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])


# Constructs a set of documents from your data. Documents can be used as inputs to your vector store.
docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        # Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    # Create a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)

# Create embeddings by inserting your documents into the vector store.
inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")


# Checks your collection to verify the documents are embedded.
# print(vstore.astra_db.collection("test").find())

# Basic Retrieval ===================


retriever = vstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
response = chain.invoke(
    "In the given context, what is the most important to allow the brain and provide me the tags?"
)

print(response)
# Cleanup
# WARNING: This will delete the collection and all documents in the collection
# vstore.delete_collection()
