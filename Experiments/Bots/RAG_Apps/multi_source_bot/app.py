import os
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Arxiv Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain import hub

### Agents
from langchain.agents import create_openai_tools_agent

## Agent Executer
from langchain.agents import AgentExecutor


load_dotenv()


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


loader = WebBaseLoader("https://www.cdc.gov/poxvirus/mpox/about/index.html")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "mpox_search",
    "Search for information about Mpox. For any questions about Mpox, you must use this tool!",
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv.name

tools = [wiki, arxiv, retriever_tool]


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages


agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What are the symptoms of Monkeypox?"})


# agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
