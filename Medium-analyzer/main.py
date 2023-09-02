from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# pinecone client
import pinecone

import os

# initialize pinecone client
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)

if __name__ == "__main__":
    print("Hello Vector stores")
    # turn text into a document
    loader = TextLoader(
        "/home/peter/LangChain/LangChain-Lab/Medium-analyzer/mediumblog1.txt"
    )

    # document: list that contains the text turned into a document
    document = loader.load()

    # initialize a text splitter object
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # split the document into chunks. texts is a list of strings (chunks)
    texts = text_splitter.split_documents(document)

    # initialize an embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # insert the chunked documents into the vector store
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    # like all chains, we supply an llm. The chain_type stuff means stuff context in prompt
    # docsearch is a vector store object, so it has a as_retriever method. This transforms it into a retriever object.
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is a vector DB? Give me a short answer meant to be understood by begginers."
    result = qa({"query": query})
    print(result)
