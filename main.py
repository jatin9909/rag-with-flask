from dotenv import load_dotenv
import os
import re
import nltk
import bs4
from flask import Flask, request
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader,ArxivLoader,PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads

app = Flask(__name__)

folder_path = "db"
cached_llm = Ollama(model="llama3")
embeddings=CohereEmbeddings()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)





# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.route("/ai", methods=["POST"])
def aipost():
    print("/ai post called")

    json_content = request.json
    query = json_content.get("query")

    response = cached_llm.invoke(query)
    print(response)

    response_answer = {"answer":response}

    return response_answer

@app.route("/simplerag", methods=["POST"])
def simplerag():

    json_content = request.json
    query = json_content.get("query")
    vectorstore = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever() # for similarity search

    # Prompt    
    # prompt = hub.pull("rlm/rag-prompt")

    # Prompt
    template = """You are a Q&A assistant, You will refer the context provided and answer the question. 
    If you dont know the answer , reply that you dont know the answer:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()}
        | prompt
        | cached_llm
        | StrOutputParser()
    )

    print(rag_chain)

    response = rag_chain.invoke(query)
    print(response)

    response_answer = {"answer":response}

    return response_answer

@app.route("/multiquery", methods=["POST"])
def multiquery():
    json_content = request.json
    query = json_content.get("query")
    vectorstore = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever() # for similarity search

    # Multi Query: Different Perspectives
    template_1 = """You are an AI language model assistant. 
    Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector 
    database. 
    By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: 
    {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template_1)

    generate_queries = (
    prompt_perspectives 
    | cached_llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )

    #Unique union of retrieved docs
    def get_unique_union(documents: list[list]):
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": query})
    # print(docs) #It will retrive the context docs

    template_2 = """Answer the following question based on this context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template_2)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | cached_llm
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question":query})
    print("response - ", response)
    response_answer = {"answer":response}

    return response_answer




@app.route("/pdf", methods=["post"])
def uploadpdf():
    file = request.files['file']
    filename = file.filename
    save_file = "pdf/"+filename
    file.save(save_file)
    print("{} saved".format(filename))

    docs = PDFPlumberLoader(save_file)
    load_docs = docs.load_and_split()

    splits = text_splitter.split_documents(load_docs)

    # Step 4: Embed the documents
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=folder_path)
    vectorstore.persist()

    response = {"status":"file saved", "filename":filename, "doc_len":len(load_docs), "chunk_len":len(splits)}
    return response



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
