# ----- importing dependencies -----
import os
import time
import cohere
import streamlit as st

from dotenv                                    import load_dotenv
from langchain_groq                            import ChatGroq
from langchain.chains                          import RetrievalQA
# from cohere.responses.chat                     import StreamEvent
from langchain_google_genai                    import GoogleGenerativeAI 
from langchain_text_splitters                  import CharacterTextSplitter
from langchain_community.embeddings            import HuggingFaceEmbeddings
from langchain_community.vectorstores          import FAISS
from langchain_community.vectorstores          import Chroma
from langchain_community.document_loaders      import WebBaseLoader
from langchain_experimental.text_splitter      import SemanticChunker
from langchain_community.document_loaders      import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders.csv_loader import CSVLoader

from pprint import pprint

from langchain_cohere import ChatCohere, CohereRagRetriever
from langchain_core.documents import Document

load_dotenv()

# llm = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
# rag = CohereRagRetriever(llm=llm, connectors=[])
llm = cohere.Client("Ty0GTOFkczasbpunJ1k01c1e5w98m7OPeUfFOkt4")

def load_data_and_get_docs(user_query):
    loader = CSVLoader('datasets/dataset1.csv')
    data = loader.load()
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embedding_model)
    # db = Chroma.from_documents(docs,embedding_model)
    docs = db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(user_query)
    # docs = retriever.invoke(user_query)
    return docs

# ----- initializing session state -----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- generating chat -----
def display_chat(user_input,response): 
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    USER = "user"
    ASSISTANT = "assistant"
    # if user_input:
    st.chat_message(USER).write(user_input)
    st.session_state.messages.append({'role':'user','content':user_input})
    st.chat_message(ASSISTANT).write(response)
    st.session_state.messages.append({'role':'assistant','content':response})

# ----- getting query from the user -----
def get_text():
    input_text = st.chat_input("Write your query here...", key="input")
    return input_text

# ----- defining the main function -----
def main():        
    st.set_page_config(page_title="RAG chatbot",layout="centered")
    user_input = get_text()

    if user_input is None or user_input == "":
        st.markdown("<h2 style=color:#2149f8; text-align=center>I am your chatbot to do a little chat.</h2>",unsafe_allow_html=True)
    else:
        docs = load_data_and_get_docs(user_input)
        documents = []
        for doc in docs:
            d = {"title":"", "snippet": doc.page_content}
            documents.append(d)
        client = llm.chat(model="command-r",message=user_input,documents=documents)
        answer = client.text
        display_chat(user_input,answer)
        
if __name__ == '__main__':
    main()

