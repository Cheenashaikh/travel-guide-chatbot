import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from model import model
import warnings

warnings.filterwarnings("ignore")


all_docs=TextLoader("txt.txt").load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chunk=splitter.split_documents(all_docs)
db=FAISS.from_documents(
chunk,
embedding
)

base_r=db.as_retriever(
    search_kwargs={'k':3}
)

qa=RetrievalQA.from_chain_type(
    llm=model,
    retriever=base_r

)


st.set_page_config(page_title="Travel Guide Chatbot",page_icon="🌍")
st.title("🌍 Travel Guide Chatbot")
st.write("Ask me anything about different countries!")

if "history" not in st.session_state:
    st.session_state.history=[]


user_query=st.chat_input("Type your travel question...")
if user_query:
    answer=qa.run(user_query)
     

    st.session_state.history.append(("user",user_query))
    st.session_state.history.append(("bot", answer))
for role,message in st.session_state.history:
    if role=="user":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")        