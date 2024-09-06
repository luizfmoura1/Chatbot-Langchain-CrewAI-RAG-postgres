from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()  

def create_embeddings(chunks):
   
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Annoy.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, max_tokens=10000, top_p=1, model_name=os.getenv('OPENAI_MODEL_NAME'))  
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain
