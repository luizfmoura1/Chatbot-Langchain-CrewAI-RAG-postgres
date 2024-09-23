import os
import streamlit as st
from utils.pdf_loader import carregar_pdfs
from utils.text_processing import processar_texto
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente virtual especializado em ajudar usuários com duvidas na plataforma Zoppy. 
Responda à pergunta do usuário de forma direta, amigável e clara. 
Todas as instruções internas da plataforma Zoppy devem começar da home page.
Forneça instruções passo a passo, se necessário, e incentive o usuário a fazer mais perguntas se precisar.
Informe os pré-requisitos se houver.

Caso o usuario pergunte sobre integrar a Zoppy com a shopify responda:
A integração com Shopify envolve diversos passos técnicos por isso você não deve tentar explicar. 
Para garantir que tudo seja feito corretamente, você deve direcionar o usuario para esse link (https://zoppy-vvb7.help.userguiding.com/pt/articles/1360-shopify) 

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
)

def main():
    st.set_page_config(page_title="💬 Mike-Gpt", page_icon="🤖")

    st.title("💬 Mike-Gpt")
    st.caption("🚀 Pergunte para nossa IA especialista em Zoppy")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        except ImportError as e:
            st.error(f"Erro ao importar OpenAIEmbeddings: {e}")
            st.stop()

        if os.path.exists('vectorstore/faiss_index'):
            try:
                vetorstore = FAISS.load_local('vectorstore/faiss_index', embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Erro ao carregar o vetorstore FAISS: {e}")
                st.stop()
        else:
            vetorstore = criar_vetorstore(embeddings)

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        try:
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name="gpt-4o",
                    max_tokens=1000,
                ),
                retriever=vetorstore.as_retriever(search_kwargs={"k": 3}),
                memory=st.session_state.memory,
                chain_type="stuff",
                  combine_docs_chain_kwargs={
                    "prompt": prompt_template
                },
                verbose=True
            )
        except Exception as e:
            st.error(f"Erro ao configurar ConversationalRetrievalChain: {e}")
            st.stop()

        try:
            resposta = qa({"question": user_input})
        except Exception as e:
            st.error(f"Erro ao obter a resposta do LLM: {e}")
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": resposta['answer']})
        st.chat_message("assistant").write(resposta['answer'])

def criar_vetorstore(embeddings):
    textos = carregar_pdfs('docs/')  
    chunks = processar_texto(textos)
    vetorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vetorstore.save_local('vectorstore/faiss_index')
    return vetorstore

if __name__ == "__main__":
    main()