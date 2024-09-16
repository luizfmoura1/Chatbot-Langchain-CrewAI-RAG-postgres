# app.py

import os
import streamlit as st
from utils.pdf_loader import carregar_pdfs
from utils.text_processing import processar_texto
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Recupera a chave da API da OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    # Configuração da página
    st.set_page_config(page_title="💬 Chatbot", page_icon="🤖")
    st.title("💬 Mike-Gpt")
    st.caption("🚀 Pergunte para nossa IA especialista em Zoppy")

    # Inicializa o histórico de conversas no estado da sessão
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]
    
    # Exibe o histórico de conversas
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada de texto do usuário
    user_input = st.chat_input("Você:")

    if user_input:
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Inicializa o objeto de embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Carrega o vetorstore existente ou cria um novo
        if os.path.exists('vectorstore/faiss_index'):
            vetorstore = FAISS.load_local('vectorstore/faiss_index', embeddings, allow_dangerous_deserialization=True)
        else:
            vetorstore = criar_vetorstore(embeddings)

        # Inicializa a memória da conversa
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Configura a cadeia de conversação com recuperação
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.2,
                model_name="gpt-4"

            ),
            retriever=vetorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            verbose=True
        )

        # Executa a consulta e obtém a resposta
        resposta = qa({"question": user_input})

        # Adiciona a resposta do chatbot ao histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta['answer']})
        st.chat_message("assistant").write(resposta['answer'])

def criar_vetorstore(embeddings):
    # Carrega e processa o texto dos PDFs
    textos = carregar_pdfs('data/')  # Ajuste o caminho conforme necessário
    chunks = processar_texto(textos)
    
    # Cria o vetorstore usando FAISS e embeddings
    vetorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Salva o índice FAISS localmente para reutilização futura
    vetorstore.save_local('vectorstore/faiss_index')
    
    return vetorstore

if __name__ == "__main__":
    main()
