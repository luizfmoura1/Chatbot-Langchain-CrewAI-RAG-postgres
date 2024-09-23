# app.py

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

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Recupera a chave da API da OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente virtual especializado em ajudar usuários com duvidas na plataforma Zoppy. 
Responda à pergunta do usuário de forma direta, amigável e clara. 
Todas as instruções internas da plataforma Zoppy devem começar da home page.
Forneça instruções passo a passo, se necessário, e incentive o usuário a fazer mais perguntas se precisar.
Informe os pré-requisitos se houver.

A integração com Shopify envolve diversos passos técnicos por isso você não deve tentar explicar. 
Para garantir que tudo seja feito corretamente, você deve direcionar o usuario para esse link (https://zoppy-vvb7.help.userguiding.com/pt/articles/1360-shopify) 

Contexto:
{context}

Pergunta:
{question}
sh
Resposta:
"""
)


def main():
    # Configuração da página
    st.set_page_config(page_title="💬 Mike-Gpt", page_icon="🤖")
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
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        except ImportError as e:
            st.error(f"Erro ao importar OpenAIEmbeddings: {e}")
            st.stop()

        # Carrega o vetorstore existente ou cria um novo
        if os.path.exists('vectorstore/faiss_index'):
            try:
                vetorstore = FAISS.load_local('vectorstore/faiss_index', embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Erro ao carregar o vetorstore FAISS: {e}")
                st.stop()
        else:
            vetorstore = criar_vetorstore(embeddings)

        # Inicializa a memória da conversa
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Configura a cadeia de conversação com recuperação
        try:
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name="gpt-4o",
                    max_tokens=1000,
                ),
                retriever=vetorstore.as_retriever(search_kwargs={"k": 5}),
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

        # Executa a consulta e obtém a resposta
        try:
            resposta = qa({"question": user_input})
        except Exception as e:
            st.error(f"Erro ao obter a resposta do LLM: {e}")
            st.stop()

        # Adiciona a resposta do chatbot ao histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta['answer']})
        st.chat_message("assistant").write(resposta['answer'])

def criar_vetorstore(embeddings):
    # Carrega e processa o texto dos PDFs
    textos = carregar_pdfs('docs/')  
    chunks = processar_texto(textos)
    
    # Cria o vetorstore usando FAISS e embeddings
    vetorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Salva o índice FAISS localmente para reutilização futura
    vetorstore.save_local('vectorstore/faiss_index')
    
    return vetorstore

if __name__ == "__main__":
    main()
