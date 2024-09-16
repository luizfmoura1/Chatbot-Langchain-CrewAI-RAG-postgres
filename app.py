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

# Prompt para a etapa de mapeamento
prompt_map = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente especializado em ajudar com tarefas dentro do software Zoppy. Sua resposta deve seguir este formato:

1. **Introdução**: Dê uma breve introdução ao tópico.
2. **Passos Detalhados**: Liste os passos de forma clara e concisa.
3. **Conclusão**: Forneça uma conclusão com orientações adicionais, se aplicável.

Contexto:
{context}

Pergunta:
{question}

Respostas:
"""
)

# Prompt para a etapa de redução
prompt_reduce = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
Você é um assistente especializado em ajudar com tarefas dentro do software Zoppy. Sua resposta deve seguir este formato:

1. **Introdução**: Dê uma breve introdução ao tópico.
2. **Passos Detalhados**: Liste os passos de forma clara e concisa.
3. **Conclusão**: Forneça uma conclusão com orientações adicionais, se aplicável.

Resumos:
{summaries}

Pergunta:
{question}

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
                    temperature=0.2,
                    model_name="gpt-4o",
                    max_tokens=500,
                    top_p=0.9
                ),
                retriever=vetorstore.as_retriever(search_kwargs={"k": 3}),
                memory=st.session_state.memory,
                chain_type="map_reduce",
                combine_docs_chain_kwargs={
                    "question_prompt": prompt_map,
                    "combine_prompt": prompt_reduce
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
    textos = carregar_pdfs('docs/')  # Ajuste o caminho conforme necessário
    chunks = processar_texto(textos)
    
    # Cria o vetorstore usando FAISS e embeddings
    vetorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Salva o índice FAISS localmente para reutilização futura
    vetorstore.save_local('vectorstore/faiss_index')
    
    return vetorstore

if __name__ == "__main__":
    main()
