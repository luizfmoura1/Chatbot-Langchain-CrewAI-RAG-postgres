import streamlit as st
from utils import chatbot, text
from streamlit_chat import message

def main():
    st.set_page_config(page_title="Zoppy Mind", page_icon="🧠", layout="wide")

    st.title("Zoppy-Mind 🧠")
    st.caption("Tire suas dúvidas sobre o Zoppy 💬")

    # Definir o caminho do PDF
    pdf = ['C:/Users/Lenovo/Documents/zoppy/LangChain-RAG-Chatbot/docs/qg_zoppy.pdf']

    # Processar e carregar o conteúdo do PDF
    all_files = text.process_files(pdf)
    chunks = text.create_text_chunks(all_files)
    
    # Criar embeddings e cadeia de conversa (uma vez no início)
    vectorstore = chatbot.create_embeddings(chunks)
    conversation = chatbot.create_conversation_chain(vectorstore)

    user_input = st.text_input("Digite sua pergunta:")  

    if user_input:
        response = conversation({'question': user_input})['answer']

        # Exibir as mensagens no chat
        message(user_input, is_user=True)
        message(response, is_user=False)


    

if __name__ == '__main__':
    main()
