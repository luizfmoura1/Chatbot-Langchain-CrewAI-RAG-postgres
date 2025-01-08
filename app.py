import os
import streamlit as st
import psycopg2
import numpy as np
from utils.text_processing import processar_texto
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Conexão com o PostgreSQL
def conectar_postgresql():
    try:
        connection = psycopg2.connect(
            host='localhost',
            database='gerdau',
            user='luiz',
            password='CgvQTiyXXEN7xSnsMHBkT5NW2MaxtC',
            port=5432
        )
        print("Conexão com o PostgreSQL estabelecida com sucesso.")
        return connection
    except Exception as e:
        st.error(f"Erro ao conectar ao PostgreSQL: {e}")
        st.stop()

def get_table_schema(table_name):
    """Obtém o esquema (colunas) de uma tabela específica no PostgreSQL."""
    connection = conectar_postgresql()
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}' AND table_schema = 'tenant_aperam'
    """)
    columns = cursor.fetchall()
    cursor.close()
    connection.close()
    return [column[0] for column in columns]


@tool("Execute multi-table query")
def run_query_multi_table(query: str):
    """Executa uma query SQL envolvendo múltiplas tabelas."""
    connection = conectar_postgresql()
    cursor = connection.cursor()
    print(f"Executing query: {query}")
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


def configurar_agente_sql(chat_history=None):
    daily_report_schema_info = get_table_schema("daily_report")
    project_schema_info = get_table_schema("project")

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
        model_name="gpt-4o-mini",
        max_tokens=1000
    )

    # Inicializar a memória apenas uma vez na sessão
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = st.session_state["memory"]

    # Atualizar a memória com o histórico de mensagens
    if st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    sql_developer_agent = Agent(
        role='Postgres analyst senior',
        goal="""Responder perguntas relacionadas às tabelas 'daily_report' e 'project'. 
        Você deve usar queries SQL para extrair dados dessas tabelas e combiná-los, caso necessário.""",
        backstory="""Você é um analista experiente, conectado ao banco de dados PostgreSQL, com o objetivo de responder perguntas 
        relacionadas às tabelas 'daily_report' e 'project'. Você realiza consultas SQL e fornece informações claras e precisas.""",
        tools=[run_query_multi_table],
        allow_delegation=False,
        verbose=True,
        memory=memory,
    )   


    sql_developer_task = Task(
        description="Responda à pergunta do usuário com base nos dados disponíveis nas tabelas 'daily_report' e 'project'.",
        expected_output="Resposta baseada nos dados do banco de dados.",
        agent=sql_developer_agent,
    )

    crew = Crew(
        agents=[sql_developer_agent],
        tasks=[sql_developer_task],
        process=Process.sequential,
        verbose=True
    )

    return crew


# Carregar dados do PostgreSQL
def carregar_dados_postgresql():
    connection = conectar_postgresql()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM tenant_aperam.daily_report")
    textos = " ".join([" ".join(map(str, row)) for row in cursor.fetchall()])
    cursor.close()
    connection.close()
    return textos


# Main com integração do CrewAI
def main():
    st.set_page_config(page_title="💬 Chat-oppem", page_icon="🤖")
    st.title("OppemBOT 🤖")
    st.caption("🚀 Pergunte para nossa IA especialista da Oppem")

    # Inicializar mensagens na sessão, se ainda não estiverem configuradas
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

    # Exibir todas as mensagens do histórico na conversa
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Configuração do agente SQL
        try:
            crew = configurar_agente_sql(chat_history=st.session_state["messages"])
            result = crew.kickoff(inputs={'question': user_input, 'chat_history': st.session_state["messages"]})
            resposta = vars(result).get("raw")
        except Exception as e:
            resposta = f"Desculpe, ocorreu um erro ao processar sua solicitação: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": resposta})
        st.chat_message("assistant").write(resposta)


if __name__ == "__main__":
    main()
