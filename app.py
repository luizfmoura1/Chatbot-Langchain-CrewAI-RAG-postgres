import os
import streamlit as st
import psycopg2
import redis
import json
import numpy as np
from utils.text_processing import processar_texto
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente virtual especializado em ajudar usuários com dúvidas relacionadas ao banco de dados PostgreSQL vinculado. Para perguntas genéricas ou saudações como "Olá", "Oi", "Tudo bem", responda apenas com "Olá, como posso ajudar?". Você só deve utilizar informações do banco de dados para responder perguntas diretamente relacionadas ao conteúdo do banco.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
)


# Conexão com o PostgreSQL
def conectar_postgresql():
    try:
        connection = psycopg2.connect(
            host='localhost',
            database='postgres',
            user='postgres',
            password='123456',
            port=5432
        )
        print("Conexão com o PostgreSQL estabelecida com sucesso.")
        return connection
    except Exception as e:
        st.error(f"Erro ao conectar ao PostgreSQL: {e}")
        st.stop()

# Função para obter o esquema da tabela "actor"
def get_actor_schema():
    connection = conectar_postgresql()
    cursor = connection.cursor()
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'actor'")
    columns = cursor.fetchall()
    cursor.close()
    connection.close()
    return ", ".join([column[0] for column in columns])

# Função de execução de consulta para o agente SQL com depuração
@tool("Execute query DB tool")
def run_query(query: str):
    """Execute a query no banco de dados e retorne os dados."""
    connection = conectar_postgresql()
    cursor = connection.cursor()
    print(query)
    if "WHERE first_name =" in query:
        # e converte o valor do nome para maiúsculas
        query = query.replace("WHERE first_name =", "WHERE first_name ILIKE")
    
    elif "WHERE last_name =" in query:
        query = query.replace("WHERE last_name =", "WHERE last_name ILIKE")

    cursor.execute(query)
    result = cursor.fetchall()
    print("Resultado da query:", result)  # Exibe o resultado da query para depuração
    cursor.close()
    connection.close()
    return result

def configurar_agente_sql(embeddings):
    actor_schema_info = get_actor_schema()

    # Instância do ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name="gpt-4o-mini",
        max_tokens=1000
    )

    # Criação do agente SQL
    sql_developer_agent = Agent(
        role='Senior SQL developer',
        goal="Return data from the 'actor' table by running the Execute query DB tool.",
        backstory=f"""Você está conectado ao banco de dados que contém a tabela 'actor' com as seguintes colunas: {actor_schema_info}.
                      Use o Execute query DB tool para realizar consultas específicas nesta tabela e retornar os resultados.""",
        tools=[run_query],
        allow_delegation=False,
        verbose=True,
        llm=llm  # Define o LLM diretamente aqui
    )

    # Definição da tarefa
    sql_developer_task = Task(
        description="""Construir uma consulta SQL para responder a pergunta: {question} usando a tabela 'actor'.""",
        expected_output="Resposta formatada para responder a pergunta de acordo com os dados encontrados pela query.",
        agent=sql_developer_agent
    )

    # Configuração do Crew
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
    cursor.execute("SELECT * FROM public.actor")
    textos = " ".join([" ".join(map(str, row)) for row in cursor.fetchall()])
    cursor.close()
    connection.close()
    return textos

# Conexão com o Redis
def conectar_redis():
    client =  redis.Redis(host='localhost', port=6379, db=0,)
    print(f'PING FUNCIONOU AQUI {client.ping()}')
    return client

def criar_indice_redis(redis_client):
    # Tenta remover o índice existente antes de criar um novo
    try:
        redis_client.ft("idx:embeddings").dropindex(delete_documents=True)
        print("Índice existente removido com sucesso.")
    except Exception as e:
        print("Nenhum índice existente encontrado ou erro ao remover o índice:", e)

    # Criação do novo índice com a dimensão correta
    idx = redis_client.ft(index_name="idx:embeddings")
    try:
        idx.create_index(
            fields=[
                VectorField(
                    name="embedding",
                    algorithm="FLAT",
                    attributes={
                        "TYPE": "FLOAT32",
                        "DIM": 6144,  # Altere para 6144
                        "DISTANCE_METRIC": "COSINE"
                    }
                ),
                TextField('content')
            ],
            definition=IndexDefinition(prefix=["emb:"], index_type=IndexType.HASH)
        )
        print("Novo índice criado com sucesso.")
    except Exception as e:
        print("Erro ao criar o índice:", e)


# Armazenar embeddings no Redis
def armazenar_embeddings_redis(redis_client, embeddings, textos):
    for idx, chunk in enumerate(textos):
        embedding_vector = embeddings.embed_query(chunk)
        # Converte o vetor para bytes
        embedding_vector_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()
        redis_client.hset(
            f"emb:{idx}",
            mapping={
                "embedding": embedding_vector_bytes,  # Vetor convertido para bytes
                "content": chunk  # Conteúdo de texto associado
            }
        )


# Função para buscar embeddings no Redis
def buscar_embeddings_redis(redis_client, query_vector):
    search_query = Query(f'*=>[KNN 1 @embedding $vec]').sort_by("content").dialect(2)

    # Converte o vetor de query para bytes
    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    params = {"vec": query_vector_bytes}

    results = redis_client.ft("idx:embeddings").search(search_query, query_params=params)

    # Se houver resultados, converta o embedding de volta para lista
    if results.docs:
        embedding_vector_bytes = redis_client.hget(results.docs[0].id, "embedding")
        embedding_vector = np.frombuffer(embedding_vector_bytes, dtype=np.float32).tolist()
        results.docs[0].embedding = embedding_vector  # Adiciona o vetor convertido ao resultado

    return results

# Main com integração do CrewAI e Redis
def main():
    st.set_page_config(page_title="💬 Chat-oppem", page_icon="🤖")
    st.title("💬 Chat-oppem")
    st.caption("🚀 Pergunte para nossa IA especialista em Oppem")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]
    
    redis_client = conectar_redis()
    criar_indice_redis(redis_client)

    # Verificar se embeddings estão carregados no Redis
    if redis_client.exists("emb:0") == 0:
        textos = carregar_dados_postgresql()
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chunks = processar_texto(textos)
        armazenar_embeddings_redis(redis_client, embeddings, chunks)

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_embedding = embeddings.embed_query(user_input)


        # Busca no Redis
        results = buscar_embeddings_redis(redis_client, query_embedding)

        # Verificar e exibir o resultado do Redis
        if results.docs:
            resposta = results.docs[0].content
            st.session_state.messages.append({"role": "assistant", "content": resposta})
            st.chat_message("assistant").write(resposta)
        else:
            # Somente acionar o CrewAI se não houver resultado no Redis
            crew = configurar_agente_sql(embeddings)
            result = crew.kickoff(inputs={'question': user_input})
            print("Estrutura completa do result:", vars(result))
            result = vars(result)
            st.session_state.messages.append({"role": "assistant", "content": result.get("raw")})
            st.chat_message("assistant").write(result.get("raw"))

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    main()