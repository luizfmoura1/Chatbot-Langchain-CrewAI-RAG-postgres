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

def configurar_agente_sql():
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
        role='Postgres analyst senior',
        goal="Sua função é fazer query no banco de dados referente a dados encontrados na table actor, quando necessário, de acordo com o pedido do usuário.",
        backstory=f"""Você está conectado ao banco de dados que contém a table 'actor' com as seguintes colunas: {actor_schema_info},
        Para perguntas referentes ao banco de dados utilize a sua tool para fazer a busca no mesmo,
        Caso a pergunta seja algo fora do tema principal, retorne uma resposta baseado em seu conhecimento geral.""",
        tools=[run_query],
        allow_delegation=False,
        verbose=True,
        llm=llm
        )

    # Definição da tarefa
    sql_developer_task = Task(
        description="""Construir uma consulta no banco para responder a pergunta: {question}, caso a pergunta seja referente a table actor do banco de dados.
        Caso a pergunta seja fora do tema do banco, apenas responda o usuário com seu conhecimento geral""",
        expected_output="Caso a pergunta seja referente ao banco, preciso de uma resposta formulada e baseada nos dados obtidos pela query, preciso apenas do nome do ator. Caso ocorra uma pergunta que não tenha relação com a table actor do banco de dados vinculado a você, responda com seus conhecimentos gerais e ao fim traga diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. Se você encontrar a resposta no banco de dados, responda apenas a pergunta de forma um pouco elaborada, sem lembrar sua função no final.",
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
        # Verificar se o embedding já existe no Redis
        if redis_client.exists(f"emb:{idx}"):
            print(f"Embedding emb:{idx} já existe, pulando...")
            continue

        # Gerar o embedding para o novo chunk de texto
        embedding_vector = embeddings.embed_query(chunk)
        embedding_vector_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()

        # Armazenar o novo embedding no Redis
        redis_client.hset(
            f"emb:{idx}",
            mapping={
                "embedding": embedding_vector_bytes,
                "content": chunk
            }
        )
        print(f"Novo embedding emb:{idx} armazenado com sucesso.")


# Função para buscar embeddings no Redis
def buscar_embeddings_redis(redis_client, query_vector, k=3):
    search_query = Query(f'*=>[KNN {k} @embedding $vec]').sort_by("content").dialect(2)

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
    st.title("OppemBOT 🤖")
    st.caption("🚀 Pergunte para nossa IA especialista da Oppem")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]
    
    redis_client = conectar_redis()
    criar_indice_redis(redis_client)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Verificar se embeddings estão carregados no Redis
    if redis_client.exists("emb:0") == 0:
        textos = carregar_dados_postgresql()
        chunks = processar_texto(textos)
        armazenar_embeddings_redis(redis_client, embeddings, chunks)

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

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
            crew = configurar_agente_sql()
            result = crew.kickoff(inputs={'question': user_input})
            print("Estrutura completa do result:", vars(result))
            result = vars(result)
            st.session_state.messages.append({"role": "assistant", "content": result.get("raw")})
            st.chat_message("assistant").write(result.get("raw"))

if __name__ == "__main__":
    main()