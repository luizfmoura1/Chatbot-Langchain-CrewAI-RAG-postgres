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

def configurar_agente_sql(chat_history=None):
    actor_schema_info = get_actor_schema()

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name="gpt-4o-mini",
        max_tokens=1000
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Atualizar a memória com o histórico do Streamlit, se existir
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    sql_developer_agent = Agent(
        role='Postgres analyst senior',
        goal="Sua função é fazer query no banco de dados referente a dados encontrados na table actor, quando necessário, de acordo com o pedido do usuário.",
        backstory=f"""Você está conectado ao banco de dados que contém a table 'actor' com as seguintes colunas: {actor_schema_info},
        Para perguntas referentes ao banco de dados utilize a sua tool para fazer a busca no mesmo,
        Caso a pergunta seja algo fora do tema principal, retorne uma resposta baseado em seu conhecimento geral,
        Você deve se lembrar de perguntas anteriores e utiliza-las como contexto para outras perguntas.""",
        tools=[run_query],
        allow_delegation=False,
        verbose=True,
        llm=llm,
        memory=memory
    )

    sql_developer_task = Task(
        description="""Construir uma consulta no banco para responder a pergunta: {question}, caso a pergunta seja referente a table actor do banco de dados.
        Caso a pergunta seja fora do tema do banco, apenas responda o usuário com seu conhecimento geral.
        Você deve se lembrar de perguntas anteriores e utiliza-las como contexto para outras perguntas.""",
        expected_output="Caso a pergunta seja referente ao banco, preciso de uma resposta formulada e baseada nos dados obtidos pela query, preciso apenas do nome do ator. Caso ocorra uma pergunta que não tenha relação com a table actor do banco de dados vinculado a você, responda com seus conhecimentos gerais e ao fim traga diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. Se você encontrar a resposta no banco de dados, responda apenas a pergunta de forma um pouco elaborada, sem lembrar sua função no final.",
        agent=sql_developer_agent
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

def verificar_e_deletar_indice(redis_client, index_name="idx:embeddings", expected_dim=24576):
    try:
        idx = redis_client.ft(index_name)
        info = idx.info()

        # Acesso à dimensão do índice (corrigido para a estrutura mais recente do Redis-py)
        attributes = info.get('attributes', [])
        for attr in attributes:
            if attr['identifier'] == 'embedding':
                embedding_dim = attr['DIM']
                if embedding_dim != expected_dim:
                    print(f"Dimensão incorreta encontrada ({embedding_dim}). Apagando o índice...")
                    idx.dropindex(delete_documents=False)
                    return True  # Índice foi deletado
                print("Índice encontrado com a dimensão correta.")
                return False  # Índice não foi deletado

        print("Não foi possível encontrar o campo 'embedding' nas informações do índice.")
        return True  # Deletar se o campo não for encontrado
    except Exception as e:
        print(f"Erro ao verificar o índice: {e}. Provavelmente o índice não existe.")
        return True  # Índice não existe ou ocorreu erro



def criar_indice_redis(redis_client):
    index_name = "idx:embeddings"
    idx = redis_client.ft(index_name)

    try:
        print("Criando um novo índice com dimensão 24576...")
        idx.create_index(
            fields=[
                VectorField(
                    name="embedding",
                    algorithm="FLAT",
                    attributes={
                        "TYPE": "FLOAT32",
                        "DIM": 24576,
                        "DISTANCE_METRIC": "COSINE"
                    }
                ),
                TextField('content')
            ],
            definition=IndexDefinition(prefix=["emb:"], index_type=IndexType.HASH)
        )
        print("Novo índice criado com sucesso com dimensão 24576.")
    except Exception as e:
        if "Index already exists" in str(e):
            print("O índice já existe. Tentando deletá-lo e recriá-lo...")
            idx.dropindex(delete_documents=False)
            criar_indice_redis(redis_client)
        else:
            print("Erro ao criar o índice:", e)




# Armazenar embeddings no Redis
def armazenar_embeddings_redis(redis_client, embeddings, textos):
    for idx, chunk in enumerate(textos):
        if redis_client.exists(f"emb:{idx}"):
            print(f"Embedding emb:{idx} já existe, pulando...")
            continue

        embedding_vector = embeddings.embed_query(chunk)
        if len(embedding_vector) != 24576:
            print(f"Erro: Dimensão do embedding incorreta ({len(embedding_vector)}), esperado 24576.")
            continue

        embedding_vector_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()

        redis_client.hset(
            f"emb:{idx}",
            mapping={
                "embedding": embedding_vector_bytes,
                "content": chunk
            }
        )
        print(f"Novo embedding emb:{idx} armazenado com sucesso.")




def buscar_embeddings_redis(redis_client, embeddings, user_input, k=3):
    try:
        historico = " ".join([msg["content"] for msg in st.session_state["messages"] if msg["role"] == "user"])
        query_vector = embeddings.embed_query(user_input)

        if len(query_vector) != 24576:
            print(f"Erro: Dimensão do vetor de consulta incorreta ({len(query_vector)}), esperado 24576.")
            return None

        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        search_query = Query(f'*=>[KNN {k} @embedding $vec]').sort_by("content").dialect(2)
        params = {"vec": query_vector_bytes}

        results = redis_client.ft("idx:embeddings").search(search_query, query_params=params)

        if results is None or not hasattr(results, "docs"):
            print("Nenhum resultado encontrado na busca.")
            return None

        return results

    except Exception as e:
        print("Erro ao buscar embeddings no Redis:", e)
        return None






def main():
    st.set_page_config(page_title="💬 Chat-oppem", page_icon="🤖")
    st.title("OppemBOT 🤖")
    st.caption("🚀 Pergunte para nossa IA especialista da Oppem")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

    redis_client = conectar_redis()

    # Verificar e deletar o índice se a dimensão estiver incorreta
    indice_deletado = verificar_e_deletar_indice(redis_client)

    # Criar o índice apenas se ele foi deletado ou não existir
    if indice_deletado:
        criar_indice_redis(redis_client)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    if redis_client.exists("emb:0") == 0:
        textos = carregar_dados_postgresql()
        chunks = processar_texto(textos)
        armazenar_embeddings_redis(redis_client, embeddings, chunks)

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        results = buscar_embeddings_redis(redis_client, embeddings, user_input)

        if results and results.docs:
            resposta = results.docs[0].content
            st.session_state.messages.append({"role": "assistant", "content": resposta})
            st.chat_message("assistant").write(resposta)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Nenhum resultado encontrado no Redis. Tentando buscar no banco de dados..."})
            crew = configurar_agente_sql(chat_history=st.session_state["messages"])
            result = crew.kickoff(inputs={'question': user_input, 'chat_history': st.session_state["messages"]})
            result = vars(result)
            st.session_state.messages.append({"role": "assistant", "content": result.get("raw")})
            st.chat_message("assistant").write(result.get("raw"))


if __name__ == "__main__":
    main()