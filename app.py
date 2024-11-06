import os
import streamlit as st
import psycopg2
import redis
from utils.text_processing import processar_texto
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
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
Você é um assistente virtual especializado em ajudar usuários com dúvidas relacionadas ao banco de dados vinculado. Sempre que possível, baseie suas respostas nas informações presentes no banco de dados Postgres vinculado para garantir que as respostas sejam precisas e atualizadas.
Lembre-se seja sempre direto e objetivo em suas respostas, fornecendo instruções claras e concisas para ajudar o usuário a resolver seu problema. Você DEVE responder apenas perguntas relacionadas ao banco de dados Postgres vinculado! Você deve responder as perguntas relacionadas ao banco, e não solicitar uma query!!

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

# Configuração do agente SQL com CrewAI
def configurar_agente_sql(embeddings):
    actor_schema_info = get_actor_schema()  # Obtém as colunas da tabela "actor"

    sql_developer_agent = Agent(
        role='Senior SQL developer',
        goal="Return data from the 'actor' table by running the Execute query DB tool.",
        backstory=f"""Você está conectado ao banco de dados que contém a tabela 'actor' com as seguintes colunas: {actor_schema_info}.
                      Use o Execute query DB tool para realizar consultas específicas nesta tabela e retornar os resultados.""",
        tools=[run_query],
        allow_delegation=False,
        verbose=True,
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")
    )
    sql_developer_task = Task(
        description="""Construir uma consulta SQL para responder a pergunta: {question} usando a tabela 'actor'. Formate todos os nomes que forem inseridos inteiramente formado de letras maiusculas""",
        expected_output="""
        Preciso que o output seja uma resposta formatada para responder a pergunta, de acordo com os dados encontrados pela tool de query no banco.
        Preciso que quando for formatado a resposta, todos os nomes inseridos sejam formatados com apenas a primeira letra em maiusculo.
        Caso tenha mais de um sobrenome para um primeiro nome ou o contrário, significa que existem mais de um ator com aquele primeiro nome, informe isto na resposta.
        Caso tenha datas de último update iguais, você deve mostrar apenas do ator que for mencionado, APENAS SE FOR REQUISITADO.
        """,
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

# Criar índice de embeddings no Redis
def criar_indice_redis(redis_client):
    idx = redis_client.ft(index_name="idx:embeddings")
    try:
        idx.info()
    except Exception as e:
        print(f"Erro ao verificar índice: {e}")
        idx.create_index(
            fields=[
                VectorField("embedding", "FLAT", {"DIM": 128}),  # Somente tipo e dimensão
                TextField("content")
            ],
            definition=IndexDefinition(prefix=["emb:"], index_type=IndexType.HASH)
        )



# Armazenar embeddings no Redis
def armazenar_embeddings_redis(redis_client, embeddings, textos):
    for idx, chunk in enumerate(textos):
        embedding_vector = embeddings.embed_text(chunk)
        redis_client.hset(
            f"emb:{idx}",
            mapping={
                "embedding": embedding_vector,  # Vetor de embedding
                "content": chunk  # Conteúdo de texto associado
            }
        )

# Função para buscar embeddings no Redis
def buscar_embeddings_redis(redis_client, query_vector):
    search_query = Query(f'*=>[KNN 1 @embedding $vec]').sort_by("content").dialect(2)
    params = {"vec": query_vector}
    results = redis_client.ft("idx:embeddings").search(search_query, query_params=params)
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
        query_embedding = embeddings.embed_text(user_input)

        # Busca no Redis
        results = buscar_embeddings_redis(redis_client, query_embedding)

        # Obter e exibir o resultado do Redis
        if results.docs:
            resposta = results.docs[0].content
            st.session_state.messages.append({"role": "assistant", "content": resposta})
            st.chat_message("assistant").write(resposta)

        # Usar CrewAI para construir a resposta com o agente SQL
        crew = configurar_agente_sql(embeddings)
        
        # Configuração do ChatOpenAI conforme o Código 2
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            model_name="gpt-4o-mini",
            max_tokens=1000
        )
        
        # Utilizando o llm no kickoff do CrewAI
        result = crew.kickoff(inputs={'question': user_input, 'llm': llm})

        print("Estrutura completa do result:", vars(result))
        result = vars(result)
        st.session_state.messages.append({"role": "assistant", "content": result.get("raw")})
        st.chat_message("assistant").write(result.get("raw"))

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    main()