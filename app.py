import os
import streamlit as st
import psycopg2
import redis
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from utils.text_processing import processar_texto
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
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
            database='gerdau',
            user='gerdau',
            password='gerdau',
            port=6432
        )
        print("Conexão com o PostgreSQL estabelecida com sucesso.")
        return connection
    except Exception as e:
        st.error(f"Erro ao conectar ao PostgreSQL: {e}")
        st.stop()

# Função para obter o esquema da tabela "actor"
def get_daily_report_schema():
    connection = conectar_postgresql()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'daily_report' AND table_schema = 'tenant_aperam'
        )
    """)
    exists = cursor.fetchone()[0]
    if not exists:
        raise Exception("Tabela 'daily_report' não encontrada no esquema 'tenant_aperam'.")
    
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'daily_report' AND table_schema = 'tenant_aperam'
    """)
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

    cursor.execute(query)
    result = cursor.fetchall()
    print("Resultado da query:", result)  # Exibe o resultado da query para depuração
    cursor.close()
    connection.close()
    return result




def configurar_agente_sql(chat_history=None):
    daily_report_schema_info = get_daily_report_schema()

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
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

     # Configurar o agente com a memória atualizada
    sql_developer_agent = Agent(
        role='Postgres analyst senior',
        goal="Sua função é fazer query no banco de dados referente a dados encontrados na table daily_report, quando necessário, de acordo com o pedido do usuário.",
        backstory=f"""Você está conectado ao banco de dados que contém a tabela 'tenant_aperam.daily_report' com as seguintes colunas: {daily_report_schema_info}.
        Para perguntas referentes ao banco de dados utilize a sua tool para fazer a busca no mesmo.
        O tema principal do banco é sobre Relatórios diários de obra(column id), o dia referente a ele (column executed_at), data de criação (column created_at), id da obra (column project_id), data de aprovação (column approved_at), numéro sequencial (column sequence), quem criou o RDO (column user_username), horário de almoço (column lunch_start_time), termino do almoço (column lunch_end_time), horário do inicio do expediente (column work_start_time), horário do fim do expediente (column work_end_time), comentários (column comment), status do RDO (column status), nome de empreiteiro (column builder_name), dia de assinatura do empreiteiro (column builder_signed_at), quantidade de revisões (column revision_number) e data de importação (column _import_at)
        Caso a pergunta seja algo fora do tema principal, retorne uma resposta baseado em seu conhecimento geral, relembre da sua função e do tema do banco sem mencionar o nome da tabela.
        Você deve se lembrar de perguntas anteriores e utilizá-las como contexto para outras perguntas.""",
        tools=[run_query],
        allow_delegation=False,
        verbose=True,
        memory=memory
    )

    sql_developer_task = Task(
        description="""Construir uma consulta no banco para responder a pergunta: {question}, caso necessário considerando o contexto da conversa anterior: {chat_history} caso a pergunta seja referente a table daily_report do banco de dados.
        Você deve realizar a query apenas se for necessário, saudações e perguntas não referentes ao tema do banco de dados não são necessárias o uso de querys.
        Caso a pergunta seja fora do tema do banco, apenas responda o usuário com seu conhecimento geral.""",
        expected_output="Caso a pergunta seja referente ao banco, preciso de uma resposta que apresente todos os dados obtidos pela query formulando a resposta a partir deles. Caso ocorra uma pergunta que não tenha relação com a table daily_report do banco de dados vinculado a você, com exessão de saudações, responda com seus conhecimentos gerais e ao fim traga diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. Se você encontrar a resposta no banco de dados, responda apenas a pergunta de forma um pouco elaborada, sem lembrar sua função no final.",
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
    cursor.execute("SELECT * FROM tenant_aperam.daily_report")
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
    idx = redis_client.ft(index_name="idx:embeddings")

    try:
        info = idx.info()
        embedding_dim = info.get("attributes", [{}])[0].get("DIM", None)
        if embedding_dim != 1536:
            print(f"Índice existente com dimensão {embedding_dim}. Apagando e recriando...")
            idx.dropindex(delete_documents=False)
            raise Exception("Índice recriado devido a dimensão incorreta.")
        print("Índice existente encontrado com dimensão correta.")
    except Exception as e:
        print(f"Erro ao verificar ou criar índice: {e}. Criando um novo índice...")
        try:
            idx.create_index(
                fields=[
                    VectorField(
                        name="embedding",
                        algorithm="FLAT",
                        attributes={
                            "TYPE": "VECTOR",
                            "DIM": 1536,
                            "DISTANCE_METRIC": "COSINE"
                        }
                    ),
                    TextField("content")
                ],
                definition=IndexDefinition(prefix=["emb:"], index_type=IndexType.HASH)
            )
            print("Novo índice criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar índice no Redis: {e}")







def armazenar_embeddings_redis(redis_client, embeddings, textos):
    for idx, chunk in enumerate(textos):
        # Verificar se o embedding já existe no Redis
        if redis_client.exists(f"emb:{idx}"):
            print(f"Embedding emb:{idx} já existe, pulando...")
            continue

        # Verificar se o chunk é válido
        if not chunk.strip():
            print(f"Chunk vazio ou inválido no índice {idx}, ignorando...")
            continue

        # Gerar o embedding para o novo chunk de texto
        try:
            embedding_vector = embeddings.embed_query(chunk)
        except Exception as e:
            print(f"Erro ao gerar embedding para o chunk {idx}: {e}")
            continue

        # Validar dimensão do embedding
        if len(embedding_vector) != 1536:
            print(f"Erro: Dimensão do embedding incorreta ({len(embedding_vector)}) no chunk {idx}, esperado 1536.")
            continue

        embedding_vector_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()

        # Armazenar o novo embedding no Redis
        try:
            redis_client.hset(
                f"emb:{idx}",
                mapping={
                    "embedding": embedding_vector_bytes,
                    "content": chunk
                }
            )
            print(f"Novo embedding emb:{idx} armazenado com sucesso.")
        except Exception as e:
            print(f"Erro ao armazenar embedding emb:{idx}: {e}")




def buscar_embeddings_redis(redis_client, embeddings, user_input, k=3):
    try:
        query_vector = embeddings.embed_query(user_input)
        if len(query_vector) != 1536:
            print(f"Erro: Dimensão do vetor de consulta incorreta ({len(query_vector)}), esperado 1536.")
            return None

        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        search_query = Query(f"*=>[KNN {k} @embedding $vec]").sort_by("content").dialect(2)
        params = {"vec": query_vector_bytes}

        results = redis_client.ft("idx:embeddings").search(search_query, query_params=params)
        if results is None or not hasattr(results, "docs"):
            print("Nenhum resultado encontrado na busca.")
            return None

        print(f"{len(results.docs)} resultados encontrados no Redis.")
        for doc in results.docs:
            print(f"Resultado: {doc.id} - {doc.content[:100]}...")  # Exibe os primeiros 100 caracteres
        return results
    except Exception as e:
        print(f"Erro ao buscar embeddings no Redis: {e}")
        return None






# Main com integração do CrewAI e Redis
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

    redis_client = conectar_redis()
    criar_indice_redis(redis_client)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    
    # Verificar se embeddings estão carregados no Redis
    if redis_client.exists("emb:0") == 0:
        textos = carregar_dados_postgresql()
        chunks = processar_texto(textos)
        armazenar_embeddings_redis(redis_client, embeddings, chunks)

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Busca no Redis com histórico
        results = buscar_embeddings_redis(redis_client, embeddings, user_input)

        result = None  # Inicializando result como None

        # Verifique se resultados existem e se a lista de docs não está vazia
        if results and hasattr(results, "docs") and len(results.docs) > 0:
            resposta = results.docs[0].content
            st.session_state["messages"].append({"role": "assistant", "content": resposta})
            st.chat_message("assistant").write(resposta)
        else:
            # Mensagem de depuração apenas no console, não no chat
            print("Nenhum resultado encontrado no Redis. Tentando buscar no banco de dados...")
            
            try:
                # Tentando buscar no banco de dados usando o agente
                crew = configurar_agente_sql(chat_history=st.session_state["messages"])
                result = crew.kickoff(inputs={'question': user_input, 'chat_history': st.session_state["messages"]})
                result = vars(result)
            except Exception as e:
                print(f"Erro ao executar o agente: {e}")
                result = None  # Garantir que result seja None caso ocorra erro

            # Verifique se result foi definido e não é None antes de tentar acessar
            if result is not None:
                resposta = result.get("raw")
                st.session_state.messages.append({"role": "assistant", "content": resposta})
                st.chat_message("assistant").write(resposta)
            else:
                # Mensagem de erro clara somente para o usuário
                resposta = "Desculpe, não consegui encontrar a resposta no momento."
                st.session_state["messages"].append({"role": "assistant", "content": resposta})
                st.chat_message("assistant").write(resposta)



if __name__ == "__main__":
    main()