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
    """
    Executa uma query no banco de dados PostgreSQL e retorna os dados em formato de lista de dicionários.

    Parâmetros:
        query (str): Query SQL a ser executada no banco de dados.

    Retorna:
        list: Lista de dicionários contendo os dados da consulta.
    """
    connection = conectar_postgresql()
    cursor = connection.cursor()
    print(f"Executando query: {query}")  # Depuração da query
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]  # Obter os nomes das colunas
        rows = cursor.fetchall()
        if not rows:
            print("A query não retornou nenhum resultado.")
            return []
        result = [dict(zip(columns, row)) for row in rows]
        result = validate_and_prepare_data(result, cursor=cursor)

        print(f"Resultado da query: {result}")  # Depuração do resultado
        return result
    except Exception as e:
        print(f"Erro ao executar a query: {e}")
        return []
    finally:
        cursor.close()
        connection.close()



def validate_and_prepare_data(data, cursor=None):
    """
    Valida e prepara os dados retornados pela query para garantir que estejam no formato esperado.

    Parâmetros:
        data (list): Dados retornados pela query.
        cursor: Cursor usado para obter os nomes das colunas, se necessário.

    Retorna:
        list: Lista de dicionários representando os dados da query.
    """
    if not isinstance(data, list):
        raise ValueError("Os dados não estão em um formato de lista.")
    
    if not all(isinstance(row, dict) for row in data):
        if cursor:
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in data]
        else:
            raise ValueError("Os dados não podem ser convertidos para lista de dicionários.")
    
    return data

def detect_columns(df):
    """
    Detecta automaticamente as colunas do DataFrame para os eixos X e Y.

    Parâmetros:
        df (DataFrame): DataFrame com os dados.

    Retorna:
        tuple: Nomes das colunas para os eixos X e Y.
    """
    if df.empty or len(df.columns) < 2:
        raise ValueError("O DataFrame não contém colunas suficientes para gerar um gráfico.")
    return df.columns[0], df.columns[1]

def validate_chart_data(df, x_col, y_col):
    """
    Valida os dados do DataFrame para garantir que são adequados para a geração de gráficos.

    Parâmetros:
        df (DataFrame): DataFrame com os dados.
        x_col (str): Coluna para o eixo X.
        y_col (str): Coluna para o eixo Y.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"As colunas {x_col} e {y_col} não existem no DataFrame.")
    if df[x_col].isnull().all() or df[y_col].isnull().all():
        raise ValueError(f"As colunas {x_col} ou {y_col} contêm apenas valores nulos.")
    if len(df[x_col]) != len(df[y_col]):
        raise ValueError(f"As colunas {x_col} e {y_col} têm tamanhos incompatíveis.")



@tool("Generate Graph Tool")
def generate_graph(data, graph_type="bar", x_col=None, y_col=None):
    """
    Gera gráficos com base nos dados fornecidos.

    Parâmetros:
        data (list): Dados obtidos pela query, em forma de lista de dicionários.
        graph_type (str): Tipo de gráfico ('bar', 'line', 'pie', etc.). O padrão é 'bar'.
        x_col (str): Nome da coluna para o eixo X (opcional).
        y_col (str): Nome da coluna para o eixo Y (opcional).

    Retorna:
        str: Imagem do gráfico codificada em base64.
    """
    validate_chart_data(df, x_col, y_col)

    try:
        if not data or len(data) == 0:
            raise ValueError("A query não retornou nenhum dado.")

        
        import pandas as pd
        df = pd.DataFrame(data)

        if df.empty:
            return "Nenhum dado disponível no DataFrame para gerar o gráfico."

        # Validação de colunas
        if not x_col or not y_col:
            x_col, y_col = detect_columns(df)  # Usar as primeiras duas colunas como padrão
        if x_col not in df.columns or y_col not in df.columns:
            return f"As colunas especificadas ({x_col}, {y_col}) não existem no DataFrame."

        # Extração dos dados para os eixos
        x_data = df[x_col]
        y_data = df[y_col]
        if x_data.empty or y_data.empty:
            return "Os dados para os eixos X ou Y estão vazios."

        # Criar o gráfico
        plt.figure(figsize=(10, 6))
        if graph_type == "bar":
            plt.bar(x_data, y_data)
        elif graph_type == "line":
            plt.plot(x_data, y_data)
        elif graph_type == "pie":
            if len(y_data) > 10:
                raise ValueError("Gráficos de pizza são melhores com no máximo 10 categorias.")
            if not pd.api.types.is_numeric_dtype(y_data):
                raise ValueError("Os valores do gráfico de pizza precisam ser numéricos.")
        else:
            return f"Tipo de gráfico '{graph_type}' não suportado."
        
        plt.title(f"Gráfico do tipo {graph_type.capitalize()}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        # Salvar o gráfico como imagem e codificar em base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return base64_image
    except Exception as e:
        return f"Erro ao gerar o gráfico: {e}"





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
        goal="Sua função é fazer query no banco de dados referente a dados encontrados na table daily_report, quando necessário, de acordo com o pedido do usuário. E se for requisitado, você deve gerar umgráfico baseados nos dados obtidos pela query.",
        backstory = f"""
        Você é um analista experiente conectado a um banco de dados que contém a tabela 'tenant_aperam.daily_report', com as seguintes colunas: {daily_report_schema_info}.
        Seu objetivo é responder perguntas relacionadas a essa tabela e fornecer informações claras e precisas. Utilize as ferramentas disponíveis para realizar consultas e gerar gráficos, seguindo estas diretrizes:

        1. Tema principal do banco:
        - Relatórios diários de obra, com as seguintes colunas:
            - ID do relatório (column id)
            - Data de execução (column executed_at)
            - Data de criação (column created_at)
            - ID da obra (column project_id)
            - Data de aprovação (column approved_at)
            - Número sequencial (column sequence)
            - Usuário criador (column user_username)
            - Início e término do almoço (columns lunch_start_time, lunch_end_time)
            - Início e término do expediente (columns work_start_time, work_end_time)
            - Comentários (column comment)
            - Status do relatório (column status)
            - Nome do empreiteiro (column builder_name)
            - Data de assinatura do empreiteiro (column builder_signed_at)
            - Quantidade de revisões (column revision_number)
            - Data de importação (column _import_at)
            - approved = aprovado
            - in_review = em análise
            - in_approver = em aberto

        2. Respostas baseadas no banco de dados:
        - Utilize ferramentas para consultas ou geração de gráficos somente quando necessário.
        - Ao usar ferramentas, siga rigorosamente este formato:
            - Thought: Explique seu raciocínio.
            - Action: Nome da ferramenta (run_query ou generate_graph).
            - Action Input: Dados no formato JSON.

        3. Perguntas fora do escopo do banco:
        - Responda com seu conhecimento geral, sem mencionar a tabela diretamente.
        - Sempre relembre a função principal: responder perguntas sobre relatórios diários de obra.

        4. Uso de ferramentas:
        - Nunca reutilize uma ferramenta já utilizada na mesma interação.
        - Se não precisar de ferramentas, forneça uma resposta final no formato:
            - Thought: Resuma seu raciocínio.
            - Final Answer: Resposta clara e completa.

        5. Contexto da conversa:
        - Lembre-se de perguntas anteriores para oferecer respostas contextualizadas e coerentes.

        Seu papel é ser eficiente, preciso e fornecer respostas claras, priorizando consultas no banco de dados relacionadas à tabela 'tenant_aperam.daily_report'.
        """,

        tools=[run_query, generate_graph],
        allow_delegation=False,
        verbose=True,
        memory=memory
    )

    sql_developer_task = Task(
        description=
        """Responda à pergunta do usuário ({question}) com base no tema principal do banco de dados, utilizando o contexto da conversa anterior ({chat_history}), se aplicável. Siga estas diretrizes:

        1. **Consultas ao banco de dados**:
        - Realize uma query apenas se for necessário para responder à pergunta.
        - Utilize as ferramentas disponíveis (run_query ou generate_graph) seguindo o formato padrão:
            - Thought: Explique o raciocínio.
            - Action: Nome da ferramenta.
            - Action Input: Entrada no formato JSON.
        - Sempre considere as colunas da tabela daily_report ao construir consultas.

        2. **Geração de gráficos**:
        - Se solicitado, gere gráficos baseados nos dados obtidos pela query.
        - Verifique a integridade dos dados antes de gerar o gráfico e escolha o tipo apropriado.

        3. **Perguntas fora do tema do banco**:
        - Se a pergunta não estiver relacionada ao banco de dados, responda com seu conhecimento geral.
        - Não utilize ferramentas para perguntas não relacionadas à tabela daily_report.

        4. **Saudações e perguntas gerais**:
        - Não use ferramentas para responder saudações ou perguntas genéricas.

        5. **Memória e contexto**:
        - Utilize o histórico da conversa para formular respostas contextuais e coerentes.

        Seu objetivo é fornecer respostas precisas, claras e úteis, priorizando o uso do banco de dados apenas quando necessário.
        """,
        expected_output="Caso a pergunta seja referente ao banco, preciso de uma resposta que apresente todos os dados obtidos pela query formulando a resposta a partir deles. Caso ocorra uma pergunta que não tenha relação com a table daily_report do banco de dados vinculado a você, com exessão de saudações, responda com seus conhecimentos gerais e ao fim traga diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. Se você encontrar a resposta no banco de dados, responda apenas a pergunta de forma um pouco elaborada, sem lembrar sua função no final. Se for solicitado, formate os dados em um gráfico",
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

    # Configurar Redis e embeddings
    redis_client = conectar_redis()
    criar_indice_redis(redis_client)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    # Verificar se embeddings estão carregados no Redis
    if redis_client.exists("emb:0") == 0:
        textos = carregar_dados_postgresql()
        chunks = processar_texto(textos)
        armazenar_embeddings_redis(redis_client, embeddings, chunks)

    # Capturar entrada do usuário
    user_input = st.chat_input("Você:")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Inicializar resposta
        response = None

        # Buscar no Redis
        results = buscar_embeddings_redis(redis_client, embeddings, user_input)
        if results and hasattr(results, "docs") and len(results.docs) > 0:
            response = results.docs[0].content
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            # Mensagem de depuração
            print("Nenhum resultado encontrado no Redis. Tentando buscar no banco de dados...")

            # Buscar no banco de dados usando o agente
            try:
                crew = configurar_agente_sql(chat_history=st.session_state["messages"])
                result = crew.kickoff(inputs={'question': user_input, 'chat_history': st.session_state["messages"]})

                # Verificar se o resultado contém os dados necessários
                if hasattr(result, 'raw'):
                    response = result.raw

                    # Verificar se o usuário pediu um gráfico
                    if "gráfico" in user_input.lower():
                        if result and isinstance(result, list) and len(result) > 0:
                            graph_base64 = generate_graph(
                                data=result, 
                                graph_type="bar",  # Altere para o tipo de gráfico desejado
                                x_col="Categoria",  # Substitua pelos nomes corretos das colunas
                                y_col="Quantidade"
                            )
                            if "Erro" not in graph_base64:
                                response += f"\n\n![Gráfico](data:image/png;base64,{graph_base64})"
                            else:
                                response += f"\n\nErro ao gerar gráfico: {graph_base64}"
                        else:
                            response += "\n\nErro: Dados insuficientes para gerar o gráfico."




                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                else:
                    raise Exception("O agente não retornou nenhum resultado.")
            except Exception as e:
                response = f"Erro ao executar o agente: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()

