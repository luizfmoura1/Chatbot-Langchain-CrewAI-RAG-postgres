
import os
import streamlit as st
import psycopg2
import redis
import numpy as np
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
        WHERE table_name = '{table_name}' AND table_schema = 'tenant_gerdau_com_br'
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
        temperature=0.5,
        model_name="gpt-4o-mini",
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
        goal=f"""Responder perguntas relacionadas às tabelas 'daily_report' e 'project'. 
        Você deve usar queries SQL para extrair dados dessas tabelas e combiná-los, caso necessário.
        As tabelas são relacionadas pela coluna 'project_id' na tabela 'daily_report' e a coluna 'id' na tabela 'project'.
        """,
        backstory = f"""
        Você é um analista experiente conectado a um banco de dados que contém a tabela 'tenant_gerdau_com_br.daily_report' e a "tenant_gerdau_com_br.project", com as seguintes colunas: {daily_report_schema_info, project_schema_info}.
        Seu objetivo é responder perguntas relacionadas a essas tabelas e fornecer informações claras e precisas. Utilize as ferramentas disponíveis para realizar consultas e gerar gráficos, seguindo estas diretrizes:

        1. Tema principal da tabela tenant_gerdau_com_br.daily_report:
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
            - 'approved' = aprovado
            - 'in_review' = em análise
            - 'in_approver' = em aberto

        2. Tema da tabela tenant_gerdau_com_br.project:
            - ID da obra (column id)
            - Data de execução (column executed_at)
            - Data de início (column start_at)
            - Data de finalização (end_at)
            - status da obra (column status)
            - Nome da obra (column name)
            - Código do contrato (column contract_code)
            - Nome do centro de custo/empresa respnsável (column cost_center_name)
            - Data de importação (column _import_at)
            - open = aberto
            - closed = fechado

        3. Respostas baseadas no banco de dados:
        - Utilize ferramentas para consultas ou geração de gráficos somente quando necessário.
        - Ao usar ferramentas, siga rigorosamente este formato:
            - Thought: Explique seu raciocínio.
            - Action: Nome da ferramenta (run_query ou generate_graph).
            - Action Input: Dados no formato JSON.

        4. Perguntas fora do escopo do banco:
        - Responda com seu conhecimento geral, sem mencionar a tabela diretamente.
        - Sempre relembre a função principal: responder perguntas sobre relatórios diários de obra.

        5. Uso de ferramentas:
        - Nunca reutilize uma ferramenta já utilizada na mesma interação.
        - Se não precisar de ferramentas, forneça uma resposta final no formato:
            - Thought: Resuma seu raciocínio.
            - Final Answer: Resposta clara e completa.

        6. Contexto da conversa:
        - Lembre-se de perguntas anteriores para oferecer respostas contextualizadas e coerentes.

        Seu papel é ser eficiente, preciso e fornecer respostas claras, priorizando consultas no banco de dados relacionadas à tabela 'tenant_gerdau_com_br.daily_report'.
        """,

        tools=[run_query_multi_table],
        allow_delegation=False,
        verbose=True,
        memory=memory,
        llm=llm
    )

    sql_developer_task = Task(
    description=
    """Responda à pergunta do usuário ({question}) com base nos dados disponíveis nas tabelas 'daily_report' e 'project', utilizando o contexto da conversa anterior ({chat_history}), se aplicável. Siga estas diretrizes:
    Utilize a relação entre 'daily_report.project_id' e 'project.id' para criar consultas combinadas quando necessário.
    Caso a pergunta não mencione explicitamente as tabelas, inferir com base nas colunas mencionadas.
    
    1. **Consultas ao banco de dados**:
    - Realize uma query apenas se for necessário para responder à pergunta.
    - Utilize as ferramentas disponíveis (run_query) seguindo o formato padrão:
        - Thought: Explique o raciocínio.
        - Action: Nome da ferramenta.
        - Action Input: Entrada no formato JSON.
    - Sempre considere as colunas das tabelas daily_report e project ao construir consultas.

    2. **Perguntas fora do tema do banco**:
    - Se a pergunta não estiver relacionada ao banco de dados, responda com seu conhecimento geral.
    - Não utilize ferramentas para perguntas não relacionadas as tabelas daily_report e project.
    - Não utilize a ferramenta de gerar gráficos quando a palavra "gráfico" não estiver presente na pergunta do usuário.

    3. **Saudações e perguntas gerais**:
    - Não use ferramentas para responder saudações ou perguntas genéricas.

    4. **Memória e contexto**:
    - Utilize o histórico da conversa para contexto, mas não para a detecção da palavra "gráfico".

    5. **Detecção da palavra "gráfico"**:
    - Verifique se a palavra "gráfico" está presente **apenas na pergunta atual do usuário ({question})**, sem considerar o histórico.
    - Defina `option_graph` como `True` somente se a palavra "gráfico" estiver presente na pergunta atual.
    - Caso contrário, defina `option_graph` como `False`.

    6. **Respostas**:
    - Se `option_graph` for `False`, responda à pergunta utilizando dados do banco, mas não gere gráficos.
    - Se `option_graph` for `True`, indique que um gráfico será gerado e siga o fluxo apropriado.

    Seu objetivo é fornecer respostas precisas, claras e úteis, priorizando o uso do banco de dados apenas quando necessário.
    Caso a pergunta **contenha explicitamente** a palavra "gráfico", faça uma resposta utilizando os dados encontrados pela query, como se você estivesse montando um gráfico com esses dados, sugira também o tipo de gráfico que você prefere na situação.
    """,
    expected_output="""Caso a pergunta seja referente ao banco, preciso de uma resposta que apresente todos os dados obtidos pela query formulando a resposta a partir deles. 
    Caso ocorra uma pergunta que não tenha relação com as tabelas daily_report e project do banco de dados vinculado a você, com exceção de saudações, responda com seus conhecimentos gerais e ao fim diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. 
    Se você encontrar a resposta no banco de dados, responda apenas a pergunta de elaborada, sem lembrar sua função no final.
    A consulta SQL deve incluir as tabelas relevantes. Se ambas forem necessárias, a query deve ser um JOIN entre 'daily_report' e 'project'.
    Responda à pergunta de forma apropriada, seguindo as diretrizes acima.""",
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
    cursor.execute("SELECT * FROM tenant_gerdau_com_br.daily_report")
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

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5,
        model_name="gpt-4o-mini",
    )

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

        # Detecção direta da palavra "gráfico" na entrada do usuário
        if "gráfico" in user_input.lower() or "grafico" in user_input.lower():
            graph_condition = True
        else:
            graph_condition = False

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

            if graph_condition:
                if os.path.exists('graficos/graph.png'):
                    os.remove('graficos/graph.png')
                    
                # Configurar o agente de gráficos
                graph_agent = Agent(
                    role='Graph generator',
                    goal="""Sua função é fazer um gráfico por meio de código e parar a execução quando ele for gerado.
                    ## **ATENÇÃO**
                    - Nunca se esqueça de importar a lib os 
                    - Caso o código retorne **True**, **encerre a task**
                    - A tool utilizada **deve** esperar um resultado booleano (True ou False)
                    """,
                    backstory="""Você é um programador especialista em matplotlib e plotar gráficos por código em geral""",
                    allow_code_execution=True,
                    max_execution_time=300,  # 5-minute timeout
                    max_retry_limit=3, # More retries for complex code tasks
                    function_calling_llm="gpt-4o",  # Cheaper model for tool calls
                    verbose=True,
                    llm=llm
                )

                graph_agent_task = Task(
                    description=
                    """Sua tarefa é fazer um gráfico utilizando code e matplotlib para as informações a seguir:
                    {infos}
                    -----
                    Quero que as informações do gráfico sejam em português-BR, e o dpi da imagem deve ser 90dpi.
                    E neste código salve o gráfico como png no caminho graficos/graph.png e faça uma verificação com esse seguinte código:
                    
                    ```python
                    import os

                    if os.path.exists(caminho):
                        return True
                    else:
                        return False
                    ```
                    """,
                    expected_output="""É esperado um gráfico com as informações solicitadas, e um valor booleano sinalizando se o gráfico foi criado no caminho verificado.""",
                    agent=graph_agent,
                )

                graph_crew = Crew(
                    agents=[graph_agent],
                    tasks=[graph_agent_task],
                    verbose=True
                )


                # Executar o agente para gerar o gráfico
                graph_crew.kickoff(inputs={'infos': result.get('raw')})

                graph_result = os.path.exists('graficos/graph.png')

                if graph_result:
                    st.chat_message("assistant").image('graficos/graph.png', caption="Aqui está a imagem solicitada!")

            else:
                # Verifique se result foi definido e não é None antes de tentar acessar
                if result is not None:
                    resposta = result.get("raw")
                    st.session_state["messages"].append({"role": "assistant", "content": resposta})
                    st.chat_message("assistant").write(resposta)
                else:
                    # Mensagem de erro clara somente para o usuário
                    resposta = "Desculpe, não consegui encontrar a resposta no momento."
                    st.session_state["messages"].append({"role": "assistant", "content": resposta})
                    st.chat_message("assistant").write(resposta)




if __name__ == "__main__":
    main()