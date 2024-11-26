from langchain.text_splitter import CharacterTextSplitter

def processar_texto(texto):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(texto)
    print(f"Texto dividido em {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:100]}...")  # Imprime os primeiros 100 caracteres para depuração
    return chunks


