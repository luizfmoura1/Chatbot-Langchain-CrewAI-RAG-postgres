from langchain.text_splitter import CharacterTextSplitter

def processar_texto(texto):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(texto)
    return chunks

