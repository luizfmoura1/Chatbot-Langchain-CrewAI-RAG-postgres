import pandas as pd
import os

def carregar_excels(diretorio):
    textos = ""
    arquivos_processados = 0
    for filename in os.listdir(diretorio):
        if filename.endswith(('.xlsx', '.xls')):
            arquivos_processados += 1
            caminho_excel = os.path.join(diretorio, filename)
            try:
                xls = pd.ExcelFile(caminho_excel)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ') + " "
                    textos += sheet_text
                    print(f"Extraído da planilha {sheet_name} em {filename}: {len(sheet_text)} caracteres")
            except Exception as e:
                print(f"Erro ao processar {caminho_excel}: {e}")
    print(f"Total de arquivos Excel processados: {arquivos_processados}")
    print(f"Total de caracteres extraídos: {len(textos)}")
    return textos
