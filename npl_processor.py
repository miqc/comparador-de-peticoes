import re


def extrair_e_validar_merito(texto_completo, nome_arquivo):
    """Extrai e valida o mérito de uma petição jurídica."""
    texto_maiusculo = texto_completo.upper()
    pontuacao = 0
    
    # Critérios para garantir que o arquivo é uma petição válida
    evidencias = {
        r'\bEXCELENTÍSSIMO\b': 2, r'\bDOUTO JUÍZO\b': 2, 
        r'\bAÇÃO ORDINÁRIA\b': 2, r'\bPEDE DEFERIMENTO\b': 2
    }

    for padrao, pontos in evidencias.items():
        if re.search(padrao, texto_maiusculo):
            pontuacao += pontos

    inicio_fatos = texto_maiusculo.find("DOS FATOS")
    if inicio_fatos != -1: pontuacao += 3

    if pontuacao < 4:
        raise ValueError(f"O arquivo {nome_arquivo} não possui estrutura jurídica reconhecida.")

    # Isola o mérito (Fatos) para evitar similaridade por cabeçalho
    if inicio_fatos != -1:
        fim_fatos = texto_maiusculo.find("DO DIREITO", inicio_fatos)
        if fim_fatos == -1:
            fim_fatos = texto_maiusculo.find("DOS PEDIDOS", inicio_fatos)
        return texto_completo[inicio_fatos:fim_fatos].strip()
    
    return texto_completo[:4000].strip()
