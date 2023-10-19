# Perfil_Autom-tico_Candidato
A partir de um input como "Procuro um profissional dedicado com capacidade analítica e experiência no varejo, que domine Pyhton, SQL, Excel, PySpark e algoritmos de machine learning." O código encontra os perfis ideiais de cadidatos usando NLP e similaridade de cossenos.

## Modelagem do processamento de palavras

Primeiramente, a base de dados utilziada foi extraída do site Apify e pode ser encontrada no link [Linkedin Companies & Profiles Bulk Scraper](https://console.apify.com/actors/od6RadQV98FOARtrp/console). Ela utiliza um ator pago, mas é possível fazer um teste gratis. Pelos próprios parâmetros é possível colocar alguns filtros de assunto e localidade, portanto utilizei "Data Science" e "Ciência de dados" como filtro de assunto e "Brasil" como filtro de localidade. Foram extraídos 2955 perfis, valor que pode ser alterado dependendo da disponibilidade e preferência do usuário.

Iniciamos a modelagem importando os pacotes que foram utilziados no código:

```
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import RSLPStemmer
import re
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from unidecode import unidecode
from scipy.spatial import distance
```
Importando a base e chamando as funções stopwords e RSLPStemmer() do pacote NLTK. O pacote stopwords consiste em palavras que não tem valor analítico, como artigos e conectivos e RSLPStemmer() realiza o processo de stemmização, ou seja, utiliza apenas a raiz das palavras, utilizando, por exemplo, "prog" para referenciar "programador" ou "programação".

```
base_linkedin = pd.read_excel('base_linkedin.xlsx')

sem_stop_words = set(stopwords.words('portuguese'))
Stem_ptbr = RSLPStemmer()
```
### Tratando descrição

Primeiramente, será realizado o processo de tratamento e tokenização das descrições do candidato, pois tem grande valor de informação no perfil do linkedin. A coluna no dataset correspondente é "description/pt_BR".

```
def tratamento_palavras(col):
    lista_descricoes_tratadas = []

    stemmer = RSLPStemmer()

# Para cada linha no dataset aplicará o tratamento

    for index, row in base_linkedin.iterrows():
        bloco = row[col]
        bloco_texto = str(bloco)

        # Removendo caracteres especiais e tokenizando as palavras
        tokens = word_tokenize(re.sub(r'[^\w\s]', '', bloco_texto), language='portuguese')

        # Removendo as stop words e stemmizando
        palavras_stemmizadas = [stemmer.stem(unidecode(token).lower()) for token in tokens if token not in sem_stop_words]

        lista_descricoes_tratadas.append(palavras_stemmizadas)

    # Criar um DataFrame com uma coluna chamada 'Tokens' que contém as listas
    df_stem = pd.DataFrame({'Tokens': lista_descricoes_tratadas})

    return df_stem
```
Para chamar a função basta passar o nome da coluna. O resultado é um DataFrame pandas contendo uma lista de tokens em cada linha.

### Tratando habilidades

No caso das habilidades do candidato, elas estão distribuídas ao longo de 20 colunas nomeadas f"skills/{i}" onde i varia de 0 a 19.

```
def tratamento_skill():
    bag_of_skills = []
    stemmer = RSLPStemmer()

    for index, row in base_linkedin.iterrows():
        all_skills = []

# Itera sobre as colunas de habildiades

        for i in range(20):
            skill = row[f"skills/{i}"]

            if not pd.isna(skill) and isinstance(skill, str):
                skill = skill.lower() # Tudo em letra minúscula
                skill = re.sub(r'[^a-zA-Z0-9\s]', '', skill) # Tira acentos e caracteres especiais
                skill_tokens = word_tokenize(skill)
                skill_stemmed = [stemmer.stem(token) for token in skill_tokens]
                all_skills.extend(skill_stemmed)

        bag_of_skills.append(all_skills)

    df_bag_of_skills = pd.DataFrame({'merged_skills': bag_of_skills}) # Junta a informação de todas as colunas em uma única lista em uma coluna

    return df_bag_of_skills
```

### Similaridade por cossenos

Uma métrica bastante utilizadas para calcular afinidade entre conjuntos de palavras é a similaridade por cossenos dos vetores de frequência desses conjuntos. Portanto, primeiro iremos gerar esses vetores.

```
def vetores_desc():
    lista_vetor_desc = []

    for index, row in bag_descricao.iterrows():
        descricao = len(row['Tokens']) * [0]
        lista_vetor_desc.append(descricao)

    return lista_vetor_desc

def vetores_skill():
    lista_vetor_skills = []

    for index, row in bag_skills.iterrows():
        skills = len(row['merged_skills']) * [0]
        lista_vetor_skills.append(skills)

    return lista_vetor_skills

```

Nas funções acima, é criado um vetor contendo zeros para cada token encontrado na linha. Já nas funções abaixo, ele preenche esses vetores zerados com as frequências de cada token encontrado com base no índice do token.

```
def preench_vetor():

    vetor1 = vetores_desc()
    vetor2 = vetores_skill()

    frase_consulta = lista_frase_stem

    for index, row in bag_descricao.iterrows():
        for palavra in frase_consulta:
            if palavra in row['Tokens']:
                index_palavra1 = row['Tokens'].index(palavra)
                for i in range(0, len(vetor1)):
                    if index_palavra1 < len(vetor1[i]):
                        vetor1[i][index_palavra1] += 1

    for index, row in bag_skills.iterrows():
        for palavra in frase_consulta:
            if palavra in row['merged_skills']:
                index_palavra2 = row['merged_skills'].index(palavra)
                for t in range(0, len(vetor2)):
                    if index_palavra2 < len(vetor2[t]):
                        vetor2[t][index_palavra2] += 1

    return vetor1, vetor2
```

A frase de input também precisa ser tratada dessa forma:

```
bag_descricao = tratamento_palavras("description/pt_BR")
bag_skills = tratamento_skill()

frase_busca = input("Qual o perfil do profissional procurado?")
lista_frase = frase_busca.split()
lista_frase_stem = [Stem_ptbr.stem(unidecode(palavra).lower()) for palavra in word_tokenize(frase_busca) if palavra not in sem_stop_words]


def vetor_busca_desc():
    list_vet1 = []
    vetor1, vetor2 = preench_vetor()
    tamanho_vetor1 = len(vetor1[0])  # Tamanho do vetor de descrição

    for index, row in bag_descricao.iterrows():
        vet = [0] * tamanho_vetor1
        for palavra in lista_frase_stem:
            if palavra in row['Tokens']:
                index_palavra1 = row['Tokens'].index(palavra)
                if index_palavra1 < tamanho_vetor1:
                    vet[index_palavra1] += 1
        list_vet1.append(vet)

    return list_vet1

def vetor_busca_skill():
    list_vet2 = []
    vetor1, vetor2 = preench_vetor()
    tamanho_vetor2 = len(vetor2[0])

    for index, row in bag_skills.iterrows():
        vet = [0] * tamanho_vetor2  # Define o tamanho do vetor igual ao de vetor2

        for palavra in lista_frase_stem:
            if palavra in row['merged_skills']:
                index_palavra2 = row['merged_skills'].index(palavra)
                if index_palavra2 < tamanho_vetor2:
                    vet[index_palavra2] += 1

        list_vet2.append(vet)
```

Dispondo dos dados tratados, criaremos a função do cálculo do cosseno. No entanto, o cálculo só funciona com vetores de mesmo tamanho, portanto a função deve conter cláusulas para igualar os vetores gerados pelas habilidades e descrição com os vetores gerados pelas frases de input, adicionando zeros ao menor até igualar.

```
def calc_sim_coss():
    vetor1, vetor2 = preench_vetor()
    vet_b_d = vetor_busca_desc()
    vet_b_s = vetor_busca_skill()

    dist1 = []
    dist2 = []

    # Ajustar o tamanho dos vetores em vet_b_d e vetor1
    for i in range(len(vet_b_d)):
        max_len1 = max(len(vet_b_d[i]), len(vetor1[i]))

        if len(vet_b_d[i]) < max_len1:
            vet_b_d[i] += [0] * (max_len1 - len(vet_b_d[i]))

        if len(vetor1[i]) < max_len1:
            vetor1[i] += [0] * (max_len1 - len(vetor1[i]))

    # Ajustar o tamanho dos vetores em vet_b_s e vetor2
    for i in range(len(vet_b_s)):
        max_len2 = max(len(vet_b_s[i]), len(vetor2[i]))

        if len(vet_b_s[i]) < max_len2:
            vet_b_s[i] += [0] * (max_len2 - len(vet_b_s[i]))

        if len(vetor2[i]) < max_len2:
            vetor2[i] += [0] * (max_len2 - len(vetor2[i]))

    # Calcule as distâncias de similaridade cosseno para vet_b_d e vetor1
    for i in range(len(vet_b_d)):
        try:
            dist1.append(1 - distance.cosine(vet_b_d[i], vetor1[i]))
        except Exception:
            pass

    # Calcule as distâncias de similaridade cosseno para vet_b_s e vetor2
    for i in range(len(vet_b_s)):
        try:
            dist2.append(1 - distance.cosine(vet_b_s[i], vetor2[i]))
        except Exception:
            pass

    return dist1, dist2
```

## Modelagem da diferença de datas

No Dataset os meses de entrada e saída em uma empresa ou cargo estão em colunas diferentes, logo primeiro precisamos juntá-los.

```
def concat_data(mes, ano):
    data_concat = []

    for index, row in base_linkedin.iterrows():
        try:
            data_linha = str(int(round(row[mes], 0))) + "-" + str(int(round(row[ano], 0)))

            data_concat.append(data_linha)

        except ValueError:
            data_concat.append(datetime.now().strftime('%m-%Y'))

    return data_concat
```

As datas entrarão no formato M-YYYY.

Para os anos de experiência foram considerados apenas as 3 últimas empresas que os candidatos trabalharam e seus 3 últimos cargos em cada uma. A fim de automatizar a aplicação das funções foi criado um dicionário para cada empresa:

```
empresa_0 = {
    "mes start": ["positions/0/positions/0/startDate/month",
                  "positions/0/positions/1/startDate/month",
                  "positions/0/positions/2/startDate/month"],
    "ano start": ["positions/0/positions/0/startDate/year",
                  "positions/0/positions/1/startDate/year",
                  "positions/0/positions/2/startDate/year"],
    "mes end": ["positions/0/positions/0/endDate/month",
                "positions/0/positions/1/endDate/month",
                "positions/0/positions/2/endDate/month"],
    "ano end": ["positions/0/positions/0/endDate/year",
                "positions/0/positions/1/endDate/year",
                "positions/0/positions/2/endDate/year"],
}

empresa_1 = {
    "mes start": ["positions/1/positions/0/startDate/month",
                  "positions/1/positions/1/startDate/month",
                  "positions/1/positions/2/startDate/month"],
    "ano start": ["positions/1/positions/0/startDate/year",
                  "positions/1/positions/1/startDate/year",
                  "positions/1/positions/2/startDate/year"],
    "mes end": ["positions/1/positions/0/endDate/month",
                "positions/1/positions/1/endDate/month",
                "positions/1/positions/2/endDate/month"],
    "ano end": ["positions/1/positions/0/endDate/year",
                "positions/1/positions/1/endDate/year",
                "positions/1/positions/2/endDate/year"],
}

empresa_2 = {
    "mes start": ["positions/2/positions/0/startDate/month",
                  "positions/2/positions/1/startDate/month",
                  "positions/2/positions/2/startDate/month"],
    "ano start": ["positions/2/positions/0/startDate/year",
                  "positions/2/positions/1/startDate/year",
                  "positions/2/positions/2/startDate/year"],
    "mes end": ["positions/2/positions/0/endDate/month",
                "positions/2/positions/1/endDate/month",
                "positions/2/positions/2/endDate/month"],
    "ano end": ["positions/2/positions/0/endDate/year",
                "positions/2/positions/1/endDate/year",
                "positions/2/positions/2/endDate/year"],
}
```

Os valores são nomes das colunas escolhidas. Cria-se um data frame pandas.

```
empresa_0_df = pd.DataFrame(empresa_0)
empresa_1_df = pd.DataFrame(empresa_1)
empresa_2_df = pd.DataFrame(empresa_2)
```

A função abaixo cálcula a diferença entre período de entrada e saída em cada caso e para cada cargo compõe uma coluna, totalizando 3 colunas.

```
def data_diff(empresa):
    data_diff_lista = []

    for index, row in empresa.iterrows():

        mes_in = row["mes start"]
        ano_in = row["ano start"]
        mes_fim = row["mes end"]
        ano_fim = row["ano end"]

        data_inicio = concat_data(mes_in, ano_in)

        if mes_fim == "" or ano_fim == "":
            data_fim = datetime.now().strftime('%m-%Y')
        else:
            data_fim = concat_data(mes_fim, ano_fim)

        for data_in, data_f in zip(data_inicio, data_fim):

            data_inicio_data = datetime.strptime(data_in, '%m-%Y')
            data_fim_data = datetime.strptime(data_f, '%m-%Y')

            data_difference = (data_fim_data - data_inicio_data)
            data_diff_lista.append(data_difference)

    df_data_diff = pd.DataFrame(data_diff_lista)

    total_linhas = len(df_data_diff)
    linhas_por_coluna = total_linhas // 3

    parte1 = df_data_diff.iloc[:linhas_por_coluna].reset_index(drop=True)
    parte2 = df_data_diff.iloc[linhas_por_coluna:2 * linhas_por_coluna].reset_index(drop=True)
    parte3 = df_data_diff.iloc[2 * linhas_por_coluna:].reset_index(drop=True)

    df_data_diff_final = pd.concat([parte1, parte2, parte3], axis=1)
    df_data_diff_final.columns = ['Cargo 0', 'Cargo 1', 'Cargo 2']
    return df_data_diff_final
```

Por fim, define-se uma função para somar os tempos das 3 colunas e calcular o tempo total de experiência do profissional.

```
def tempo_por_empresa():

    t_empresa_0 = data_diff(empresa_0_df)
    t_empresa_1 = data_diff(empresa_1_df)
    t_empresa_2 = data_diff(empresa_2_df)

    tempo_por_empresa = {
        "E0": (t_empresa_0["Cargo 0"] + t_empresa_0["Cargo 1"] + t_empresa_0["Cargo 2"]),
        "E1": (t_empresa_1["Cargo 0"] + t_empresa_1["Cargo 1"] + t_empresa_1["Cargo 2"]),
        "E2": (t_empresa_2["Cargo 0"] + t_empresa_2["Cargo 1"] + t_empresa_2["Cargo 2"]),
    }

    return tempo_por_empresa

def exp_prof():

    exp = tempo_por_empresa()
    df_exp = pd.DataFrame(exp)

    df_tempo_in_days = df_exp.apply(lambda col: col.map(lambda x: x.days if pd.notna(x) else 0))

    df_tempo_in_days['Total Dias'] = df_tempo_in_days[["E0", "E1", "E2"]].sum(axis=1)

    df_tempo_in_days['Total Anos'] = round(df_tempo_in_days['Total Dias'] / 360, 1)

    df_exp_anos = df_tempo_in_days[['Total Anos']]

    return df_exp_anos
```

## Criação do Data Frame consolidado

O entregável é uma tabela um excel ordenada pela média de similaridade de cossenos entre descrição e habildiades com a frase de inpute também contendo o tempo de experiência profissional, URL, cargo e empresa atual.
Calcula-se as médias:

```
def media():

    m1, m2 = calc_sim_coss()

    media = [(a + b) / 2 for a, b in zip(m1, m2)]

    return media
```

Consolida o Data Frame final:

```
def consolidado():
    dist1, dist2 = calc_sim_coss()
    anos_exp = exp_prof()

    dic_consolidado = {
        "URL": base_linkedin["url"],
        "Cargo Atual": base_linkedin["positions/0/positions/0/title"],
        "Empresa": base_linkedin["positions/0/companyName"],
        "Anos de experiência": anos_exp['Total Anos'],
        "Similaridade descrição": dist1,
        "Similaridade habilidades": dist2,
        "Média similaridades": media()
    }

    df_consolidado = pd.DataFrame(dic_consolidado)
    df_condolidado_ordenado = df_consolidado.sort_values(by="Média similaridades", ascending=True)
    df_condolidado_ordenado.to_excel("consolidado_sorted.xlsx", index=False)

    return df_condolidado_ordenado
```




