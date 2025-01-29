import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def LLMconnect(column_headers):
    """Takes a list of column headers as input and returns a description of the dataset based on the column headers"""

    groq_api_key = "YOUR_API_KEY"

    client = Groq(api_key=groq_api_key)

    model = "llama3-8b-8192"

    llm_content = f"""
    Given the following column headers of dataset X, suggest a description for that same dataset.
    Return only the description, no other text.

    ** COLUMN HEADERS **
    {column_headers}
    """


    messages = [
        {
            "role": "user",
            "content": llm_content,
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return (chat_completion.choices[0].message.content)

def retrieve_metadata(file_path):
    """
    Takes a CSV file as input and returns a CSV file with metadata;
    - title
    - URL
    - column_names
    - column_count
    - row_count
    - value_type
    - missing_values_count
    - value_range (if numeric)
    - column_means
    - column_medians
    - column_sd
    - description
    """

    df = pd.read_csv(file_path, delimiter=';')



    column_names = df.columns.tolist()

    column_count_int = len(df.columns)
    column_count = [column_count_int] * column_count_int

    row_count_int = len(df)
    row_count = [row_count_int] * column_count_int

    value_type = df.dtypes

    missing_values_count = df.isna().sum().tolist()



    columns_numeric = df.select_dtypes(include = ['number'])
    value_range = columns_numeric.max() - columns_numeric.min()

    #columns_categoric = df.select_dtypes(exclude = ['number']).columns

    contents = []

    for column in df.columns:
        if column in columns_numeric.columns:
            contents.append(value_range[column])

        else:
            contents.append(None)



    numeric_means = df.select_dtypes(include = "number").mean()

    column_means = []

    for column in df.columns:
        if column in numeric_means.index:
            column_means.append(numeric_means[column])

        else:
          column_means.append(None)

    numeric_medians = df.select_dtypes(include = "number").median()

    column_medians = []

    for column in df.columns:
        if column in numeric_medians.index:
          column_medians.append(numeric_medians[column])

        else:
          column_medians.append(None)



    numeric_sd = df.select_dtypes(include = "number").std()

    column_sd = []

    for column in df.columns:
        if column in numeric_sd.index:
          column_sd.append(numeric_sd[column])

        else:
          column_sd.append(None)



    rounded_column_means = [round(value, 2) if value is not None else None for value in column_means]

    rounded_column_medians = [round(value, 2) if value is not None else None for value in column_medians]

    rounded_column_sd = [round(value, 2) if value is not None else None for value in column_sd]



    input_name = input("What is the dataset called? ")
    input_URL = input("What is the dataset URL? ")
    if input("Do you want to use LLM to generate a description of the dataset? (LLM/Manual) ") == "LLM":
        description = LLMconnect(column_names)
    else:
        description = input("Please provide a description of the dataset: ")

    output_data = {
    "title": input_name,
    "URL": input_URL,
    "column_count": column_count,
    "row_count": row_count,
    "column_names": column_names,
    "value_type": value_type,
    "range (if numeric) / top 5 occurring (if categoric)": contents,
    "missing_values_count": missing_values_count,
    "column_means": rounded_column_means,
    "column_medians": rounded_column_medians,
    "column_sd": rounded_column_sd,
    "description": description
    }

    output_dataframe = pd.DataFrame(output_data)
    #output_dataframe.to_csv('output.csv', index = False)

    return output_dataframe

def embedder(file):
    '''receives a dataframe and returns a dataframe with the embeddings of the column names in the dataframe'''
    
    output_df = retrieve_metadata(file)

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    output_term_list = output_df['column_names'].tolist()

    output_embedding_list = []

    for term in output_term_list:
        embedding = model.encode(term, convert_to_tensor=False)
        output_embedding_list.append({'term': term, 'embedding': [emb.tolist() for emb in embedding]})

    output_embedding_df = pd.DataFrame(output_embedding_list)
    #output_embedding_df.to_csv('output_embedding.csv', index=False)

    return output_embedding_df, output_df

def metastat(dataset, thesaurus = "thesaurus_embedding.csv", n = 3, threshold = 0.25, cosineinc = False):
    '''Receives a dataframe with the embeddings of the column names in the dataframe and a thesaurus with embeddings 
    of terms. Returns a dataframe with the cosine similarity between the embeddings of the column names in the dataframe 
    and the embeddings of the top n terms in the thesaurus that have a cosine similarity above the set threshold'''
    
    output_embeddings_df, output_df = embedder(dataset)

    
    output_embeddings_df["embedding"] = output_embeddings_df["embedding"].apply(eval)
    output_embeddings_df["embedding"] = output_embeddings_df["embedding"].apply(np.array)

    cessda_embeddings_df = pd.read_csv(thesaurus)
    cessda_embeddings_df["embedding"] = cessda_embeddings_df["embedding"].apply(eval)
    cessda_embeddings_df["embedding"] = cessda_embeddings_df["embedding"].apply(np.array)

    results = []
    for _, output_row in output_embeddings_df.iterrows():
        output_embedding = output_row["embedding"]
        
        temp = [] 

        for _, cessda_row in cessda_embeddings_df.iterrows():
            cessda_term = cessda_row["term"]
            cessda_embedding = cessda_row["embedding"]

            similarity = cosine_similarity([cessda_embedding], [output_embedding])[0][0]
            sub = {cessda_term : similarity}
            if similarity > threshold:
                temp.append(sub)
        
        similarities = {}

        for d in temp:
            similarities.update(d) 
            
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        top_similarities = sorted_similarities[:n]

        if cosineinc == False:
            cosex = []
            for x in top_similarities:
                cosex.append(x)
            top_similarities = cosex

        results.append(top_similarities)

    output_df["enrichment_categories"] = results

    output_df.to_csv('output.csv', index = False)
    
    print("data written to output.csv ")

#print(retrieve_metadata("Existing_own_homes__index_Netherlands_09012025_154126.csv"))

metastat("toy/Existing_own_homes__index_Netherlands_09012025_154126.csv", "toy/cessda_embedding.csv", n = 3, threshold = 0.25)