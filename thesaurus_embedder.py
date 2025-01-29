from sentence_transformers import SentenceTransformer
import pandas as pd

def thesaurus_embedder(thesaurus):
    thesaurus_df = pd.read_csv(thesaurus)

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    thesaurus_term_list = thesaurus_df['Code descriptive term'].tolist()

    thesaurus_embedding_list = []

    for term in thesaurus_term_list:
        embedding = model.encode(term, convert_to_tensor=False)
        thesaurus_embedding_list.append({'term': term, 'embedding': [emb.tolist() for emb in embedding]})

    thesaurus_embedding_df = pd.DataFrame(thesaurus_embedding_list)
    thesaurus_embedding_df.to_csv('thesaurus_embedding.csv', index=False)
    
    return(print("Thesaurus embedding complete"))