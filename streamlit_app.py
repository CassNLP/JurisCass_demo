import streamlit as st
import requests
import numpy as np
import faiss
import numpy as np
import faiss
import pandas as pd


#index_file_path = "med_faiss_index.index"
#index_file_path = "full_faiss_index.index"
#index = faiss.read_index(index_file_path)

d = 768  
num_parts = 10
index_file_base_path = "faiss_index_part"

# Create a new index to store the merged vectors
index = faiss.IndexFlatIP(d)

# Load each part index and append it to the merged index
for part_num in range(num_parts):
    part_index_file_path = f"{index_file_base_path}_{part_num}.index"
    part_index = faiss.read_index(part_index_file_path)
    
    # Retrieve the vectors from the part index
    xb_part = part_index.reconstruct_n(0, part_index.ntotal)    
    index.add(xb_part)

xq=np.load('xq.npy')

df_urls = pd.read_pickle('df_urls.pkl')

query_list=["délai d'action pour demander un rappel de salaire?",\
            "évaluation de l'indemnité d'éviction dans un bail commercial",\
            "conditions de validité de l'acte de cautionnement du locataire",\
            "exonération de responsabilité du conducteur en cas de faute de la victime d'un accident de la circulation",\
            "obligation de délivrance du bailleur",\
            "Responsabilité du médecin après signature d'une décharge",\
            "Fixation de la prestation compensatoire des époux séparés de biens",\
            "Qui du bailleur ou du locataire doit prendre en charge les frais de désinsectisation ?",\
            "Validité de la clause de renonciation à recours contre le bailleur commercial",\
            "Licenciement du salarié en cas de dénigrement sur un réseau social ",\
            "Preuve de l'usucapion",\
            "Prescription du délit de blanchiment",\
            "Renonciation à succession et actions des créanciers",\
            "Assurance-vie et quotité disponible",\
            "Prescription des droits de donation",\
            "Sanction contre le bailleur du logement indigne",\
            "Devoir de secours entre concubins",\
            "Sanction tapage nocturne copropriétaire ",\
            "Devoir d'information du notaire acquisition immobilière",\
            "Droit d'enregistrement en cas de cession de parts sociales en usufruit ?"\
                      ]


#def semantic_src(request_number, faiss_index, faiss_retrieval_limit, sim_threshold, docx_generation):
def semantic_src(i, index, k, threshold):
    D, I = index.search(xq, k)       # D contains inner products, I indices of the neighbors
    query=query_list[i]
    ids=list(df_urls.index[I[i]])
    dic_score={}
    for j in range(k):
        if D[i][j]>threshold:
            dic_score[ids[j]]=D[i][j]
    print(f'\n{len(dic_score)} results found with similarity to request > {threshold} to query : {query}')    
    print(f'{dic_score}')

    return dic_score

requests_list=[]

for item in query_list:
    requests_list.append({"name" : item})

# Create a menu to select a request
request_option = st.selectbox("Choisissez une recherche à effectuer : ", [r["name"] for r in requests_list])

# Find the index of the selected request
selected_index = [r["name"] for r in requests_list].index(request_option)

# Display the selected index
#st.write(f"You selected option at index {selected_index}")



sim_threshold = st.slider("Niveau de similarité minimal accepté", 0.3, 0.95, 0.3, 0.05)
faiss_retrieval_limit = st.slider("Nombre maximal de documents retournés", 10, 210, 10, 40)

# Button to execute the request
if st.button("Rechercher"):
    try:
        #st.write(request_option)
        faiss_index=index
        #faiss_retrieval_limit=100
        #sim_threshold=0.5
        dic_score=semantic_src(selected_index, faiss_index, faiss_retrieval_limit, sim_threshold)
        df = pd.DataFrame(list(dic_score.items()), columns=['ID', 'Similarity Score'])
        df['URL'] = df['ID'].map(df_urls['url'])
        df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        df=df.drop('ID',axis=1)

        df['RANK']=df.index+1
        df=df.reset_index(drop=True)
        df=df.set_index('RANK')

        # Sort the DataFrame by 'Similarity Score' in descending order
        df = df.sort_values(by='Similarity Score', ascending=False)

        
        # Display the dataframe with clickable URLs in Streamlit
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# x = st.slider("Select a value")
# st.write(x, "squared is", x * x)