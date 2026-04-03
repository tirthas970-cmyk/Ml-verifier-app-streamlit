from sentence_transformers import SentenceTransformer
import pandas as pd
import torch


class CosinSim:

    def __init__(self):
        #load model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def findCosineOfTrainModel(self):

        #open the microsoft corpus file
        try:
            df = pd.read_csv('CrossCheckDataTrain.csv', encoding='windows-1252')
        except UnicodeDecodeError:
            print("windows-1252 failed, trying latin1")
            df = pd.read_csv('CrossCheckDataTrain.csv', encoding='latin1')

        # Clean Data
        df['#1 String'] = df['#1 String'].fillna("")
        df['#2 String'] = df['#2 String'].fillna("")

        #Convert it to embeddings (vectors), while making it into a tensor, allowing for easy cosine similairty score
        String1_encode = self.model.encode(df['#1 String'].tolist(), convert_to_tensor=True)
        String2_encode = self.model.encode(df['#2 String'].tolist(), convert_to_tensor=True)

        #torch is used to only find the similairty between each row
        cosine_scores = torch.nn.functional.cosine_similarity(String1_encode, String2_encode)


        df['similarity_score'] = cosine_scores.tolist()

        return df
    
    def findCosineofNewData(self, df_test):
       
       TestEmbedding1 = self.model.encode(df_test["String 1"].tolist(), convert_to_tensor=True)
       TestEmbedding2 = self.model.encode(df_test["String 2"].tolist(), convert_to_tensor=True)

       cosine_scores_test = torch.nn.functional.cosine_similarity(TestEmbedding1, TestEmbedding2)

       df_test["Similairty_Score"] = cosine_scores_test.tolist()

       return df_test
       

 

    

       

 

    
