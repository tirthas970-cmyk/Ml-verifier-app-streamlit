
import pandas as pd 
from sklearn.linear_model import LogisticRegression
import os
import joblib
from CosinSimAnalysis import CosinSim

class MLCrossCheck:

    def __init__(self):

        self.cosinesimcheck = CosinSim()
        self.PredictionProba = None
    
    def LogisticRegPred(self, datalist):

        df_test = pd.DataFrame([datalist], columns=['String 1', 'String 2'])

        #Saves model, so it doesn't train every time
        model_file = 'TrainData.joblib'
    
        if os.path.exists(model_file):
            model = joblib.load(model_file) 
        else:
            #Loads the dataset for training
            df_train = self.cosinesimcheck.findCosineOfTrainModel()
            X_train = df_train['similarity_score']
            y_train = df_train['Quality']
          
            model=LogisticRegression()
            #reshape is done to make it to a 2D array
            model.fit(X_train.values.reshape(-1, 1), y_train)
            joblib.dump(model, model_file)

        
        df_test_withCosinSim = self.cosinesimcheck.findCosineofNewData(df_test)
        X_test = df_test_withCosinSim["Similairty_Score"].values.reshape(-1, 1) 
        
        predictions = model.predict(X_test)

        #Needed for second method
        self.PredictionProba = model.predict_proba(X_test)
        
        #States if it is 1 or 0
        prediction_val = predictions[0]
        return prediction_val
    
    def PredictionScore(self):

        predictions_val_score = self.PredictionProba[0][1]
        predictions_score = round(predictions_val_score, 2) * 100
        
        return predictions_score

