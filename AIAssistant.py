from DataSaving import DataHandler 
import wikipedia
import requests
from ddgs import DDGS
import time
from CrossCheckLogisticRegressionModel import MLCrossCheck


class Assistant:
    def __init__(self):
        self.ds = DataHandler() #instance variable using DataSaving Class

    #Method to get info from Wiki
    def GetInfo(self, topic, numsentence):

        try: 
            #Goes to wikipedia and sumarizes topic in amount of sentences requested
            summary = wikipedia.summary(topic, sentences=int(numsentence), auto_suggest=False) 
            return summary 
        except wikipedia.exceptions.DisambiguationError as e:
            # If there are multiple meanings, tell the user the first few options
            errormessage1 = f"There are many meanings for that. Did you mean: {', '.join(e.options[:3])}?"
            self.ds.log(f"Ai Assistant (Disambiguation Error): {errormessage1} ")
            return errormessage1
        except wikipedia.exceptions.PageError:
            # If the topic doesn't exist at all, it will
            errormessage2 = "I'm sorry, I couldn't find a Wikipedia page for that."
            self.ds.log(f"Ai Assistant (Page Error): {errormessage2}")
            return errormessage2
        except Exception:
            errormessage3 =  "I'm having trouble connecting to Wikipedia right now."
            self.ds.log(f"Ai Assistant (Exception): {errormessage3}")
            return errormessage3
        
    #This method gets the user's text generated using the method above
    #It also generates DDG text
    #It compares those two texts by using the logistic regression model class made
    #The cosine similairty is compared, allowing the model to predict if they are similar using semantic analysis
    def Crosscheck(self, ask_topic, ask_sentence_length):

        cross_check_ML = MLCrossCheck()

        self.ds.log(f"User requested info on topic: {ask_topic} with length {ask_sentence_length}.")

        wiki_info_user = self.GetInfo(ask_topic, ask_sentence_length)
    
        
        #The formatted summary that will be printed 
        formatted_summary = wiki_info_user

        text = "No additional text found"

        with DDGS() as ddg:
            try:
                time.sleep(2)  # Critical: stops DDG from flagging you as a bot
        # Use 'lite' backend and 'keywords=' argument
                results = list(ddg.text(query=ask_topic, region='uk-en', max_results=3, backend='lite'))
        
                if results:
            # Safely grab the first result
                    first_res = results[0]
                    snippet = first_res.get('body') or first_res.get('snippet', '')
                    text = f"Snippet: {snippet}\n"
                else:
                    text = "DDG is throttling this IP. Try again in a few minutes."
            except Exception as e:
                text = f"Search Error: {e}"
        
        dataFrame = [wiki_info_user, text]
        result = cross_check_ML.LogisticRegPred(dataFrame)
        result_score = cross_check_ML.PredictionScore()
         
        if result == 1:
            return f'Information is verfied:\nHere is your summary on {ask_topic}:\n \n{formatted_summary}\n\n'f"Here is DDG's summary on {ask_topic}:\n \n{text} \nSimilarity Score: {result_score}%"
        else:
            return f'Wikipedia Information is not verfied:\nHere is your summary on {ask_topic}:\n \n{formatted_summary} \n\n'f"Here is DDG's summary on {ask_topic}:\n \n{text} \nSimilarity Score: {result_score}%"
        

    #Method for finding definition of words 
def FindDefinition(self, sentence): 
    self.ds.log(f"Ai Assistant: User Requested definition(s) on {sentence}")

    def_list = [] 
    words = sentence.split()
    for word in words:
        word = word.lower().strip(".,!?;:")
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

        
        try:
            # Adding a timeout of 5 seconds so the app doesn't hang forever
            response = requests.get(url, timeout=5) 
            
            if response.status_code == 200:
                data = response.json()
                # Accessing the nested dictionary safely
                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                def_list.append(f'{word}: {definition}')
            else:
                def_list.append(f"{word}: Definition not found.")
        except Exception as e:
            def_list.append(f"{word}: Error connecting to service.")

    formatted_list = [f"{i+1}. {item}" for i, item in enumerate(def_list)]
    result = "\n".join(formatted_list)
    self.ds.log(result, log_user_prompt=True)
    return result 




    
