from DataSaving import DataHandler 
import wikipedia
import requests
from duckduckgo_search import DDGS
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

# 1. Let DDGS handle headers internally for better success rates
        with DDGS() as ddg:
            try:
        # 2. 'keywords' is a mandatory named argument in many versions
                results = ddg.text(keywords=ask_topic, region='wt-wt', max_results=5)
        
        # 3. 'results' is now a list. Check if it's not empty.
                if results:
            # 4. Use .get() to check for 'body' first, then 'snippet'
                    first_result = results[0]
                    snippet = first_result.get('body') or first_result.get('snippet') or ""
            
                    if snippet:
                        text = f"Snippet: {snippet}\n"
                    else:
                        text = "Result found, but no text snippet was available."
                else:
                    text = "No results found on DuckDuckGo."
            
            except Exception as e:
        # Log the specific error to your DataHandler for debugging
                self.ds.log(f"DDG Search Error: {str(e)}")
                text = f"DuckDuckGo search failed: {str(e)}"
        dataFrame = [wiki_info_user, text]
        result = cross_check_ML.LogisticRegPred(dataFrame)
        result_score = cross_check_ML.PredictionScore()
         
        if result == 1:
            return f'Information is verfied:\nHere is your summary on {ask_topic}:\n \n{formatted_summary}\n\n'f"Here is DDG's summary on {ask_topic}:\n \n{text} \nSimilarity Score: {result_score}%"
        else:
            return f'Wikipedia Information is not verfied:\nHere is your summary on {ask_topic}:\n \n{formatted_summary} \n\n'f"Here is DDG's summary on {ask_topic}:\n \n{text} \nSimilarity Score: {result_score}%"
        

    #Method for finding definition of words 
    def FindDefinition(self, sentence): 
    
        #main = input sentence

        self.ds.log(f"Ai Assistant: User Requested definition(s) on {sentence}")

        def_list = [] #word list
        words = sentence.split()
        for word in words:
            word = word.lower().strip(".,!,?")
        
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}" #API URL

            response = requests.get(url) #actually gets the info 

            data  = response.json() #json gets data into string form 

            #data is nested, so we have to get first info:

            # data[0] → The first entry found.
            #['meanings'][0] → The first part of speech (like "noun" or "verb").
            #['definitions'][0] → The first definition block.
            #['definition'] → The actual sentence defining the word.

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                continue
            else:
                definition = data[0]['meanings'][0]['definitions'][0]['definition']
         
            def_list.append(f'{word}: {definition}')
            formatted_list = [f"{i+1}. {item}" for i, item in enumerate(def_list)] #this is the formatted list
    
        #join them with newlines into one long string
        result = "\n".join(formatted_list)
    
        # save the entire formatted block at once
        self.ds.log(result, log_user_prompt=True)

        return result 



    
