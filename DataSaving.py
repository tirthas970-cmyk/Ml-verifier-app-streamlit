
import datetime

class DataHandler:
    
    def __init__(self, filename="SavingUserDatav3"):
        self.filename = filename

    def log(self, document, log_user_prompt=False):  #takes an optional parameter to log initiaed user prompts
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") #date time
        log_entry = f"[{timestamp}]\n \n{document}\n" 
 
        try:
            with open(self.filename, 'a', encoding="utf-8") as file: #changed w to a to append everything so it doesn't refresh
            #encoding='utf-8' is added to translate other stuff like emojies
                file.write(log_entry)
            print("Document saved successfully!")
            if log_user_prompt:
                print("Doucment Saved Successfuly!")
        except OSError as e:
            print(f"Error saving document: {e}")
