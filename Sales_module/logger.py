from datetime import datetime

class log_class:
    """Logging Class
    """
    def __init__(self,folder_path,file_name):
        self.folder_path = folder_path
        self.file_name = file_name
        
    def create_log_file(self,log_message):
        """Methode: create_log_file

           input: log_message

           output: save log_message in file

           on error: raise error message
        """
        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
        
            with open(self.folder_path+self.file_name,'a') as file:
                file.write(str(self.date) + "\t" + str(self.current_time) + "\t\t" + log_message +"\n")
        except Exception as e:
            raise e