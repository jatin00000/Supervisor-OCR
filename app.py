#Importing necessary modules and library for backend
from flask import Flask, render_template, request
import os
from deeplearning import Detect
import mysql.connector as c

#Connecting to MySQL database and setting up Database and Table
mydb = c.connect(host="localhost", user="root", password="mysqluser@#")
myc = mydb.cursor()
myc.execute("CREATE DATABASE IF NOT EXISTS supervisor;") #Database
myc.execute("USE supervisor")
myc.execute("CREATE TABLE IF NOT EXISTS data (name VARCHAR(50), no_of_plate INT, text_list VARCHAR(100));") #Table Creation


#Webserver gateway interface
app = Flask(__name__)

#Place for saving the upload file
# main_path = os.getcwd()
# Path_upload = os.path.join(main_path, 'static/upload/')
BASE_PATH = os.getcwd()
Path_upload = os.path.join(BASE_PATH,'static/upload/')

#Creating decorator for ruuning server
#Allowing both mehtods through below code
@app.route('/', methods=['POST', 'GET'])
def index():

    #for Post method
    if request.method == 'POST':
        upload_file = request.files['image_name'] #Name of file i want to get
        filename = upload_file.filename #Getting filename through filename method
        path_save = os.path.join(Path_upload,filename)
        upload_file.save(path_save)
        ListOfText = Detect(path_save,filename) #Save the file

        #Joining text of all number plates of photo into one string
        new_ListOfText = ' '.join([(str(item)+";") for item in ListOfText])
        query = ("INSERT INTO data (name, no_of_plate, text_list) VALUES( %s,%s,%s);")
        values = (filename, len(ListOfText), new_ListOfText)
        myc.execute(query, values) #Writing data to MySQL
        mydb.commit()

        #Returning Homepage on succeful operation
        return render_template('index.html',upload=True,ImageUpload=filename,ListOfText=ListOfText,num=len(ListOfText))

    return render_template('index.html', upload=False)

#Running the app
if __name__ == "__main__":
    app.run(debug=True)
