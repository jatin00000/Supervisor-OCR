# <div align="center"> 🕵️Supervisor OCR 🚗</div>
**Supervisor**, is a computer vision based AI number plate recognition project. <br/>
Here user will choose an image and upload. As upload is clicked, deep learning model will process the input and returns output when it detects the number plate and then 
expected text.<br/>

### Structure <br/>
![](2Untitled.png)<br/>
- - -
### Working <br/>

![](1Untitled.png)<br/>
![](Untitled.png)<br/>
 - - -

 ## System Requirment <br/>
 + MySQL <br/>
MySQL is required for database management. Download from [here](https://www.mysql.com/downloads/) and 
follow this installation procedure for <br/>
Mac [link](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=290s) <br/> 
Windows [link](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=588s) <br/>
Then in go to app.py file and open it in any text editor like sublime or VS Code. Type 
your MySQL user name and password at place shown below <br/>
![](3Untitled.png)<br/>
<br>
 + Tesseract-OCR for reading text from image.Download it from [here](https://osdn.net/projects/sfnet_tesseract-ocr-alt/downloads/tesseract-ocr-setup-3.02.02.exe/).<br>
Follow the following installation part from [link](https://youtu.be/Rb93uLXiTwA).<br>
<br>
 + Python language to run the software.
Download python from [here](https://www.python.org/downloads/) <br>
And installation for Mac [link](https://youtu.be/ezUCZiMXB20) and windows [link](https://youtu.be/Kn1HF3oD19c) <br>
<br>
 + Python Modules for running our application.  <br>
    Follow this steps: 
    + Create a folder name as ‘Supervisor’.
    + Download all the code in this folder.
    + Open your mac or windows terminal in this folder.
    + Type following command and press enter:
      +  <p style="background-color: Black; width: 500px">pip install -r .\requirement.txt</p>
      + Pip will automatically install all the necessary modules from requirement.txt.
      <p align="right"> 
      <img src="./4Untitled.png">
      </p>

- - -
## Running the Application
Install all the Dependencies stated above in document.<br>
+ Open the folder Supervisor in your mac or windows terminal
+ Write following command and press enter <br>
  <p style="background-color: Black; width: 200px">python app.py</p>

+ Server will start running
  ![](6Untitled.png)
+ In your Web Browser, go to http://127.0.0.1:5000/ <br>
  ![](7Untitled.png)
+ Now Enjoy the Application. 🍵

- - -
<div align="center">For More details, refer Our <a href="#">Design Document</a><br/></div>
<div align="center">😄Have a Good Day😄</div>

