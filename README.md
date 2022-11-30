# <div align="center"> 🕵️Supervisor OCR 🚗</div>
**Supervisor**, is a computer vision based AI number plate recognition project. <br/>
Here user will choose an image and upload. As upload is clicked, deep learning model will process the input and returns output when it detects the number plate and then 
expected text.<br/>

### Structure <br/>
![2Untitled](https://user-images.githubusercontent.com/94428262/204727098-29e4c746-5c7b-448c-b233-6712761c1312.png)<br/>
- - -
### Working <br/>

![1Untitled](https://user-images.githubusercontent.com/94428262/204726903-5dd6b1c6-9055-42a8-b008-cd2e6447725c.png)<br/>
![Untitled](https://user-images.githubusercontent.com/94428262/204726939-92b3ea30-7563-4951-8465-61ce77423d7a.png)<br/>
 - - -

 ## System Requirment <br/>
- MySQL <br/>
MySQL is required for database management. Download from [here](https://www.mysql.com/downloads/) and 
follow this installation procedure for <br/>
Mac [link](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=290s) <br/> 
Windows [link](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=588s) <br/>
Then in go to app.py file and open it in any text editor like sublime or VS Code. Type 
your MySQL user name and password at place shown below <br/>
![3Untitled](https://user-images.githubusercontent.com/94428262/204728432-bb213653-3dc3-4118-b867-789b947feaea.png)
- Tesseract OCR for reading text from image.Download it from [here](https://osdn.net/projects/sfnet_tesseract-ocr-alt/downloads/tesseract-ocr-setup-3.02.02.exe/).<br>
Follow the following installation part from [link](https://youtu.be/Rb93uLXiTwA).<br>
- Python language to run the software.
Download python from [here](https://www.python.org/downloads/) <br>
And installation for Mac [link](https://youtu.be/ezUCZiMXB20) and windows [link](https://youtu.be/Kn1HF3oD19c) <br>
- Python Modules for running our application.  <br>
    Follow this steps: 
  - Create a folder name as ‘Supervisor’.
  - Download all the code in this folder.
  - Open your mac or windows terminal in this folder.
  - Type following command and press enter:<br>
    - `pip install -r .\requirement.txt`
    - Pip will automatically install all the necessary modules from requirement.txt. <br/>
![4Untitled](https://user-images.githubusercontent.com/94428262/204727012-d1c0de2d-7d31-4c55-bdee-097c7ddbb97b.png)

- - -
## Running the Application
Install all the Dependencies stated above in document.<br>
+ Open the folder Supervisor in your mac or windows terminal
+ Write following command and press enter <br>
  `python app.py`

+ Server will start running <br>
  ![6Untitled](https://user-images.githubusercontent.com/94428262/204727702-6d50e3f5-92f4-4958-892a-04e4085f85dc.png)
+ In your Web Browser, go to http://127.0.0.1:5000/ <br>
  ![7Untitled](https://user-images.githubusercontent.com/94428262/204727669-74dad0db-01a2-42f8-815e-9959fefb2a7f.png)

+ Now Enjoy the Application. 🍵

- - -
<div align="center">For More details, refer Our <a href="#">Design Document</a><br/></div>
<div align="center">😄Have a Good Day😄</div>

