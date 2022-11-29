import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract as pt


INPUT_WIDTH =  640
INPUT_HEIGHT = 640
#640 x 640 image


#Load our model
Model = cv2.dnn.readNetFromONNX('./static/models/best.onnx')

#Set Preference for backend and target 
Model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
Model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def DetectArea(img,Model):
    #Convert image to model input format
    image = img.copy()

    #Taking row, columns from image dimension
    Row, Col, h = image.shape

    #taking maximum value of value from row and column
    maxDimension = max(Row,Col)

    #Creating a symmetric matrix of zeros of max row size
    ImageArray = np.zeros((maxDimension,maxDimension,3),dtype=np.uint8)

    #Inserting image in this matrix
    ImageArray[0:Row,0:Col] = image

    #Get Prediction from our Model
    #1/255 is the skill factor
    blob = cv2.dnn.blobFromImage(ImageArray,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    #Setting blob as input
    Model.setInput(blob)

    #Get the predictions
    output = Model.forward()

    #On running preds.shape()
    # >>>(1,25200,6)
    #It is three dimensional but We only need two dimensional
    Result = output[0]
    # >>> (25200,6)
    # 25200 <-- number of detections and 6 <-- number of columns

    return ImageArray, Result

def NMS(ImageArray,Result):
    #Filtering Detections based on Confidence and Probability Score
    boxArray = []
    confidenceArray = []

    WidthImage, HeightImage = ImageArray.shape[:2]
    WidthFactor = WidthImage/INPUT_WIDTH
    HeightFactor = HeightImage/INPUT_HEIGHT

    #Taking each row one by one
    for i in range(len(Result)):
        row = Result[i]

        #Confidence of detecting license plate
        confidence = row[4] 
        if confidence > 0.4:
            #Probability score of license plate
            class_score = row[5] 
            if class_score > 0.25:

                #Taking center and width, height
                xCenter, yCenter , w, h = row[0:4]

                left = int((xCenter - 0.5*w)*WidthFactor)
                top = int((yCenter-0.5*h)*HeightFactor)
                width = int(w*WidthFactor)
                height = int(h*HeightFactor)
                box = np.array([left,top,width,height])

                confidenceArray.append(confidence)
                boxArray.append(box)

    #Converting array to list
    boxList = np.array(boxArray).tolist()
    confidenceList = np.array(confidenceArray).tolist()

    #NMS
    #There can be multiple boxArray for same image, doing non-maximum suppression
    index = np.array(cv2.dnn.NMSBoxes(boxList,confidenceList,0.25,0.45)).flatten()
    #Here 0.25 and 0.45 are thresold for these two

    return boxList, confidenceList, index

def GetText(image,bbox):
    x,y,w,h = bbox
    
    ROI = image[y:y+h, x:x+w]
    if 0 in ROI.shape:
        return ''
    else:
        ROI_BGR = cv2.cvtColor(ROI,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(ROI_BGR,cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
        text = pt.image_to_string(magic_color,lang='eng',config='--psm 6')
        text = text.strip()
        
        return text

def MakeDrawing(image,boxList,confidenceList,index):
    # MakeDrawing
    ListOfText = []
    for ind in index:
        x,y,w,h =  boxList[ind]
        bb_conf = confidenceList[ind]

        #Generating the confidence code
        PercentCon = 'Plate: {:.0f}%'.format(bb_conf*100)

        #Get the plate image
        LicenseNumber = GetText(image,boxList[ind])

        #Here (255,0,64) is the color for border, followed by border pixel
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,64),2)

        #Creaing two more rectangle on 30 pixel above and below for labels
        cv2.rectangle(image,(x,y-30),(x+w,y),(246, 142, 95),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(247, 245, 251),-1)

        #Displaying confidence code on image in a Above rectangle
        cv2.putText(image,PercentCon,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)

        #Displaying number plate text in a rectangle below
        cv2.putText(image,LicenseNumber,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
        
        ListOfText.append(LicenseNumber)

    return image, ListOfText


#Making Predictions
def Predictor(img,Model):

    #Detections
    ImageArray, Result = DetectArea(img,Model)

    #Doing Non Maximum Supression
    boxList, confidenceList, index = NMS(ImageArray, Result)
    #Getting image with plate marked on it and with label
    FinalImage, text = MakeDrawing(img,boxList,confidenceList,index)
    return FinalImage, text


#Main Function, this will be called by app.py
def Detect(Way, NameFile):
    #Reading image from input path
    image = cv2.imread(Way)

    # 8 bit array (0,255)
    image = np.array(image,dtype=np.uint8) 
    FinalImage, ListOfText = Predictor(image,Model)

    #Saving the result image
    cv2.imwrite('./static/predict/{}'.format(NameFile),FinalImage)
    return ListOfText

#Function to increase contrast of image
#Source: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf