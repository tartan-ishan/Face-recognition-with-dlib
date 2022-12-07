#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[3]:


from PIL import Image
import time


# In[4]:


import face_recognition


# In[5]:


import numpy as np


# In[6]:


# resize image in folder as requried
# img = Image.open('bill_face_og.jpg')
# img.thumbnail((360,480))
# img.save('bill_face.jpg')


# In[55]:


file_path = r"C:\Users\Trident\Desktop\Code\player_images"
in_name = 'ishan.jpg'
out_name = 'ishan_resized.jpg'
resize_image(file_path, in_name, out_name, 240)


# In[52]:


def resize_image(path,img_in, img_out, max_dim):
    img = Image.open(f'{path}\\{img_in}')
    img.thumbnail((max_dim,max_dim))
    img.save(f'{path}\\{img_out}')


# In[7]:


img = face_recognition.load_image_file('bill_face.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[8]:


face = face_recognition.face_locations(img_rgb, model='cnn')[0]


# In[9]:


copy = img_rgb.copy()


# In[10]:


cv2.rectangle(copy, (face[0], face[1]), (face[2], face[3]), (0,0,255), 2)
cv2.imshow('copy', copy)
cv2.imshow('face', img_rgb)
cv2.waitKey(0)


# In[11]:


train_encode = face_recognition.face_encodings(img_rgb)[0]


# In[12]:


train_encodings = [train_encode, train_encode]


# In[13]:


test = face_recognition.load_image_file('bill_face_3.jpg')
test_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test_rgb)[0]
print(face_recognition.compare_faces(train_encodings, test_encode))


# In[14]:


# To get an averaged result for face match vs input array
res = [1,0,0,1,1,1,1,1,1,1,1]
round(np.mean(res),2) > 0.8


# In[15]:


import os
from datetime import datetime
import pickle


# In[16]:


path = 'player_images'


# In[26]:


images = []
classNames = []
mylist = os.listdir(path)
for player in mylist:
    curImg = cv2.imread(f'{path}/{player}')
    images.append(curImg)
    classNames.append(os.path.splitext(player)[0])
    


# In[ ]:


classNames


# In[27]:


def findEncodings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList


# In[28]:


# To encode faces from the sample images
encoded_face_train = findEncodings(images)


# In[29]:


# To open and update 
def register_name(name):
    with open('Registered_Names.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}\n')


# In[ ]:


cap  = cv2.VideoCapture(0)    # take pictures from webcam 
s = 4    # Image rescaling factor

while True:
    time.sleep(0.01)    # To avoid jittery frames and improce general accuracy at the cost of output fps 
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 1/s ,1/s)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        #To only perform the coparison and recognition when the *spacebar* is pressed on the keyboard 
        if cv2.waitKey(1) & 0xFF == ord(' '):     
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            print(matchIndex, name)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*s,x2*s,y2*s,x1*s
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                register_name(name)
    cv2.imshow('webcam', img)

# To stop the recognition and break from the loop, press 'q' a couple of times 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Inorder to properly close the windows opened to show the cap.read() output, we must use the following two commands
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




