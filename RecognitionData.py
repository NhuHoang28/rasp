
import cv2
import numpy as np
import sqlite3
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingData.yml")
def getProfile(id):
    conn = sqlite3.connect('C:\sqlite\sqlitestudio-3.3.3\SQLiteStudio\Data.db')
    query = "SELECT * FROM People WHERE ID=" + str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile = row
    conn.close()
    return profile


def insertOrUpdate(id, name, age, gender):
    # dùng webcam nhận dạng và chụp gương mặt của mình lại
    conn = sqlite3.connect('C:\sqlite\sqlitestudio-3.3.3\SQLiteStudio\Data.db')

    query = "Select * from People Where ID=" + str(id)

    cursor = conn.execute(query)

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if (isRecordExist == 0):
        query = "Insert into People(ID,Name,Age,Gender) values(" + str(id) + ",'" + str(name) + "','" + str(
            age) + "','" + str(gender) + "')"
    else:
        query = "Update People set Name= '" + str(name) + "', Age= '" + str(age) + "', Gender='" + str(
            gender) + "' Where ID=" + str(id)

    conn.execute(query)
    conn.commit()
    conn.close()
    # insert vao db


def getImagesWidthID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []
    print(imagePaths)
    for i in imagePaths:
        faceImg = Image.open(i).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(i.split('\\')[1].split('.')[1])
        print(ID)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('Trainning', faceNp)
        cv2.waitKey(1)
    return faces, IDs
cap=cv2.VideoCapture(0)
fontface=cv2.FONT_HERSHEY_SIMPLEX
check=0
while(True):
    if (check==1):
        cap = cv2.VideoCapture(0)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
    #camera ghi hinh
    check=0
    ret, frame=cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face= face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x, y), (x+w,y+h),(0,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        id, confidence= recognizer.predict(roi_gray)
        if confidence<60:

            profile=getProfile(id)
            if(profile!=None):
                cv2.putText(frame, "Name:" + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Age:" + str(profile[2]), (x + 10, y + h + 60), fontface, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Gender:" + str(profile[3]), (x + 10, y + h + 90), fontface, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame,"unknow",(x+10,y+h+30),fontface,1,(0,0,255),2)
    cv2.imshow('Image',frame)
    if (cv2.waitKey(1)== ord('c')):
        id = input("Enter your ID:")
        name = input("Enter your Name:")
        age = input("Enter your Age:")
        gender = input("Enter your Gender:")
        insertOrUpdate(id, name, age, gender)

        # load tv
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        sampleNum = 0

        while (True):
            # camera ghi hinh
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # so anh lay tang dan
                if not os.path.exists('dataSet'):
                    os.makedirs('dataSet')
                sampleNum += 1
                print("Chup anh thu " + str(sampleNum))
                # luu anh da chup khuon matvao file du lieu
                cv2.imwrite('dataSet/User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                # thoat neu so luong anh nhieu hon 200
            if sampleNum > 200:
                break
        cap.release()
        cv2.destroyAllWindows()
        # cv2.destroyAllWindows()
        if not os.path.exists('recognizer'):
            os.makedirs('recognizer')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'dataSet'
        faces, IDs = getImagesWidthID(path)
        # training
        recognizer.train(faces, np.array(IDs))
        # Luu vao file
        recognizer.save('recognizer/trainingData.yml')
        check=1
    if (cv2.waitKey(1)== ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
