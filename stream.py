from flask import Flask, render_template, Response
import cv2
import sys
import numpy

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedfile/trainningData.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

first_shift=[1]
second_shift=[2]

id = 0
names = ['None','Vamsi','Ambica']
number=[]
nm=[]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    camera=cv2.VideoCapture(0)

    while True:
        retval, im = camera.read()
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 1)

        for (x,y,w,h) in faces:
            cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 5)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence<100):
                nm.append(names[id])
                number.append(id)
                confidence = "  {0}%".format(round(100 - confidence))
                if id in first_shift:
                    cv2.putText(im,'First Shift',(x+w-120,y+h+25),font,1,(255,255,255),2)
                if id in second_shift:
                    cv2.putText(im,'Second Shift',(x+w-120,y+h+25),font,1,(255,255,255),2)
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(im, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(im, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
