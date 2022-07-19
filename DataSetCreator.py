import cv2
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Id = input('Enter your ID: ')
sampleNum = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum = sampleNum+1
        cv2.imwrite("dataset/User."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
        
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    elif sampleNum >200:
        break
cam.release()
cv2.destroyAllWindows()
