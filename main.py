import cv2 as cv


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
webcam = cv.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.putText(frame,'FACE',(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        
    eyes = eyes_cascade.detectMultiScale(gray,1.3,5)
    for (ex,ey,w,h) in eyes:
        cv.putText(frame,'EYE',(ex,ey),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv.LINE_AA)
        cv.rectangle(frame,(ex,ey),(ex+w,ey+h),(0,0,255),4)


    cv.imshow('Webcam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv.destroyAllWindows()