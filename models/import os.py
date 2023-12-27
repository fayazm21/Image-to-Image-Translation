
import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
#from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import img_to_array





classifier =load_model('"C:/Users/aishw/Downloads/model3_200ep_ivc.h5"')
face_haar_cascade = cv2.CascadeClassifier("C:/Users/aishw/Downloads/haarcascade_frontalface_default.xml")


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(218,165,32),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        #img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(image.img_to_array(roi_gray), axis=-1)
        img_pixels = np.expand_dims(img_pixels, axis=0)


        #img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = classifier.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

    resized_img = cv2.resize(test_img, (1000, 700))
    #resized_img = cv2.flip(test_img, 1)
    cv2.imshow('Facial emotion recogntion test ',resized_img)



    if cv2.waitKey(1) & 0xFF == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()