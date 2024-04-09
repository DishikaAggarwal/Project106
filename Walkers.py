import cv2


# Create our body classifier
face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    faces = face_cascade.detectMultiScale(gray)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(233,53,43),2)

    # Display the resulting frame
    cv2.imshow("img", frame)

    #32 is the Space Key
    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
