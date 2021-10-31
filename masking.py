import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(680,560))
# to save the output file 

# to start the webcam
cap = cv2.VideoCapture(0)
img = cv2.imread('me.jpg')

# masking the bg 

while True :
    ret,frame = cap.read()
    print(ret)
    #resizing the images
    frame = cv2.resize(frame,(640,480))
    img = cv2.resize(img,(640,480))

    #selecting the ranges
    l_black = np.array([30,30,0])
    u_black = np.array([104,153,70])

    #removing black from bg
    mask = cv2.inRange(frame,l_black,u_black)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    f = frame-res
    f = np.where(f == 0 , img , f)

    finaloutput = cv2.addWeighted(frame,1,res,1,0)
    output_file.write(finaloutput)

    #showing the output
    cv2.imshow("video",frame)
    cv2.imshow("mask",f)

    if cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()