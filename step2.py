import cv2
import face_recognition
import os
import numpy as np

path = "pic"
listImg = os.listdir(path)
listEncode = []
print(listImg)
for i in listImg:
    tg = face_recognition.load_image_file("pic/"+i)
    tg = cv2.cvtColor(tg, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(tg)[0]
    listEncode.append(encode)
    decodeImg = {i : encode}
#
# print(decodeImg)

cap = cv2.VideoCapture(0);
while True:
    ret, frame = cap.read()
    frames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detectFace = face_recognition.face_locations(frames)
    encodeFace = face_recognition.face_encodings(frames)
    #print(detectFace,encodeFace)
    for i,j in zip(detectFace,encodeFace):
        matches = face_recognition.compare_faces(listEncode,j)
        facedis = face_recognition.face_distance(listEncode,j)
        print(facedis)
        matchIndex = np.argmin(facedis) #lay min
        print(facedis[matches])

        if facedis[matchIndex] < 0.50 :
            name = listImg[matchIndex][0:len(listImg[matchIndex])-4]
        else:
            name = "Del bt"

        y1, x2, y2, x1 = i

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        #cv2.rectangle(frame, (i[3], i[0]), (i[1], i[2]), (255, 0, 0), 2)
        #j so với decodeImg{ten : encode}
        # for key in decodeImg:
        #     result = face_recognition.compare_faces([decodeImg[key]], j)
        #     distance = face_recognition.face_distance([decodeImg[key]], j)
        #     if distance<0.5:
        #         cv2.rectangle(frame, (i[3], i[0]), (i[1], i[2]), (255, 0, 0), 2)
        #         cv2.putText(frame, f"{key}", (i[1], i[2]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("video",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

