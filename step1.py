import cv2
import face_recognition

imgElon = face_recognition.load_image_file("pic/trinh0.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgCheck = face_recognition.load_image_file("pic/trinh.jpg")
imgCheck = cv2.cvtColor(imgCheck, cv2.COLOR_BGR2RGB)

#locate face
location = face_recognition.face_locations(imgElon)[0]
#y1,x2,y2,x1
print(location)
#encode img
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(location[3],location[0]),(location[1],location[2]),(255,0,255),2)

locaCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(locaCheck[3],locaCheck[0]),(locaCheck[1],locaCheck[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeElon],encodeCheck)
distance = face_recognition.face_distance([encodeElon],encodeCheck)

print(result, distance)
cv2.putText(imgCheck,f"{result}{round(distance[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

cv2.imshow("Core", cv2.resize(imgElon,(0,0),fx=0.5,fy=0.5))
cv2.imshow("Filter", cv2.resize(imgCheck,(0,0),fx=0.5,fy=0.5))
cv2.waitKey()