# This Python code is a facial recognition system that uses the `face_recognition` library along with
# OpenCV (`cv2`) for capturing video frames and processing images. Here's a breakdown of what the code
import face_recognition
import os
import cv2
import numpy as np


datasetDir = "dataset"


knownEncodings = []
knownNames = []


for personName in os.listdir(datasetDir):
    personDirc = os.path.join(datasetDir, personName)
    
    if os.path.isdir(personDirc):
        for imgName in os.listdir(personDirc):
            imgPath = os.path.join(personDirc, imgName)
            image = face_recognition.load_image_file(imgPath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                knownEncodings.append(encodings[0])
                knownNames.append(personName)

print("Entrenamiento completado.")


videoCapture = cv2.VideoCapture(0)

while True:
    ret, frame = videoCapture.read()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(rgbFrame)
    faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)

    for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
        matches = face_recognition.compare_faces(knownEncodings, faceEncoding)
        name = "Desconocido"

        faceDistances = face_recognition.face_distance(knownEncodings, faceEncoding)
        bestMatchIndex = np.argmin(faceDistances)

        if matches[bestMatchIndex]:
            name = knownNames[bestMatchIndex]

        top, right, bottom, left = faceLocation
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()
