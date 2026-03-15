
from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_faces(image_path):

    image = cv2.imread(image_path)

    results = detector.detect_faces(image)

    faces = []

    for result in results:
        x, y, w, h = result['box']
        face = image[y:y+h, x:x+w]
        faces.append(face)

    return faces
