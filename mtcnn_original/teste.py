from src import detect_faces
from PIL import Image
import cv2
img = cv2.imread('image.jpg')
image = Image.open('image.jpg')
bboxes, landmarks = detect_faces(image)
#print(landmarks)

for i,bounding_box in enumerate(bboxes):

	print (bounding_box)
	cv2.rectangle(img,(int(bounding_box[0]), int(bounding_box[1])),(int(bounding_box[2]), int(bounding_box[3])),(0,155,255),2)

    #cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    #cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    #cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    #cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    #cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
cv2.imwrite("frame1.jpg", img)