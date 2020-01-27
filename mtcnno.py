from PIL import Image
from mtcnn_original.src.detector import detect_faces
from mtcnn_original.src.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import time

class MTCNN_O():
    def __init__(self):
        self.refrence = get_reference_facial_points(default_square= True)
    def align(self, img):
        _, landmarks = detect_faces(img)
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        
        return Image.fromarray(warped_face)
    
    def align_multi(self, img, limit=None, min_face_size=30.0):
        boxes, landmarks = detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
            ini = time.time()
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
            #print("Tempo de alinhamento: %.4f"% (time.time()-ini))
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    