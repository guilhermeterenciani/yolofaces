import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from yolov3 import YOLOV3
from mtcnno import MTCNN_O
from retinanet import RETINANET
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import numpy as np
import traceback
import sys, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.19, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-fi", "--face_identified", help="Tipo do identificador de faces utilizado: yolov3, mtcnn,mtcnno,retina",type=str,default='yolov3')
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    yolov3 = YOLOV3()
    print("Yolov3 loaded")
    mtcnno = MTCNN_O()
    print("mtcnno loaded")
    retinanet = RETINANET("mobile0.25")
    print("retinanet loaded")
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    cap = cv2.VideoCapture("rtsp://before:beforeti@192.168.1.132:554/live/ch0")
    #cap = cv2.VideoCapture(1)
    cap.set(3,1280)
    cap.set(4,720)
    if args.save:
        video_writer = cv2.VideoWriter(str(conf.data_path)+'/resultado.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:            
            try:
                if args.face_identified=='yolov3':

                    image = frame
                    
                    bboxes, faces = yolov3.align_multi(image, conf.face_limit, conf.min_face_size)
                    
                    
                elif args.face_identified=="mtcnn":
                
                    #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                    image = Image.fromarray(frame)
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                elif args.face_identified=="mtcnno":
                
                    #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                    image = Image.fromarray(frame)
                    bboxes, faces = mtcnno.align_multi(image, conf.face_limit, conf.min_face_size)
                elif args.face_identified=="retina":
                    image = frame
                    bboxes, faces = retinanet.align_multi(image, conf.face_limit, conf.min_face_size)

                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice

                results, score = learner.infer(conf, faces, targets, args.tta)
                
                #print(bboxes)
                #opencvImage = cv2.cvtColor(np.array(faces), cv2.COLOR_RGB2BGR)
                #cv2.imshow("Crop face",opencvImage);

                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                print(traceback.format_exc())
                
        cv2.imshow('face Capture', frame)
        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()    