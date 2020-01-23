from config import get_config
import argparse
import cv2
from PIL import Image
import cv2
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import os
import time

def getExtension(filename):
    fileName, fileExtension = os.path.splitext(filename)
    return fileExtension
def inArray(array, to_look):
    for x in array:
        if(to_look[1:] == x):
            return True
    return False
def isImage(filename):
    extensions = ['jpeg', 'jpg', 'jpe', 'tga', 'gif', 'tif', 'bmp', 'rle', 'pcx', 'png', 'mac', 'pnt', 'pntg', 'pct', 'pic', 'pict', 'qti', 'qtif']
    extension = getExtension(filename)
    
    if (inArray(extensions, extension)):
        return True
    else:
        return False

def procurandoDiretorio(directory):
    arquivos = os.listdir(os.path.expanduser(directory))
    listaimagens = []
    for arquivo in arquivos:
        #print(arquivo)
        if(isImage(arquivo)):
            listaimagens.append(arquivo)
    return listaimagens



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)

    args = parser.parse_args()

    conf = get_config(False)
    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')



    imagens = procurandoDiretorio("./procurarface")
    print(imagens)

    for x in imagens:
        img = cv2.imread(str(conf.face_verify_path/x))
        frame = img#cv2.resize(img, (1280, 720))
        try:
          #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
          image = Image.fromarray(frame)
          bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
          bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
          bboxes = bboxes.astype(int)
          bboxes = bboxes + [-1,-1,1,1] # personal choice    
          results, score = learner.infer(conf, faces, targets, args.tta)
          for idx,bbox in enumerate(bboxes):
              if args.score:
                  frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
              else:
                  frame = draw_box_name(bbox, names[results[idx] + 1], frame)
        except:
            print('detect error')    
            
        cv2.imwrite('./resultado/'+x, frame)


        cv2.waitKey(0)