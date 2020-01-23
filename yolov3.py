from yolov3_pytorch.src.models import *  # set ONNX_EXPORT in models.py
import torch.nn.functional as F
import numpy as np
from PIL import Image
class YOLOV3(object):
    def __init__(self,weights = "yolov3_pytorch/weights/lastwolverine.pt",):
        with torch.no_grad():
            self.model = Darknet("yolov3_pytorch/cfg/yolov3-face.cfg")
            self.img_size = 416
            self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else "0")
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
            # Eval mode
            self.model.to(self.device).eval()
            # Half precision
            self.half = False
            half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                self.model.half()
            pass
    def align_multi(self, img, limit=None, min_face_size=30.0):
        with torch.no_grad():
            # Padded resize
            imgNova = self.letterbox(img, new_shape=self.img_size)[0]

            # Convert
            imgNova = imgNova[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgNova = np.ascontiguousarray(imgNova, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
            imgNova /= 255.0  # 0 - 255 to 0.0 - 1.0


            # Get detections
            imgNova = torch.from_numpy(imgNova).to(self.device)
            if imgNova.ndimension() == 3:
                imgNova = imgNova.unsqueeze(0)
            pred = self.model(imgNova)[0]
            if self.half:
                pred = pred.float()
            # Apply NMS
            pred = non_max_suppression(pred, 0.3, 0.5,classes=None,agnostic=True)

            boxes = []
            faces = []
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(imgNova.shape[2:], det[:, :4], img.shape).round()
                    

                    # Write results
                    
                    for *xyxy, conf, cls in det:
                        boxes.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]),1])
                        #boxes.append([xyxy[0],xyxy[1],xyxy[2],xyxy[3]])
                        faces.append(Image.fromarray(img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]).resize((112,112), Image.ANTIALIAS))
            auxaux = np.asarray(boxes, dtype=np.float32)
            return auxaux, faces
    def letterbox(self,img, new_shape=(416, 416), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = max(new_shape) / max(shape)
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)