import numpy as np
import torch
from PIL import Image
from retinaface_pytorch.data import cfg_mnet, cfg_re50
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
import torch.backends.cudnn as cudnn
from retinaface_pytorch.layers.functions.prior_box import PriorBox
from retinaface_pytorch.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from retinaface_pytorch.models.retinaface import RetinaFace
from retinaface_pytorch.utils.box_utils import decode, decode_landm
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RETINANET():
    def __init__(self,network="resnet50"):
        cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        torch.set_grad_enabled(False)
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        if network == "mobile0.25":
            self.net = self.load_model(self.net,"retinaface_pytorch/mobilenet0.25_Final.pth",False)
        else:
            self.net = self.load_model(self.net,"retinaface_pytorch/Resnet50_Final.pth",False)
        self.net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cuda")
        self.net = self.net.to(self.device)


        self.refrence = get_reference_facial_points(default_square= True)

    def align_multi(self, img, limit=None, min_face_size=30.0):
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j],landmark[j+1]] for j in range(0,10,2)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
            #cv2.imshow("faceCrop",warped_face)
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def detect_faces(self, image, min_face_size=20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        
        
        
        resize = 1

        # testing begin
        #image_path = "./curve/test.jpg"
        img_raw = image

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        #tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)[0]#args=confidence_threshold
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]#args=top_k
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets,0.4 )#args.nms_threshold
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:750, :] # args.args.keep_top_k ==750
        landms = landms[:750, :]# args.args.keep_top_k ==750

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if True:
            #print("dets:")
            x = dets[dets[:,4]>=0.6]
            bounding_boxes = x[:,:5]
            landmarks = x[:,5:]
            
            # print(dets[dets[:,4]>=0.6])
            # print("----------")
            # print(dets)
            # for b in dets:
            #     print(b)
            #     print(type(b))
            #     if b[4] < 0.6:#args.vis_thres==0.6
            #         continue
            #     text = "{:.4f}".format(b[4])
            #     b = list(map(int, b))
            #     cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            #     cx = b[0]
            #     cy = b[1] + 12
            #     cv2.putText(img_raw, text, (cx, cy),
            #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            #     # landms
            #     cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            #     cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            #     cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            #     cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            #     cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # # save image

            # name = "test.jpg"
            # cv2.imshow(name, img_raw)
        return bounding_boxes,landmarks

    def load_model(self,model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
            
    def check_keys(self,model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self,state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}