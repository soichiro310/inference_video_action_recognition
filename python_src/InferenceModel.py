import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image 

class InferenceModel():
    def __init__(self, model, weight_path=None, label_map_path=None, use_device='cpu'):
        self.model = model.to(torch.device(use_device))
        
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))
        
        self.classes = [x.strip() for x in open(label_map_path)] if label_map_path is not None else []
        
        # 入力データの前処理において行われる変換
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224), # 動画フレームの中心を切り出す
                transforms.ToTensor(),  # PIL形式からtorch.Tensorに変換
                transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                ),  # 正規化
            ]
        )
        self.use_device = use_device
    
    # 動画ファイルの前処理を行う
    def preprocessVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print('open error')
            return
        
        idx = 0
        frames = []
        
        while cap.isOpened():
            idx += 1
            ret, frame = cap.read()
            if ret:
                pil_img = Image.fromarray(frame)
                pil_img = self.transform(pil_img)
                frames.append(pil_img)
            else:
                break
        
        # 深層学習モデルへ入力するための形式に変換
        X = torch.stack(frames, dim=0)
        X = X.permute(1,0,2,3)
        X = X.unsqueeze(0)
        
        return X
    
    # 深層学習モデルへデータ(torch.Tensor)を入力して推論結果を返す
    def inference(self, input_tensor):
        
        self.model.eval()
        input_tensor = input_tensor.to(torch.device(self.use_device))
        
        with torch.no_grad():
            out_predictions, out_logits = self.model(input_tensor)
            
            # torch.Tensor形式からnumpy形式へ変換
            out_predictions = out_predictions[0].cpu().numpy()
            out_logits = out_logits[0].cpu().numpy()
        
        return out_predictions, out_logits