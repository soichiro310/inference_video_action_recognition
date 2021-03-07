import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image 

from python_src.myException import *

class InferenceModel():
    def __init__(self, model, weight_path=None, label_map_path=None):
        
        # cudaが使える環境であれば，cudaを使用する
        if torch.cuda.is_available():
            print(' * Use Device: GPU')
            self.device = torch.device('cuda')
        else :
            print(' * Use Device: GPU')
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))
        
        self.classes = [x.strip() for x in open(label_map_path)] if label_map_path is not None else []

        if len(self.classes) == 0:
            raise LabelMapSettingError('label map is empty')

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
        
    
    # 動画ファイルの前処理を行い，DNNモデルによる推論を行う
    def inferenceVideo(self, video_path):
        # 動画ファイルを読み込む
        cap = cv2.VideoCapture(video_path)
        
        # 動画ファイルが開けなかった場合，VideoOpenErrorを出す
        if not cap.isOpened():
            raise VideoOpenError('Video Open Failed')

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # 動画フレームをPIL形式に変換し，前処理を行う
                pil_img = Image.fromarray(frame)
                pil_img = self.transform(pil_img)
                frames.append(pil_img)
            else:
                break
        
        # DNNモデルへ入力するための形式に変換
        X = torch.stack(frames, dim=0)
        X = X.permute(1,0,2,3)
        X = X.unsqueeze(0)
        X = X.to(self.device)
        
        # 推論を行う
        self.model.eval()
        with torch.no_grad():
            out_predictions, _ = self.model(X)
            
            # 推論結果をtorch.Tensor形式からnumpy形式へ変換
            out_predictions = out_predictions[0].cpu().numpy()
        
        return out_predictions