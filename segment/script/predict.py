# link dataset https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
# import module
from script.custom_dataset import DatasetPredict
from script.Train_data import transfms
#import lib
import cv2
from skimage import morphology
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

import torch
import numpy as np
import matplotlib.pyplot as plt

def dataloaderPre(data_dir, batch_size, transfms, train = True, Neg = True): # data COVIDx

    if train == True:
        data_dir += 'EDA_Train/'
    else:
        data_dir += 'EDA_Test/'

    if Neg == True:
        data_dir += 'Negative/'
    else:
        data_dir += 'Positive/'
    
    print('datadir: ', data_dir)

    data = DatasetPredict(data_dir, transform = transfms)

    loader = DataLoader(
            data, 
            batch_size=batch_size,
            shuffle=False
            )
    
    return loader


def predict(dataloader, model, device): # dataset COVIDx
    model.eval()
    with torch.no_grad():
        image, y_predict = [], []
        for x, _ in tqdm(dataloader):
            x = x.to(device)       
            y_pred = model(x)
            pred = y_pred.cpu().numpy() # mask output          

            pred = pred.reshape(len(pred), 256, 256)
            pred = pred > 0.3 #threshold
            pred = np.array(pred, dtype=np.uint8)
            y_predict.append(pred)

            x = x.cpu().numpy()
            x = x.reshape(len(x), 256, 256)
            x = x*0.3 + 0.59
            x = np.squeeze(x)

            x = np.clip(x, 0, 1)
            image.append(x)       

    return image, y_predict


def postprocess(img):

    areas = []
    img = img.astype("uint8")
    blur = cv2.GaussianBlur(img, (3,3), 0) #làm mờ ảnh
    _, thresh = cv2.threshold(blur, 0,1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #nhị phân hóa ảnh

    contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #tính contours

    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    areas = np.sort(areas)[::-1]

    thresh = thresh.astype(bool)
    if len(contours) > 1:
        thresh = morphology.remove_small_objects(thresh.copy(),areas[1])
    if len(contours) > 2 :
        thresh = morphology.remove_small_holes(thresh, areas[2])
    
    return thresh


    


