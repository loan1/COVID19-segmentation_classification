#!/usr/bin/env python
# coding: utf-8

#importing the libraries
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
import warnings
import os 
import torchvision.transforms as transforms
# import albumentations  as A
# from albumentations.pytorch.transforms import ToTensorV2

# import module
# https://www.geeksforgeeks.org/python-import-module-from-different-directory/


# importing sys
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/mnt/DATA/covid19_resnet152_python-main/archive_14gb/mainscript/segment/script/')

from Train_data import augs, transfms
from script.utils import *
from script.train import training_loop
from script.dataset import ImageDataset, LungImageDataset
from script.test import img_transform, test_loop
from script.visualize import *
# from script.data import *
#for reading and displaying images

from PIL import Image

#Pytorch libraries and modules
import torch
from torch.nn import CrossEntropyLoss
import random
#for evaluating model
from sklearn.metrics import accuracy_score

import argparse

import matplotlib.pyplot as plt

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", default = './model/FT_ResNet152_cp_CXR_RGB_mean_std_compute.pt',type=str)
    parser.add_argument("--modelpath", default = './model/modelCXRFTResNet152_CXR_RGB_mean_std_compute.pt',type=str)

    parser.add_argument('--train_path', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/COVIDxCXR3/train/', type=str)
    parser.add_argument('--test_path', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/COVIDxCXR3/test/', type= str)

    parser.add_argument('--mask_path', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/mainscript/segment/visualize/FPN_DenseNet121/lung_mask/', type= str)

    parser.add_argument('--train_metadata', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/COVIDxCXR3/train_set.txt', type=str)
    parser.add_argument('--test_metadata', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/COVIDxCXR3/test_set.txt', type=str)
    parser.add_argument('--val_metadata', default='/mnt/DATA/covid19_resnet152_python-main/archive_14gb/COVIDxCXR3/val_set.txt', type=str)

    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--classes', default=['Negative', 'Positive'])
    parser.add_argument('--num_epochs', default= 50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    

    # parser.add_argument('--device', action='store_true', default=True)
    parser.add_argument('--feature_extract',action='store_true', default= False)
 

    opt = parser.parse_args()
    return opt

# def augment():
#     data_transforms = {
#         'train' : transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomAffine(degrees = 0, shear = 0.2),    
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225])),
#         ]),
#         'test' : transforms.Compose([
#             transforms.Resize((256,256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225]))
#         ])
#     }
#     return data_transforms

def dataloader():

    opt = get_opt()

    train_txt= pd.read_csv(opt.train_metadata, sep= '\s+', header=None)
    test_txt = pd.read_csv(opt.test_metadata, sep= '\s+', header=None)
    val_txt = pd.read_csv(opt.val_metadata,sep = "\s+", header= None)
    

    train_txt.columns = ["file_name","label"]
    test_txt.columns = ["file_name","label"]
    val_txt.columns = ["file_name","label"]

    # print(train_txt.count())
    # print(val_txt['label'].value_counts())
    # print(val_txt.count())

    train_dataset = ImageDataset(train_txt,opt.train_path,augs)
    test_dataset = ImageDataset(test_txt,opt.test_path,transfms)
    val_dataset = ImageDataset(val_txt,opt.train_path,transfms)

    # print(train_txt)

    # train_dataset = LungImageDataset(train_txt,opt.train_path,opt.mask_path,True,augs)
    # test_dataset = LungImageDataset(test_txt,opt.test_path,opt.mask_path,False,transfms)
    # val_dataset = LungImageDataset(val_txt,opt.train_path,opt.mask_path, True, transfms)


    # print(train_dataset)

    loader ={
        'train' : DataLoader(
            train_dataset, 
            batch_size= opt.BATCH_SIZE,
            shuffle=True
        ),
        'val' : DataLoader(
            val_dataset, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        ),
        'test' : DataLoader(
            test_dataset, 
            batch_size=opt.BATCH_SIZE,
            shuffle=True
        )
    }   
    return loader

def visualiz():

    opt = get_opt()
    # Get a batch of training data
    _,_,_,image, label,_ = next(iter(dataloader()['train']))
    fig = plt.figure(figsize=(25, 7))

    # display batch_size = 40 images
    for idx in np.arange(opt.BATCH_SIZE):
        ax = fig.add_subplot(4, opt.BATCH_SIZE/4, idx+1, xticks=[], yticks=[])
        imshow(image[idx]) # lay 1 cap co nghia la o day show anh
        ax.set_title(opt.classes[label[idx]]) # vì đã chuyển từ nes/pos -> 0,1 -> tensor 0,1
    plt.show()

def predict(path_img, model_ft, verbose = False):
    if not verbose:
        warnings.filterwarnings('ignore')
    try:
        checks_if_model_is_loaded = type(model_ft)
    except:
        pass
    model_ft.eval()
    if verbose:
        print('Model loader ...')
    image = img_transform(path_img, transfms)
    image1 = image[None,:,:,:]
    
    with torch.no_grad():
        outputs = model_ft(image1)
        
        _,pred_int = torch.max(outputs.data, 1)
        _,top1_idx = torch.topk(outputs.data, 1, dim = 1)
        pred_idx = int(pred_int.cpu().numpy())
        if pred_idx == 0:
            pred_str = str('Negative')
            print('img: {} is: {}'.format(os.path.basename(path_img), pred_str))
        else:
            pred_str = str('Positive')
            print('img: {} is: {}'.format(os.path.basename(path_img), pred_str))

def load_model(CHECKPOINT_PATH, model):
    checkpoint = torch.load(CHECKPOINT_PATH)#, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():

    # resnet = initialize_model(opt.num_classes, opt.feature_extract,use_pretrained=True)
    optimizer, scheduler = optimi(resnet,device, opt.feature_extract, opt.lr, opt.num_epochs)

    since = time.time()
    loss_list, acc_list = training_loop(resnet, optimizer, criterion, scheduler, device, opt.num_epochs, dataloader, opt.CHECKPOINT_PATH, opt.modelpath)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    visualize_loss(loss_list, './report/CXR/lossFTResNet152_CXR_RGB_mean_std_compute.png')
    visualize_acc(acc_list,'./report/CXR/ACCFTResNet152_CXR_RGB_mean_std_compute.png')

    # resnet = load_model(opt.CHECKPOINT_PATH, resnet)
    y_true, y_pred = test_loop(resnet, device, dataloader()['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes, './report/CXR/confusionmatrix_CXR_RGB_mean_std_compute.png')
    report(y_true, y_pred, opt.classes, './report/CXR/classification_reportpy152_CXR_RGB_mean_std_compute.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,resnet)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()

def testreport(resnet):
    opt = get_opt()
    resnet = load_model(opt.CHECKPOINT_PATH, resnet)
    y_true, y_pred = test_loop(resnet, device, dataloader()['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes, './report/CXR/confusionmatrix_CXR_RGB_mean_std_compute.png')
    report(y_true, y_pred, opt.classes, './report/CXR/classification_reportpy152_CXR_RGB_mean_std_compute.txt')
    
    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image,resnet)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    plt.show()

def seed_everything(seed):        

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) # set python seed

    np.random.seed(seed) # seed the global NumPy RNG

    torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA):
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)
    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    resnet = initialize_model(opt.num_classes, opt.feature_extract)
    
    # visualiz()
    main()
    # dataloader()
    # testreport(resnet)
