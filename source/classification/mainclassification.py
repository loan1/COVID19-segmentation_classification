
#importing the libraries

import time
from unittest import loader

#for reading and displaying images

from PIL import Image

#Pytorch libraries and modules
import torch
from torch.nn import CrossEntropyLoss

#for evaluating model
from sklearn.metrics import accuracy_score

# visualize
import matplotlib.pyplot as plt

# import module

import sys
sys.path.insert(0, '/mnt/DATA/research/classificationCOVID19applyseg/source/segmentation/dataloader/')

from configs.myconfigs import get_opt
from dataloader.mydataloader import custom_dataloader
from evaluation.mymetric import confusion, report
from executor.train import training_loop
from executor.test import test_loop
from executor.predict import predict
from model.mymodel import optimi, initialize_model, load_model, load_chekpoint
from utils.myutils import seed_everything
from utils.visualize import visualize_acc, visualize_loss
from transform import augs, transfms

# from ..segmentation 


def main(opt):

    # setting variable
    checkpoint_path = opt.checkpoint_path + 'cp'
    model_path = opt.checkpoint_path + 'model'
    result_path = opt.result_path
    feature_extract = opt.feature_extract
    img_path = opt.img_path

    bs = opt.batch_size
    myloader = custom_dataloader(img_path, augs, transfms, bs)
    lr = opt.lr
    num_epochs = opt.num_epochs
    num_classes = opt.num_classes
    

    resnet = initialize_model(num_classes, feature_extract)
    optimizer, scheduler = optimi(resnet,device, feature_extract, lr, num_epochs)

    # TRAIN
    # since = time.time()
    # loss_list, acc_list = training_loop(resnet, optimizer, criterion, scheduler, device, num_epochs, myloader, checkpoint_path, model_path)
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # visualize loss and acc
    _, loss_list, acc_list = load_chekpoint(checkpoint_path + 'ResNet152.pt')
    visualize_loss(loss_list, result_path + 'CXR/lossFTResNet152_CXR_RGB_mean_std_compute.png')
    visualize_acc(acc_list,result_path + 'CXR/ACCFTResNet152_CXR_RGB_mean_std_compute.png')


# TEST
    resnet = load_model(model_path + 'ResNet152.pt')
    y_true, y_pred = test_loop(resnet, device, myloader['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, opt.classes, result_path +'CXR/confusionmatrix_CXR_RGB_mean_std_compute.png')
    report(y_true, y_pred, opt.classes, result_path +'CXR/classification_reportpy152_CXR_RGB_mean_std_compute.txt')
    
    # pred_str = str('')

    # path_image = './pred/covid.jpg'

    # img = Image.open(path_image)
    # plt.imshow(img)

    # predict(path_image,resnet)
    # plt.title('predict:{}'.format(pred_str))
    # plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    # plt.show()


if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)
    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    # resnet = initialize_model(opt.num_classes, opt.feature_extract)

    main(opt)

