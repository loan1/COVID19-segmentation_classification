# import modules
from dataloader.transform import augs, transfms
from dataloader.mydataloader import set_dataloader
from executor.train import fit
from utils.visualize import visualize_train, plot_acc_loss, plot_IoU_DSC
from utils.myutils import seed_everything
from executor.test import test
from executor.predict import dataloaderPre, predict, save_filename, save_lungmask
from model.mymodels import load_checkpoint, set_model, list_model_FPN, list_model_UNetPP, list_modelUNet
from configs.myconfigs import get_opt
from evaluation.mymetric import calculate_metrics


# import lib
import torch
import numpy as np
from torch.optim import lr_scheduler


def train (hist_model, opt, first = True):

 
    if first == False:    
        hist_model = load_checkpoint (opt.checkpoint_path, hist_model)
        hist_model['scheduler'] = lr_scheduler.CosineAnnealingLR(hist_model['optimizer'], T_max=10)

    train_dict, val_dict = fit (hist_model, set_dataloader,  calculate_metrics, opt)
    
    # plot result
    visualize_train(train_dict, val_dict)

    return train_dict, val_dict

def inference (img_path, mask_path, img_np, predict_np, filename, model, opt, train = True, neg = True):

    #load model
    model.load_state_dict(torch.load(opt.model_path)) 

    #predict
    images, y_predict = predict(dataloaderPre(img_path, opt.batch_size, transfms, train, neg), model, device) 

    # luu ket qua inference
    np.save(img_np,images)
    np.save(predict_np,y_predict)

    # save file name
    save_filename(dataloaderPre(img_path, opt.batch_size, transfms, train, neg), filename, opt)

    # save lung mask
    save_lungmask (img_path, train, neg, filename, mask_path , predict_np,opt)

if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)

    opt = get_opt()
    device = torch.device('cuda:0' 
    if torch.cuda.is_available() else 'cpu')

    model = list_model_FPN[5].to(device)
    hist_model = set_model (opt.lr, model)
    hist_model['start_epoch'] = opt.start_epoch
    hist_model['train_dict'], hist_model['val_dict'] = [],[]

  

    # TRAIN
    # train_dict, val_dict = train(hist_model,opt)

    # LOAD CHECKPOINT
    # hist_model = load_checkpoint (opt.checkpoint_path, hist_model)

    #visualize train
    train_loss = [0.1941, 0.0761, 0.0656, 0.0607, 0.0577, 0.0553, 0.0532, 0.051, 0.0498, 0.0486, 0.0483, 0.0485, 0.0486, 0.0492, 0.0496, 0.05, 0.0496, 0.0503, 0.0495, 0.0485, 0.0476, 0.0466, 0.0455, 0.0434, 0.0418, 0.0402, 0.0381, 0.0368, 0.0357, 0.0349, 0.035]
    val_loss = [0.0755, 0.0614, 0.0576, 0.0552, 0.0531, 0.0532, 0.0519, 0.0513, 0.0507, 0.0509, 0.0508, 0.0507, 0.0508, 0.0512, 0.0512, 0.0511, 0.053, 0.0573, 0.0505, 0.051, 0.0499, 0.0508, 0.049, 0.05, 0.0518, 0.0501, 0.0504, 0.0508, 0.0514, 0.0513, 0.0514]
    train_acc = [0.9642, 0.9853, 0.9873, 0.9883, 0.9888, 0.9893, 0.9896, 0.9901, 0.9903, 0.9905, 0.9906, 0.9905, 0.9905, 0.9904, 0.9903, 0.9902, 0.9903, 0.9902, 0.9903, 0.9905, 0.9907, 0.9909, 0.9911, 0.9914, 0.9918, 0.9921, 0.9925, 0.9927, 0.9929, 0.9931, 0.993]
    val_acc = [0.9859, 0.9883, 0.9888, 0.9891, 0.9898, 0.9895, 0.9899, 0.9904, 0.9904, 0.9905, 0.9905, 0.9905, 0.9905, 0.9905, 0.9905, 0.9904, 0.9903, 0.9898, 0.9904, 0.9905, 0.9907, 0.9907, 0.9908, 0.9905, 0.9907, 0.991, 0.9909, 0.991, 0.991, 0.991, 0.991]

    plot_acc_loss(train_loss, val_loss, train_acc, val_acc,'/mnt/DATA/research/project/classificationCOVID19applyseg/result/segmentation/result/Loss_Acc.png')

    train_iou = [0.8682, 0.9392, 0.9472, 0.951, 0.9533, 0.9552, 0.9567, 0.9585, 0.9594, 0.9602, 0.9608, 0.9604, 0.9603, 0.9599, 0.9595, 0.9591, 0.9594, 0.9589, 0.9595, 0.9602, 0.9609, 0.9617, 0.9625, 0.9642, 0.9654, 0.9667, 0.9683, 0.9694, 0.9704, 0.971, 0.9708]
    val_iou = [0.94, 0.9495, 0.9515, 0.9528, 0.9561, 0.9546, 0.9562, 0.9587, 0.9584, 0.959, 0.959, 0.9587, 0.9589, 0.9588, 0.9591, 0.9585, 0.9583, 0.9562, 0.9583, 0.959, 0.9597, 0.9599, 0.96, 0.9587, 0.9602, 0.9611, 0.9608, 0.9609, 0.9612, 0.9611, 0.9611]
    train_dice = [0.9255, 0.9686, 0.9729, 0.9749, 0.9761, 0.9771, 0.9779, 0.9788, 0.9793, 0.9797, 0.98, 0.9798, 0.9797, 0.9795, 0.9793, 0.9791, 0.9793, 0.979, 0.9793, 0.9797, 0.9801, 0.9805, 0.9809, 0.9817, 0.9824, 0.9831, 0.9839, 0.9845, 0.985, 0.9853, 0.9852]
    val_dice = [0.969, 0.974, 0.9751, 0.9758, 0.9775, 0.9767, 0.9775, 0.9788, 0.9787, 0.979, 0.979, 0.9789, 0.9789, 0.9789, 0.9791, 0.9788, 0.9786, 0.9775, 0.9786, 0.979, 0.9794, 0.9795, 0.9795, 0.9788, 0.9796, 0.9801, 0.98, 0.98, 0.9801, 0.9801, 0.9801]

    plot_IoU_DSC(train_iou, val_iou,train_dice, val_dice, '/mnt/DATA/research/project/classificationCOVID19applyseg/result/segmentation/result/IoU_DSC.png')
    # plot_loss(train_loss, val_loss)

    # visualize_train(hist_model)    
    
    # TEST 
    # model.load_state_dict(torch.load(opt.model_path))
    # test_dataloader = set_dataloader(opt.data_path, opt.batch_size, augs, transfms)['test']

    # image, y_true, y_pred, test_dict = test(test_dataloader, device, model, calculate_metrics) 
    # print(test_dict)
    
    # # ghi ket qua
    # with open('/mnt/DATA/research/project/classificationCOVID19applyseg/result/segmentation/result/test_result.txt', 'w') as wf:
    #     wf.writelines(str(test_dict.items()))

    ## INFERENCE

    img_path = '/mnt/DATA/research/project/classificationCOVID19applyseg/dataset/COVIDxCXR3/'
    mask_path = '/mnt/DATA/research/project/classificationCOVID19applyseg/result/segmentation/lung_mask/'
    
    # EDATest_Neg
    # inference(img_path, mask_path + 'EDA_Test/Negative/', mask_path + 'EDA_Test/img_npN.npy', mask_path + 'EDA_Test/y_predictN.npy', mask_path +'EDA_Test/filenameN.txt', model,opt,train = False, neg = True)

    # # EDATest_Pos
    # inference(img_path, mask_path + 'EDA_Test/Positive/', mask_path + 'EDA_Test/img_npP.npy', mask_path +'EDA_Test/y_predictP.npy', mask_path +'EDA_Test/filenameP.txt', model,opt, train=False, neg = False)


    # #EDATrain_Neg
    # inference(img_path, mask_path + 'EDA_Train/Negative/', mask_path + 'EDA_Train/img_npN.npy', mask_path +'EDA_Train/y_predictN.npy', mask_path +'EDA_Train/filenameN.txt', model, opt)

    # #EDATrain_Pos
    # inference(img_path, mask_path + 'EDA_Train/Positive/', mask_path + 'EDA_Train/img_npP.npy', mask_path +'EDA_Train/y_predictP.npy', mask_path +'EDA_Train/filenameP.txt', model,opt, neg = False)

    # EDATrain_Pos (non pos processing)
    # inference(img_path, mask_path + 'EDA_Train/Positivenon/', mask_path + 'EDA_Train/img_npPnon.npy', mask_path +'EDA_Train/y_predictPnon.npy', mask_path +'EDA_Train/filenamePnon.txt', model,opt, train=True, neg = False)

