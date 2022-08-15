# import lib

# read file csv
import pandas as pd

# call DataLoader
from torch.utils.data import DataLoader

# import module

from .datasetCOVIDx import ImageDataset, LungImageDataset


def custom_dataloader(data_path, augs, transfms, batch_size):

    # duong dan den cac file txt    
    train_metadata = data_path + 'metadata/train_set.txt'
    val_metadata = data_path + 'metadata/val_set.txt'
    test_metadata = data_path + 'metadata/test_set.txt'

    # duong dan den dataset
    train_path = data_path + 'train/'
    test_path = data_path + 'test/'

    # doc file csv
    train_txt= pd.read_csv(train_metadata, sep= '\s+', header=None)
    val_txt = pd.read_csv(val_metadata, sep = "\s+", header= None)
    test_txt = pd.read_csv(test_metadata, sep= '\s+', header=None)
    
    # gan ten cho cac cot
    train_txt.columns = ["file_name","label"]
    test_txt.columns = ["file_name","label"]
    val_txt.columns = ["file_name","label"]

    # goi dataset CXR
    train_dataset = ImageDataset(train_txt, train_path,augs)
    test_dataset = ImageDataset(test_txt, test_path,transfms)
    val_dataset = ImageDataset(val_txt, train_path,transfms)

    # goi dataset Lung
    # train_dataset = LungImageDataset(train_txt,opt.train_path,opt.mask_path,True,augs)
    # test_dataset = LungImageDataset(test_txt,opt.test_path,opt.mask_path,False,transfms)
    # val_dataset = LungImageDataset(val_txt,opt.train_path,opt.mask_path, True, transfms)

    loader ={
        'train' : DataLoader(
            train_dataset, 
            batch_size = batch_size,
            shuffle = True
        ),
        'val' : DataLoader(
            val_dataset, 
            batch_size = batch_size,
            shuffle = True
        ),
        'test' : DataLoader(
            test_dataset, 
            batch_size = batch_size,
            shuffle = True
        )
    }   
    return loader