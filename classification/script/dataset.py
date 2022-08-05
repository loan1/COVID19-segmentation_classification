#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
from PIL import Image
import cv2
from matplotlib.pyplot import flag
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

# import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class ImageDataset(Dataset): #train tren CXR
    def __init__(self,csv,img_folder,transform): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int đúng không?
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
        # print(self.img_folder + self.image_names.iloc[index])
    
        image=Image.open(self.img_folder + self.image_names.iloc[index]).convert('RGB')
        # image=cv2.imread(self.img_folder + self.image_names.iloc[index], 0) #GRAYSCALE
        # image=cv2.imread(self.img_folder + self.image_names.iloc[index], 1) #BGR
        # # convert BGR => RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        # print(type(image))

        if self.transform != None:
            aug = self.transform(image = image)
            image=aug['image']

        # image=self.transform(image)
        targets=self.labels[index]
        # print(type(targets))
        targets = int(targets)
        # print(type(targets))
        targets = torch.tensor(targets, dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        return image, targets # chua 1 cap

class LungImageDataset(Dataset): #train tren Lung
    def __init__(self,csv, img_folder, mask_folder, train = True, transform = False): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
        self.mask_folder = mask_folder
        self.train = train
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int đúng không?
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
    
        image=cv2.imread(self.img_folder + self.image_names.iloc[index], 0)

        # image = np.array(image, dtype=np.float32)

        if self.train  == True:
            flag = 'EDA_Train'
        else:
            flag = 'EDA_Test'

        if self.labels[index] == 0:
            lab = '/Negative/'
        else:
            lab = '/Positive/'
        
        # print(self.labels[index])
        # print(lab)
        # print('self.mask_folder: ',self.mask_folder + flag + lab + self.image_names.iloc[index])

        mask = cv2.imread(self.mask_folder + flag + lab + self.image_names.iloc[index], 0)
        # print(mask)


        _, maskthres = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #nhị phân hóa ảnh

        image = cv2.resize(image,(256,256))
        maskthres = cv2.resize(maskthres,(256,256))
        # print(image.shape)
        # print(maskthres.shape)
        

        res = cv2.bitwise_and(image,maskthres)
        # print(type(res))
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        # print(type(res))
        res = cv2.resize(res,(256,256))
        # print(res.shape)
        # res = res.transpose((2,0,1)) #ham cua numpy
        # res = torch.tensor(res)
        # res = torch.transpose(res,2,0)
        # res = np.array(res, dtype=np.float32)
        # print(type(res))

        if self.transform != False:  
            aug = self.transform(image = res)
            res=aug['image']
            res = res.float()
 

        name = self.image_names[index]
        targets=self.labels[index]
        targets = torch.tensor(int(targets), dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        # print(type(res))
        # print(type(targets))

        return image, mask, maskthres, res, targets, name # chua 1 cap

class CXRImageDataset(Dataset): #train tren not lung
    def __init__(self,csv, img_folder, mask_folder, transform = False): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
        self.mask_folder = mask_folder
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int đúng không?
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
        # print(self.img_folder + self.image_names.iloc[index])
    
        # image=cv2.imread(self.img_folder + self.image_names.iloc[index], 0)
        image=cv2.imread(self.img_folder + self.image_names.iloc[index], 1) #BGR
        # convert BGR => RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        image = cv2.resize(image,(256,256))

        if self.labels[index] == 0:
            lab = '/Negative/'
        else:
            lab = '/Positive/'

        print('self.mask_folder', self.mask_folder)

        mask = cv2.imread(self.mask_folder + lab + self.image_names.iloc[index], 0)
        _, maskthres = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        res1 = cv2.bitwise_and(image,maskthres)
        res1 = cv2.cvtColor(res1, cv2.COLOR_GRAY2RGB)

        res1 = cv2.resize(res1,(256,256))
        # res = res.transpose((2,0,1)) #ham cua numpy
        res1 = torch.tensor(res1)
        res1 = torch.transpose(res1,2,0)

        if self.transform != False:       
            aug = self.transform(image = res1)
            res1=aug['image']
            res1 = res1.float()

        _, maskinv = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #nhị phân hóa ảnh       

        res = cv2.bitwise_and(image,maskinv)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

        res = cv2.resize(res,(256,256))
        # res = res.transpose((2,0,1)) #ham cua numpy
        res = torch.tensor(res)
        res = torch.transpose(res,2,0)

        if self.transform != False:       
            aug = self.transform(image = res)
            res=aug['image']
            res = res.float()
       

        name = self.image_names[index]
        targets=self.labels[index]
        targets = torch.tensor(int(targets), dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        return image, mask, maskinv, res1, res, targets, name


def augment():
    data_transforms = {
        'train' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomAffine(degrees = 0, shear = 0.2),    
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225])),
        ]),
        'val' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225]))
        ]),
        'test' : transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  std = np.array([0.229, 0.224, 0.225]))
        ])
    }
    return data_transforms



if __name__ == '__main__':
    val_txt= pd.read_csv('/home/trucloan/LoanDao/COVID_QU_Ex-main/COVIDx/val_set.txt', sep= '\s+', header=None)

    val_txt.columns = ['file_name', 'label']
    data = LungImageDataset(val_txt, '/home/trucloan/LoanDao/COVID_QU_Ex-main/COVIDx/train/', '/home/trucloan/LoanDao/COVID_QU_Ex-main/visualize/FPN_DenseNet121/lung_mask',augment()['train'])
    _,_,img, mask = next(iter(data))








