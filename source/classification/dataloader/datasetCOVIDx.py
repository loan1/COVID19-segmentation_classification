#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset): #train tren CXR
    def __init__(self,csv,img_folder,transform): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
        # print(self.img_folder+ self.image_names.iloc[index])

        image=cv2.imread(self.img_folder + self.image_names.iloc[index], 1) #BGR
        # # convert BGR => RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image, dtype=np.float32)

        if self.transform != None:
            aug = self.transform(image = image)
            image=aug['image']

        targets=self.labels[index]
        targets = int(targets)
        targets = torch.tensor(targets, dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        return image, targets # chua 1 cap

class LungImageDataset(Dataset): #train tren Lung + Img
    def __init__(self,csv, img_folder, mask_folder, img_size, train = True, transform = None): # 'Initialization'
        self.csv=csv
        self.transform=transform
        self.img_folder=img_folder
        self.mask_folder = mask_folder
        self.train = train
        self.img_size = img_size
    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= np.array(self.csv[:]['label']) # note kiểu mảng int
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
    
        image=cv2.imread(self.img_folder + self.image_names.iloc[index], 0) # grayscale

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

        _, maskthres = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #nhị phân hóa ảnh

        _, maskinv = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #invert

        # print(image.shape)

        image = cv2.resize(image,(self.img_size,self.img_size))
        maskthres = cv2.resize(maskthres,(self.img_size,self.img_size))
        maskinv = cv2.resize(maskinv,(self.img_size,self.img_size))

        res = cv2.bitwise_and(image,maskthres)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        lung = cv2.resize(res,(self.img_size,self.img_size))

        resnotlung = cv2.bitwise_and(image,maskinv)
        resnotlung = cv2.cvtColor(resnotlung, cv2.COLOR_GRAY2RGB) #them vao
        cxrnotlung = cv2.resize(resnotlung,(self.img_size,self.img_size))

        # print(lung.shape) # (256, 256, 3)
        if self.transform != None:  
            aug = self.transform(image = lung)
            lung=aug['image']
            lung = lung.float()

        # print(lung.shape) # torch.Size([3, 256, 256])
        # print(cxrnotlung.shape) # (256, 256)
        if self.transform != None:  
            aug = self.transform(image = cxrnotlung)
            cxrnotlung=aug['image']
            cxrnotlung = cxrnotlung.float()
 

        name = self.image_names[index]
        targets=self.labels[index]
        targets = torch.tensor(int(targets), dtype=torch.long) #đọc từng phần tử của mảng, chuyển từ array -> tensor; kiểu int64 tương ứng với long trong pytorch

        return image, mask, maskthres, lung, cxrnotlung, targets, name 


# if __name__ == '__main__':
#     val_txt= pd.read_csv('/home/trucloan/LoanDao/COVID_QU_Ex-main/COVIDx/val_set.txt', sep= '\s+', header=None)

#     val_txt.columns = ['file_name', 'label']
#     data = LungImageDataset(val_txt, '/home/trucloan/LoanDao/COVID_QU_Ex-main/COVIDx/train/', '/home/trucloan/LoanDao/COVID_QU_Ex-main/visualize/FPN_DenseNet121/lung_mask',augment()['train'])
#     _,_,img, mask = next(iter(data))








