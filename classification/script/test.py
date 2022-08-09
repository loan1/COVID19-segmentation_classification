import torch

from PIL import Image
import cv2
import numpy as np

def test_loop(model_ft, device, test_dataloader):
    with torch.no_grad():
        y_true = []
        y_pred = []
        model_ft.to(device)
        model_ft.eval()
        for data, target in test_dataloader:
        # for _,_,_,_,data, target,_ in test_dataloader:
            batch_size = data.size(0)
            data = data.to(device)
            target = target.to(device)
            output = model_ft(data)
            _,pred = torch.max(output, 1)
            y_true += target.tolist()
            y_pred += pred.tolist()
    return y_true, y_pred

def img_transform(path_img, test_transform): 
    img=cv2.imread(path_img, 0)   
    # img = Image.open(path_img)   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = cv2.resize(img,(256,256)) 
    aug = test_transform(image = img)
    res=aug['image']
    res = res.float().cuda()
    return res


            