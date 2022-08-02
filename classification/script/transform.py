# from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from dataset import ImageDataset


def augment(mean, std):
    data_transforms = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees = 0, shear = 0.2),    
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'test' : transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    }
    

