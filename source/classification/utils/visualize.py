import matplotlib.pyplot as plt

import numpy as np

def imshow(inp, title=None):
    """imshow for Tensor."""
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std*inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    # plt.show()

def visualize_loss (loss, path_loss):
    train_loss = [x['train_loss'] for x in loss]
    valid_loss = [x['valid_loss'] for x in loss]
    fig, ax = plt.subplots(figsize = (18, 14.5))
    ax.plot(train_loss, '-gx', label='Training loss')
    ax.plot(valid_loss , '-ro', label='Validation loss')
    ax.set(title="Loss over epochs of Model Lung FTResNet152 ",
    xlabel='Epoch',
    ylabel='Loss')
    ax.legend()
    fig.show()
    plt.savefig(path_loss)

def visualize_acc (acclist, path_acc):
    train_acc = [x['train_acc'] for x in acclist]
    valid_acc = [x['valid_acc'] for x in acclist]
    fig, ax = plt.subplots(figsize = (18, 14.5))
    ax.plot(train_acc, '-bx', label='Training acc')
    ax.plot(valid_acc , '-yo', label='Validation acc')
    ax.set(title="Accuracy over epochs of Model Lung FT_ResNet152 ",
    xlabel='Epoch',
    ylabel='Accuracy')
    ax.legend()
    fig.show()
    plt.savefig(path_acc)



# def visualiz():

#     opt = get_opt()
#     # Get a batch of training data
#     _,_,_,image, label,_ = next(iter(dataloader()['train']))
#     fig = plt.figure(figsize=(25, 7))

#     # display batch_size = 40 images
#     for idx in np.arange(opt.BATCH_SIZE):
#         ax = fig.add_subplot(4, opt.BATCH_SIZE/4, idx+1, xticks=[], yticks=[])
#         imshow(image[idx]) # lay 1 cap co nghia la o day show anh
#         ax.set_title(opt.classes[label[idx]]) # vì đã chuyển từ nes/pos -> 0,1 -> tensor 0,1
#     plt.show()