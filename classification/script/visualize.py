import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
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

def confusion (y_true, y_pred, classes, path):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    fix, ax = plt.subplots(figsize = (10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = classes)
    disp.plot(include_values = True, cmap = 'viridis_r', ax = ax, xticks_rotation = 'vertical')
    plt.savefig(path)

def report(y_true, y_pred, classes, path):
    # path_rp = '../report/reportFT_ResNet50_152.txt'
    path_rp = path
    try:
        s = classification_report(y_true, y_pred, target_names = classes)
        with open(path_rp, mode ='w+') as f:
            f.write(s)
        with open(path_rp) as f:
            print(f.read())
        f.close()
    except FileExistsError:
        pass