# COVID19_segmentation_classification: Classification of COVID-19 from chest X-ray images using Deep Convolution Neural Networks
## Introductions
-  Direct classification on chest X-ray images uses existing deep learning models such as [VGG19_bn](https://arxiv.org/pdf/1409.1556.pdf), [ResNet50](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), and [EfficientnetB7](https://arxiv.org/pdf/1905.11946.pdf).
 - Segmentation of chest X-ray images uses [Feature Pyramid Network (FPN) model with DenseNet121 encoder](https://arxiv.org/pdf/1612.03144.pdf), generated ground-truth lung segmentation masks for the benchmark COVIDx CXR3 dataset. Then combining segmented image and original chest x-ray image for classification.
 - The experimental results on original Chest X-ray (CXR), Segmented Lung, CXR without Lung images with three models: ResNet50, VGG19_bn, EfficientNetB7 show the VGG19_bn achieves 99.00% accuracy on Segmented Lung images 
## Requirements
- Ubuntu >= 20.04 LTS
- GeForce RTX 3080
- Python 3.7
- cuda 11.6
- cuDNN 8.3.20
- PyTorch 1.12.0
