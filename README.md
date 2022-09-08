# COVID19_segmentation_classification: Classification of COVID-19 from chest X-ray images using Deep Convolution Neural Networks
## Introductions
- Constructed ground-truth lung segmentation masks for the benchmark COVIDx CXR3 dataset.
- The experimental results on original Chest X-ray (CXR), Segmented Lung, CXR without Lung images with three models: ResNet50, VGG19_bn, EfficientNetB7 show the VGG19_bn achieves 99.00% accuracy on Segmented Lung images 
## Requirements
- Ubuntu >= 20.04 LTS
- GeForce RTX 3080
- Python 3.7
- cuda 11.6
- cuDNN 8.3.20
- PyTorch 1.12.0
