# temporal-stochastic-softmax
Example implementation of Temporal Stochastic Softmax, with inflated vgg for FER in AFEW.

Read our WACV paper for more information: 
https://openaccess.thecvf.com/content/WACV2021/html/Ayral_Temporal_Stochastic_Softmax_for_3D_CNNs_An_Application_in_Facial_WACV_2021_paper.html
https://arxiv.org/abs/2011.05227


This is intended to help integrating stochastic softmax in other models.
Multiprocessing and files are used to handle communication between the dataloader and the training loop (synchronize training steps, sampling strategy etc.).





This project also contains the code for inflating the 2D model to 3D CNN.
Examples of 2D pretrained vgg model: 
https://github.com/XiaoYee/emotion_classification
https://github.com/prabhuiitdhn/Emotion-detection-VGGnet-architecture-fer2013

2D models can be inflated with inflate.py to build and save a 3D model as vgg_3d.pth to then be finetuned.



