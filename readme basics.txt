


Example implementation of Temporal Stochastic Softmax, with inflated vgg for FER in AFEW.

This is intended to help integrating stochastic softmax in other models.

Link to the 2d pretrained vgg model.
Use it with inflate.py to build and save a 3D model
then finetune this vgg_3d.pth model


multiprocessing and files are used to handle communication between the dataloader and the training loop
(synchronize training steps, sampling strategy etc.)
