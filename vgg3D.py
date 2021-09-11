
##This  is VGG face  i3d


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import os

import math

from torchvision.models import vgg16
#import collections
from inflate import inflate_vgg_features


class VggFace(nn.Module):
    def __init__(self, vgg3d_model_path):
        super(VggFace, self).__init__()
        #VGG-VD-16 trained on VGG-Faces by S. Albanie (Oxford)


        loaded_model = vgg16(False, num_classes=7) # not pretrained
        #original_model = vgg16(pretrained = True) # remove the 1000 features classifier
        #'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

        self.is_3d = False


#==========================================
# TRANSFER THE VGG FACE PRETRAINED MODEL TO PYTORCH
#=======================================

#        w = torch.load('models/best7164.pth')
#        m = loaded_model.state_dict()
#        name = [k for k,_ in m.items()]
#        weights = collections.OrderedDict()

#        for idx ,(k,v) in enumerate(w.items()):
#          if k == 'fc6.weight':
#                v = v.contiguous().view(4096, -1)
#          weights[name[idx]] = v

#        loaded_model.load_state_dict(weights)
#        torch.save(loaded_model.state_dict(), 'models/vgg_to_pytorch.pth')
#        self.features = loaded_model.features
#==============================================


#=================
# load 2d vgg pytorch
#========================
#        dict = torch.load("models/vgg_to_pytorch.pth")
#        loaded_model.load_state_dict(dict)
#=======================


#===================================
# Transform pytorch VGG into pretrained and inflated
#======================================
        self.features = loaded_model.features
        self.inflate_features() # we only inflate the convolutional parts
        reused = list(loaded_model.classifier.children())[:-1]
        # This is the linear classifier, we do not inflate this (we reapeat across time, L is arbitrary)
        self.reused_classifier = torch.nn.Sequential(*reused)
#=======================================

#=======================
# SAVE THE 3D MODEL
#=====================
#        state = {'is_3d': self.is_3d,
#             'features': self.features.state_dict(),
#             'reused_classifier': self.reused_classifier.state_dict(),
#             }
#        torch.save(state, 'models/vgg_3d.pth')
#=====================


#=================================
# LOAD THE 3D PRETRAINED MODEL
#=====================================
        ckpt = torch.load(vgg3d_model_path)

        # load variables from checkpoint
        self.features.load_state_dict( ckpt['features'])
        self.reused_classifier.load_state_dict( ckpt['reused_classifier'])
#===============================================

        for p in self.features.parameters():
          p.requires_grad = True
        for p in self.reused_classifier.parameters():
          p.requires_grad = True

        loaded_model = None

        self.relu = nn.ReLU()


    def inflate_features(self):

        self.features = inflate_vgg_features(self.features)
        self.features.cuda()



    #  THIS DOES B*C*L*H*W TO L*B*C
    def forward(self, x):

        B,C,L,H,W = x.size()

        x = self.features(x) #BCLHW features

        x = x.permute(2,0,1,3,4) # LBC..
        x = x.contiguous().view(L,B, 25088)
        # L,B ,C is seen as L*B,C for the linear module

        x = self.reused_classifier(x)

        x = self.relu(x)

        return x
