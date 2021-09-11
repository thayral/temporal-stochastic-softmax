import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg3D import VggFace


class ModelConv3D(nn.Module):
    def __init__(self, vgg3d_model_path, inv_temperature, num_classes, num_frames ):
        super(ModelConv3D, self).__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.inv_temperature = inv_temperature

        self.name = "VGGi3D"
        self.feature_extractor = VggFace(vgg3d_model_path)


        self.preclassifier = nn.Linear(4096, 128)
        self.classifier = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()

        self.classifier_dropout = nn.Dropout(p=0.5)
        print("Model built")


    def forward(self, input):

        x = input
        B,C,L,H,W = x.size()


        x = self.feature_extractor(x) # BCL TO LBC

        x = self.classifier_dropout(x)
        x = self.preclassifier(x)
        x = self.relu(x)
        x = self.classifier(x)

        x = x.permute(1,2,0) #BCL

        # maxpool temporal with 16-frame receptive field -> only one value during training
        x = F.max_pool1d(x, kernel_size = min(L, self.num_frames), stride = 1) # L can be inferior in eval mode

        scores = x # for training we want one score per clip ...
        # this influences validation distrib but we dont care much

        w = F.softmax( x*self.inv_temperature , dim = 2)
        x = x*w
        x = x.sum(2)


        return F.log_softmax(x, dim=1), scores
