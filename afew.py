######
## Inspired by the SVHN example on torchvision
## for AFEW dataset with folders of frame images
#####


import torch

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler, Sampler


import os
from list import list_dir, list_files
from PIL import Image

import numpy as np
import pickle

import multiprocessing as mp

# Thanks to Hassony2 for the pytorch transforms for videos, consistant across frames
import videotransforms.video_transforms as vidtrans
import videotransforms.volume_transforms as voltrans
import videotransforms.tensor_transforms as tentrans


import random


class AFEW(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``AFEW`` exists.
        subset (string): train or test
    """
    ## This is a bit tricky because the train and val come from training folder
    ## and the test comes from val folder

    def __init__(self, config, workers_delight_mp, subset):


        self.workers_delight_mp = workers_delight_mp

        self. tmp_dir = config.tmp_dir

        self.weight_init_value = config.weight_init_value
        self.exploration_epochs = config.exploration_epochs
        self.inv_temperature = config.inv_temperature

        #data
        self.subset = subset  # training set or test set
        self.valid_size = config.valid_size
        # alphabetical order anyway
        self.label_map= {'Angry': 0,'Disgust':1, 'Fear':2, 'Happy':3,
                        'Neutral':4, 'Sad':5, 'Surprise':6}

        self.afew_dir = config.afew_dir



################################################################################
        if self.subset == "train_val":
            self.subset_dir = os.path.join(self.afew_dir,"train")
        elif self.subset == "test":
            self.subset_dir = os.path.join(self.afew_dir,"val")
        else:
            #if self.subset not in ["train_val","test"] :
            raise ValueError('Wrong subset entered! Please use subset="train_val" '
                             'subset="test"')


        if len(list_dir(self.subset_dir)) != 7:
            print("Wrong number of emotion directories ....")
            exit()
################################################################################


        # temporal uniformization of samples : clip sampling
        self.num_frames = config.num_frames # number of frames in each sample of the batch
        print("number of frames in training clips: ", self.num_frames)


        # spatial uniformization of samples : resize and crop
        self.resize_size = config.resize_size
        self.crop_size = config.crop_size


        # split validation and training from train set, store samples folder names (sid)
        if self.subset == "train_val":
            self.fold_splitting()

        # Actual building of the dataset entries, store filenames of frames
        self.build_dataset()


        # Prepare the weighting, initializing
        if self.subset == 'train_val' :
            # clip exploration check boxes and init distribs to uniform
            self.init_sampling_distribution_and_roadmap()


        self.setup_video_transforms()


        print("dataset created")


    def fold_splitting (self):
        # splits the train folder in training and validation samples
        # returns a list of the sids (sample identifier) in valid_split_sids
        # first look into the folders to split the datasets train and valid
        # we split to maintain the emotion distribution in both
        # not exact because of discarded samples down below

        sids_in_emotion = {} # dict of samples (by sample identifiers) in an emotion directory
        valid_split_sids = [] # list of samples in the valid split

        # ['Angry','Disgust','Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        for emotion_dir in list_dir(self.subset_dir,prefix=True):

            emotion_samples = []
            for video_id in list_dir(emotion_dir,prefix=False):
                emotion_samples.append(video_id)
                #print(video_id)
            emotion_name = os.path.basename(os.path.normpath(emotion_dir))
            sids_in_emotion[emotion_name] = emotion_samples

            samples = sids_in_emotion[emotion_name]
            random.shuffle(samples) # this is the random splitting
            valid_split_sids.extend( samples[ : int(self.valid_size*len(samples)) ] )
            #train_idx.extend( samples[ int(self.valid_size*len(samples)) : ] ) # useless

        #print(str(len(valid_idx)) + " samples for valid...")
        #print(str(len(train_idx)) + " samples for train...")

        # sids_in_emotion contains the entire videos list for each emotion
        # valid_split_sids contains the sub group that is assigned to validation subset
        self.valid_split_sids = valid_split_sids




    def build_dataset (self):

        # create dataset entries for all samples
        # load the filenames of each frames and store some info abour length etc

        if self.subset == 'train_val':
            self.train_indices = []
            self.valid_indices = []


        self.dataset = [] # the main collection


        for emotion_dir in list_dir(self.subset_dir,prefix=True):

            sample_emotion = os.path.basename(emotion_dir)
            print("Loading emotion :" , sample_emotion)

            for video_dir in list_dir(emotion_dir,prefix=True):

                #nb_available_frames = len(list_files(video_dir,".png"))
                file_list = list_files(video_dir,".png",prefix=True) # stores frames path
                video_name = os.path.basename(os.path.normpath(video_dir))

                # flag to know if sample is in validation, train or test set
                if self.subset == "train_val":
                    if video_name in self.valid_split_sids :
                        group = "validation"
                    else:
                        group = "training"
                else:
                    group = "test"



                if self.subset == "train_val" and len(file_list) < 1:
                    # discard empty sample... face detection failed or something
                    # in train set only for no cheat
                    continue


                if group == "training" :
                    self.train_indices.append(len(self.dataset))
                elif group == "validation" :
                    self.valid_indices.append(len(self.dataset))

                sample_label = self.label_map[sample_emotion]


                sample = {'files': file_list, 'label': sample_label, 'length': len(file_list), 'group': group, 'sid': video_name}
                self.dataset.append(sample)


        print("afew loaded : " + str(len(self.dataset))+ " samples")
        print(self.subset)




    def init_sampling_distribution_and_roadmap(self):

        # We loop through the entire dataset to initialize temporal distributions of the correct size for sampling clips
        # and the roadmap is a list of positions for the deterministic exploration that initializes the distributions

        # keep checks of what clips we saw to have smooth exploration in warmup...
        self.exploration_roadmap_mp = {} # a dic of shared arrays, sid : array roadmap

        # The running estimates of clip scores (weights for softmax sampling distributions)
        self.temporal_weights_dict_mp = {} # dict of shared distribs , sid : array distrib

        for sample in self.dataset :
            if sample['group'] == 'training':

                # for each epoch of deterministic exploration,
                # a flag tells us if we checked the n^th clip
                self.exploration_roadmap_mp[sample['sid']] = mp.Array('i', self.exploration_epochs) # is zeroed

                # just an array for each training video,
                # with a score value for each possible clip (video_length-clip_length+1)
                # init value is useless since we use deterministic exploration to score "all" clips
                self.temporal_weights_dict_mp[sample['sid']] = mp.Array('f', max( 1, sample['length']-(self.num_frames-1)) ) # is zeroed

        print("Initialized temporal sampling distributions for "+ str(len(self.temporal_weights_dict_mp))+" videos.")



    def setup_video_transforms (self):

        print("Preparing video transforms...")

        self.mean = torch.FloatTensor( [0.2572, 0.2000, 0.1644])
        self.std = torch.FloatTensor([1,1,1])

        self.num_channels = 3

        self.pad_path = None # just put zeros


        # Initialize transforms
        video_transform_list = [
            vidtrans.RandomHorizontalFlip(),
            vidtrans.RandomRotation(20),
            vidtrans.Resize(self.resize_size),
            vidtrans.RandomCrop(self.crop_size),
            vidtrans.ColorJitter(0.2, 0.2, 0.2, 0.1),
            voltrans.ClipToTensor(channel_nb=self.num_channels),
            #transforms.Normalize(mean, std)
            # we normalise in getitem because videos are not supported ?
            ]

        # Initialize transforms
        ResAndTens = [
            vidtrans.Resize(self.resize_size),
            vidtrans.CenterCrop(self.crop_size),
            voltrans.ClipToTensor(channel_nb=self.num_channels),
            ]


        # Transforms for train (data augmentation) and for eval
        self.no_aug_transform = vidtrans.Compose(ResAndTens)
        self.yes_aug_transform = vidtrans.Compose(video_transform_list)



    def __getitem__(self, index):


        # this getitem has lots of if cases because of different clip-sampling phases
        # eval vs training, and warmup / exploration / stochastic_softmax


        sample = self.dataset[index]
        sid = sample['sid']
        file_list = sample['files']
        sample_frames = [] # frames that will be sampled for training clip


        if sample['group'] == "training" :
            transform = self.yes_aug_transform
        else:   # val and test
            transform = self.no_aug_transform

        self.test_max_num_frames = 999

        idx_explo = -1 # an index to know which clip has been explorated during deterministic exploration, init -1 if not used ...


        if sample['group'] == "test" or sample['group']== "validation":
        # test and validation
        # we do not uniformize samples : full length inference, batch = 1
        # validation on short clips is nice for time savings through
        # just add workers and batch size below, and remove valid from this if statement

            # we dont pad in eval mode
            if len(file_list) <= self.test_max_num_frames :
                clip_start = 0
                sample_frames = file_list
            else :
                # crop in time for memory limit if VERY long video
                clip_start = random.randint(0, len(file_list)-self.test_max_num_frames)
                clip = [file_list[f] for f in range(clip_start, clip_start + self.test_max_num_frames)]
                sample_frames.extend(clip)


        else :
        # train with clip-sampling, batches ...
        # contiguous clips




            # We read the sampling phase (warmup, exploration, stochastic_softmax)
            # from a file that synchronizes the dataset.sampler.getitem workers with the training loop
            fn = os.path.join(self.tmp_dir, "workers_delight.wut")
            with open( fn , "r") as file:
                s = file.read()

            if s == "stochastic_softmax" :
                self.sampling_phase = "softmax"

            elif s == "exploration":
                self.sampling_phase = "explore"

            elif s == "warmup":
                self.sampling_phase = "warm-up"


            else :
                print("warmup worker delight file corrupt ? unknown sampling phase")
                exit()


            if (self.workers_delight_mp[:] != self.sampling_phase) and sample['group']!='test':
                print("mp sampling phase dont match")
                print("in wut is : ", self.sampling_phase)
                print("in mp is : ", self.workers_delight_mp[:])
                exit()




            if len(file_list) <= self.num_frames:
            # pad training clip if shorter

                #zero-pad if shorter, pad_path is just None, will be black image
                clip_start = 0
                idx_explo = 0

                sample_frames.extend( file_list )
                sample_frames.extend( [self.pad_path] * (self.num_frames-len(file_list)) )


            else :
            # training-clip sampling from long videos

                #weighted temporal sampling
                if sample['group'] == "training":
                    # Always true, but simplifies modifications

                    # now we have several strategies depending on phases :
                    # warmup, exploration, stochastic_softmax


                    if not ( len(self.temporal_weights_dict_mp[sid][:]) == len(file_list)-(self.num_frames-1) or len(self.temporal_weights_dict_mp[sid][:]) == 1) :
                        # number of weights in temporal distribution doesnt match with video length
                        # this means init_sampling_distribution_and_roadmap failed ?
                        # or something broken
                        print(" ERROR : afew.py : the samples do no correspond in getitem and the temporal_weights mp infos ...")
                        print(self.temporal_weights_dict_mp[sid].size)
                        print(len(file_list))
                        exit(0)


                    weights = self.temporal_weights_dict_mp[sid]



                    if self.sampling_phase == "warm-up":
                        # random uniform sampling

                        clip_start = random.randint(0, len(file_list)-self.num_frames)
                        clip = [file_list[f] for f in range(clip_start, clip_start + self.num_frames)]
                        sample_frames.extend(clip)


                    elif self.sampling_phase == "explore" :

                        # the exploration_roadmap specifies which clip have not been explored
                        # random choice from the avilable ones, working with indexes

                        clips_to_explore = [idx for (idx, explored) in enumerate(self.exploration_roadmap_mp[sid]) if not explored]

                        idx = random.choice(clips_to_explore)
                        idx_explo = idx # to indicate this idx has now been explored

                        self.exploration_roadmap_mp[sid][idx_explo] = 1 # from 0 to 1, 1 flag means explored ... I wanted bools but idk how to multiprocess that, not that I know with arrays


                        # translate roadmap idx to frame position
                        clip_start = int( (len(file_list)-(self.num_frames-1)-1) * idx /(self.exploration_epochs-1))
                        # de 0 en t0 à nbclips-1 en t= nbepochs-1
                        clip = [file_list[f] for f in range(clip_start, clip_start + self.num_frames)]
                        sample_frames.extend(clip)


                    elif self.sampling_phase == "softmax" :

                        # THIS IS TEMPORAL STOCHASTIC SOFTMAX SAMPLING
                        weights_tensor = torch.FloatTensor(weights) * self.inv_temperature
                        distrib = torch.distributions.Categorical( logits = weights_tensor) # logits : it is turned into probas with softmax
                        clip_start = distrib.sample().item() # sample the clip position with the softmax distribution based on the temporal weights (scores)
                        clip = [file_list[f] for f in range(clip_start, clip_start + self.num_frames)]
                        sample_frames.extend(clip)

                    else :
                        print("error : unknown sampling phase for temporal stochastic softmax, in afew.py")
                        exit()

                else :
                # Never used, but can be used for short clip validation (faster)
                # Uniform sampling
                # normal case, inference, uniform sampling

                    clip_start = random.randint(0, len(file_list)-self.num_frames)
                    clip = [file_list[f] for f in range(clip_start, clip_start + self.num_frames)]
                    sample_frames.extend(clip)




        # end of the switch to decide the frames we take, now we load for real
        # load images from frame file path

        # create a padding frame, in case it is needed
        pad_array = np.zeros((self.crop_size[0],self.crop_size[1],3),np.uint8)
        sample_images = []

        for frame_file in sample_frames:
            #print(frame_file)
            if frame_file is None:
                image = Image.fromarray(pad_array)
            else:
                image = Image.open( frame_file ) #.convert('RGB') #rgb for channeles first
            sample_images.append(image)

        sample_tensor = transform(sample_images) # data-augment or not depending on eval or train
        #sample_tensor_no_aug = self.no_aug_transform(sample_images) # also give no_aug tensor if you want to score on clean samples

        # normalisation
        sample_data = [(sample_tensor[c]-self.mean[c])/self.std[c] for c in range(self.num_channels)]
        #sample_data = [sample_tensor[c] for c in range(self.num_channels)]
        # we normalise on cpu...

        sample_data = torch.stack(sample_data)

        # alternative padding method, maybe better to normalize before 0-padding
        #sample_data = F.pad(sample_data, (0,0, 0,0, 0,self.num_frames - sample_data.size(1)), mode = 'constant', value=0 )

        #sample_data_no_aug = [(sample_tensor_no_aug[c]-self.mean[c])/self.std[c] for c in range(self.num_channels)]
        #sample_data_no_aug = torch.stack(sample_data_no_aug)


        # This is to be sent to the training loop
        loaded_item = {'data': sample_data, 'label': sample['label'],
                'sid': sample['sid'], # ID / name of the video
                'temporal_position': clip_start, # positon of the clip, for stochastic sampling : update the distributions with obtained scores
                'idx_explo' : idx_explo, # update exploration roadmap
                }


        return loaded_item





    def __len__(self):
        return len(self.dataset)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.afew_dir)
        return fmt_str




############################################################################
###### Data_loader modified from github.com/kevinzakka/recurrent-visual-attention
############################################################################



def get_train_valid_loader(config, workers_delight_mp):


    error_msg = "[!] config.valid_size should be in the range [0, 1]."
    assert ((config.valid_size >= 0) and (config.valid_size <= 1)), error_msg


    # load dataset
    dataset = AFEW(config, workers_delight_mp, subset="train_val")

    random.shuffle(dataset.valid_indices)
    random.shuffle(dataset.train_indices)

    train_sampler = SubsetRandomSampler(dataset.train_indices)
    valid_sampler = SubsetNotRandomSampler(dataset.valid_indices)


    print("Train batch size : ", config.batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last = True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=valid_sampler,
        num_workers=1, pin_memory=True,
    )

    return (train_loader, valid_loader), dataset.temporal_weights_dict_mp





def get_test_loader(config, workers_delight_mp):

    # load dataset
    dataset = AFEW( config, workers_delight_mp = None, subset="test")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
    )

    return data_loader




# for validation, more readable ~
class SubsetNotRandomSampler(Sampler):
    """Samples elements not randomly from a given list of indices, without replacement.

    Arguments:
    indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
