import torch
import torch.nn.functional as F

from torch import autograd
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy
import math

import os
import time
import shutil
import pickle


import multiprocessing as mp

from utils import AverageMeter

from Classifier3D import ModelConv3D

from afew import get_test_loader, get_train_valid_loader




##################
#### Trainer code inspired from  kevinzakka
############


class Trainer(object):

    def __init__(self, config):

        self.config = config

        self.tmp_dir = config.tmp_dir
        self.log_dir = config.log_dir

        self.num_classes = config.num_classes
        self.num_frames = config.num_frames

        # training params
        self.epochs = config.epochs
        self.start_epoch = 1

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.batch_accumulation_factor = config.batch_accumulation_factor
        self.patience = config.patience
        self.resume = config.resume
        self.exploration_epochs = config.exploration_epochs
        self.warmup_epochs = config.warmup_epochs


        # init stuff
        self.best_valid_acc = 0.
        self.best_valid_loss = float("inf")
        self.counter = 0



        self.workers_delight_mp = mp.Array('u', 'warm-up')

        # Data loaders

        if config.is_train == False :

            test_loader = get_test_loader(config, self.workers_delight_mp)

            self.test_loader = test_loader
            self.num_test = len(self.test_loader.dataset)

        if config.is_train:

            data_loader, self.temporal_weights_dict_mp = get_train_valid_loader( config, self.workers_delight_mp)

            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)



        self.model_name = 'vgg3d_'

        # build i3d VGG model
        self.model = ModelConv3D(config.vgg3d_model_path, config.inv_temperature, self.num_classes, self.num_frames)


        self.use_gpu = True
        if self.use_gpu:
            if torch.cuda.is_available():
                self.model.cuda()
            else:
                print("no cuda available")
                exit()
        else :
            print("use_gpu=False but this wont work without using gpu...")
            exit()



        print('[*] Number of vgg parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        pcount=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("train params: ", pcount)



        self.optimizer = SGD(
                [
                {'params': self.model.preclassifier.parameters(), 'lr': config.lr_core ,'weight_decay': config.wd_base},
                {'params': self.model.classifier.parameters(), 'lr': config.lr_core ,'weight_decay': config.wd_base},

                {'params': self.model.feature_extractor.features.parameters(), 'lr': config.lr_ft, 'weight_decay': config.wd_base/10},
                {'params': self.model.feature_extractor.reused_classifier.parameters(), 'lr': config.lr_ft, 'weight_decay': config.wd_base/10},
                ] , momentum=0.9)


        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=config.scheduler_patience, factor = 0.5
        )

        self.model.feature_extractor.log_dir = config.log_dir



    def train(self):

        # This is for visualizations and log
        self.sampling_heatmaps = {}
        self.sampling_distributions = {}


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )


        if self.resume:
            print("best valid loss not resumed DOESNT WORK")
            self.load_checkpoint(best=False)


        fn = os.path.join(self.tmp_dir, "workers_delight.wut")
        with open( fn , "w") as file:
            file.write("warmup")

        # evaluate on validation set
        valid_loss, valid_acc =   self.validate(-1)
        print("validation before train is : ", valid_acc )



        timer_epoch_start = time.time()



        # The actual training iteration
        for epoch in range(self.start_epoch, self.epochs+1):


            timer_epoch_end = time.time()
            print("epoch time : ", timer_epoch_end - timer_epoch_start )
            timer_epoch_start = timer_epoch_end


            # Select training phase :
            # warmup, exploration or weighted sampling (stochastic_softmax)
            # and write it in a file for the data loader to read
            if epoch >= self.warmup_epochs+self.exploration_epochs+1 :
                self.sampling_phase = "stochastic_softmax"
                self.workers_delight_mp[:] = "softmax"
            elif epoch >= self.warmup_epochs+1 :
                self.sampling_phase = "exploration"
                self.workers_delight_mp[:] = "explore"

            else  : # start without weighting
                self.sampling_phase = "warmup"
                self.workers_delight_mp[:] = "warm-up"

            fn = os.path.join(self.tmp_dir, "workers_delight.wut")
            with open( fn , "w") as file:
                file.write( self.sampling_phase )


            print( '\nEpoch: {}/{} '.format(  epoch, self.epochs))


            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # reduce lr if validation loss plateaus
            self.scheduler.step(valid_loss)

            is_best_acc = valid_acc > self.best_valid_acc
            is_best_loss = valid_loss < self.best_valid_loss
            is_best = is_best_loss # or is_best_acc

            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"

            if is_best_loss:
                msg2 += " [+]"
            if is_best_acc:
                msg2 += " [*]"

            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))


            # check for improvement
            if not is_best:
                self.counter += 1
            else:
                self.counter = 0 # reset patience


            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.best_valid_loss = min(valid_loss, self.best_valid_loss)


            if is_best :

                self.save_checkpoint(
                    {'epoch': epoch,
                     'model_state': self.model.state_dict(),
                     'optim_state': self.optimizer.state_dict(),
                     'best_valid_acc': self.best_valid_acc,
                     'best_valid_loss': self.best_valid_loss,
                     }, is_best
                )


            if self.counter > self.patience:
                print("[!] No improvement in a while, stopping training.")
                break


        print(".................. End of training ....................")

        # save stochastic softmax data for visualizations
        fn = os.path.join(self.log_dir, "sampling_heatmaps.pkl")
        with open( fn , "wb") as file:
            pickle.dump(self.sampling_heatmaps, file)

        fn = os.path.join(self.log_dir, "sampling_distributions.pkl")
        with open( fn , "wb") as file:
            pickle.dump(self.sampling_distributions, file)



    def forward_pass(self, x, target):
        # for training and validation

        # B, C, L, H, W
        #print(x.shape)

        log_probas, scores = self.model(x) #LBC scores

        loss = F.nll_loss(log_probas, target)

        predicted = torch.max(log_probas, 1)[1]

        # compute accuracy
        correct = (predicted == target).float()
        acc = 100 * (correct.sum() / len(target)) # sum over batch dim, class dim is one-hot

        return acc, loss, scores



    def dont(self):
        pass



    def train_one_epoch(self, epoch):

        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()

        self.model.train()

        self.optimizer.zero_grad()

        for batch_idx , batch in enumerate(self.train_loader):

            x = batch['data']
            target = batch['label']

            current_batch_size = target.shape[0] # current batch size

            if self.use_gpu:
                x = x.cuda()
                target = target.cuda()

            acc, loss, scores = self.forward_pass(x, target)

            # For updating distributions with "clean" clips (without data augmentation)
            # requires modifications in afew.py, getitem has to return the no_aug
            #self.model.eval()
            #with torch.no_grad():
            #    _, _, scores = self.forward_pass(batch['no_aug'].cuda(), target)
            #self.model.train()


            temporal_position = batch['temporal_position'] # this is an array of temporal positions of training-clips in this batch
            sample_ids = batch['sid'] # all sids (sample identifiers) for the samples in this batch

            # all clip samples in batch
            for s in range(current_batch_size):
                # s is a batch index, we need to loop through the batch to update distributions

                sid = sample_ids[s]
                tp = temporal_position[s].item()


                if self.sampling_phase == "stochastic_softmax"  :

                    # update weights distrib, propagate around the sampled position
                    sampled_score = scores[s, target[s]] # score for this sample and the correct target class (Oracle)

                    w_size = len(self.temporal_weights_dict_mp[sid]) # corresponds to number of clips possible in the video

                    # propagate the score update to the neighbouring clips (based on num_frames distance)
                    propag_min = int( max(0, (tp-self.num_frames) ) )
                    propag_max = int ( min( (tp+self.num_frames), w_size) ) # -1 cause in range

                    for i in range  ( propag_min, propag_max ):
                        update_weight = (self.num_frames - abs(tp-i)) / float(self.num_frames) # a percentage of modification of the existing value
                        self.temporal_weights_dict_mp[sid][i] = self.temporal_weights_dict_mp[sid][i] + update_weight * (sampled_score - self.temporal_weights_dict_mp[sid][i])
                        # this is like linear interpolation between existing values (running estimates) and new one


                elif self.sampling_phase == "exploration" :

                    # update weights distrib with propagate
                    sampled_score = scores[s, target[s]]

                    w_size = len(self.temporal_weights_dict_mp[sid])

                    # propagate
                    nb_nonscored = max(0,(w_size-self.exploration_epochs))
                    prop_dist = math.ceil(  (nb_nonscored / (self.exploration_epochs-1)) /2 ) # distance of propagation, on both sides

                    propag_min = int( max(0, (tp-prop_dist) ) )
                    propag_max = int ( min((tp+prop_dist+1), w_size) ) # -1 cause index in range

                    for i in range  ( propag_min, propag_max ):
                        self.temporal_weights_dict_mp[sid][i] = sampled_score
                        # we update hard, 100%, because it is not initialized


                # for visualizations
                # log the heatmaps and distrib
                if sid not in self.sampling_heatmaps :
                    self.sampling_heatmaps[sid] = []
                    self.sampling_distributions[sid] = []

                self.sampling_heatmaps[sid].append(tp)
                self.sampling_distributions[sid].append(    numpy.copy(self.temporal_weights_dict_mp[sid])    )


            # Log training infos
            # Update loss and accuracy
            losses.update(loss.data.item(), current_batch_size)
            accs.update(acc.data.item(), current_batch_size)


            # compute gradients and update SGD
            loss.backward()
            # this is accumulation of gradients accross batches ...
            if (batch_idx+1)%self.batch_accumulation_factor == 0 :
                self.optimizer.step()
                self.optimizer.zero_grad()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)


        # return performance to the main loop, to know when to stop training
        return losses.avg, accs.avg




    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        self.model.eval()


        with torch.no_grad():
            for _, batch in enumerate(self.valid_loader):


                x = batch['data']
                target = batch['label']

                current_batch_size = target.shape[0] # current batch size

                if self.use_gpu:
                    x, target = x.cuda(), target.cuda()

                # we dont use scores in validation
                acc, loss, _ = self.forward_pass(x, target)

                losses.update(loss.data.item(), current_batch_size)
                accs.update(acc.data.item(), current_batch_size)

        return losses.avg, accs.avg





    def test(self, load_mode):


        if load_mode == "best":
            best_epoch = self.load_checkpoint(best = True)
            print("best loaded")
        elif load_mode == "latest":
            self.load_checkpoint(best = False)
            print("latest loaded")
        elif load_mode != "no":
            print("wrong load mode in test")
            exit()

        print("Testing --- ")

        per_class_pred = numpy.zeros(self.num_classes)

        correct = 0 # counter of correct classification
        self.model.eval()

        with torch.no_grad():

            for _, batch in enumerate(self.test_loader):

                #torch.cuda.empty_cache()

                x = batch['data']
                target = batch['label']

                if self.use_gpu:
                    x, target = x.cuda(), target.cuda()

                log_probas, _ = self.model(x)
                predicted = torch.max(log_probas, 1)[1]

                for i in range(len(x)):
                    per_class_pred[predicted[i]]+=1

                correct += (predicted == target).float().sum()


        acc = 100 * (correct / self.num_test)

        print(per_class_pred)

        print("correct : ",correct.item())
        print(self.num_test)
        print(acc.item())


        return acc.item(), best_epoch



### here go the save and load checkpoints function


    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model.name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)


        filename = self.model.name + '_model_best.pth.tar'
        shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
        )


    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model.name + '_ckpt.pth.tar'
        if best:
            filename = self.model.name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.best_valid_loss = ckpt['best_valid_loss']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
            print("best valid loss : ", self.best_valid_loss)

        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

        return ckpt['epoch']
