

import torch

from trainer import Trainer
from config import get_config

import os

import random
import datetime
import math
import numpy

import shutil


def main(config):

    #universal_seed = config.random_seed
    universal_seed = 888

    reset_seeds(universal_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(universal_seed)
    else :
        print("Cuda not available. You need cuda to run this.")
        # or try removing cuda functions
        exit(0)


    # This is for work on GPU Clusters of Compute Canada ..., set your own paths here
    delicate_root_dir = os.environ["DELI_WD"]

    # a dir for temporary files
    config.tmp_dir = os.path.join(  os.environ["SLURM_TMPDIR"] , "work")
    # where to save the model
    config.ckpt_dir = os.path.join(  delicate_root_dir , "ckpt")
    # where to find the dataset
    config.afew_dir = os.path.join(  os.environ["SLURM_TMPDIR"] , "FebSeetaCompleted")
    number_reprod = 2


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    time_id_of_launch = datetime.datetime.now().strftime("%d%m%Y_%Hh%M")


    # The experiment report file
    fneores = os.path.join(delicate_root_dir, "neores.txt" )
    with open(fneores, "a") as neores :
        neores.write("\n"+"################# LIVIA #################")
        neores.write("\n"+ "time_id_of_launch: " +  time_id_of_launch)
        neores.write("\n" + "universal_seed: " + str(universal_seed))


    count = 0


    config.universal_seed = universal_seed


    # some more config parameters
    config.num_classes = 7
    config.resize_size = (256, 256)
    config.crop_size = (224, 224)
    config.scheduler_patience = 5
    config.batch_accumulation_factor = 8

    #inv_temperature = 0.5

    for inv_temperature in [ 0, 1, ]:

        print("\n")

        config.inv_temperature = inv_temperature

        time_id_of_exp = datetime.datetime.now().strftime("%d%m%Y_%Hh%M")


        with open(fneores, "a") as neores :
            neores.write("\n" + "======== EXP "+ str(count) +"  ========")
            neores.write("\n" + "num_frames: " + str(config.num_frames ))
            neores.write("\n" + "batch_size: " + str(config.batch_size ))
            neores.write("\n" + "inv_temperature: " + str(inv_temperature ))


        print ( "======== EXP "+str(count)+" ========")



        reset_seeds(universal_seed)

        exp_epochs = []
        exp_accs = []

        for reprod_cpt in range(number_reprod):


            rep_dir = str(time_id_of_launch)+"_"+ str(time_id_of_exp) + "_rep" + str(reprod_cpt)
            config.log_dir = os.path.join(  delicate_root_dir , rep_dir )
            os.mkdir(config.log_dir)

            main_path = os.path.join(delicate_root_dir, "main.py")
            shutil.copy(main_path, config.log_dir)

            python_rdm_state = random.getstate()


            config.is_train = True
            torch.backends.cudnn.benchmark = True
            Trainer(config).train()


            random.setstate(python_rdm_state)
            config.is_train = False
            torch.backends.cudnn.benchmark = False
            acc, epoch = Trainer(config).test( load_mode = "best")
            config.inv_temperature = inv_temperature

            count+=1 # exp count

            exp_accs.append(acc)
            exp_epochs.append(epoch)




        with open(fneores, "a") as neores :

            mean = sum(exp_accs)/len(exp_accs)
            neores.write("\n"+"acc : "+ str(mean))

            eps_avg = sum(exp_epochs)/len(exp_epochs)
            neores.write("\n"+"epochs : "+ str(eps_avg))


            if len(exp_accs) > 0:
                std = numpy.sqrt(sum((x - mean)**2 for x in exp_accs) / len(exp_accs))
                neores.write("\n"+"std : "+ str(std))

            neores.write("\n"+"accs : "+ str(exp_accs))
            neores.write("\n"+"epochs :"+ str( [str(i) for i in exp_epochs] ) )



def reset_seeds(universal_seed):
    random.seed( universal_seed )
    torch.manual_seed(universal_seed)
    torch.cuda.manual_seed(universal_seed)
    numpy.random.seed(universal_seed)



if __name__ == '__main__':

    config, unparsed = get_config()
    main(config)
