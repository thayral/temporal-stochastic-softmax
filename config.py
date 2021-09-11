import argparse


###### modified from github.com/kevinzakka/recurrent-visual-attention


arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg




# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.2,
                      help='Proportion of training set used for validation')
                      # 20% of train data is allocated to validate, 80% to train
data_arg.add_argument('--batch_size', type=int, default=4,
                      help='# of clips in each batch of data')
data_arg.add_argument('--num_frames', type=int, default=16,
                      help='duration of training clips')
data_arg.add_argument('--vgg3d_model_path', type=str, default='/home/thayral/pretrained_models/vgg_3d.pth',
                      help='Path to the 3D VGG model .pth file')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--patience', type=int, default=10,
                       help='Max # of epochs to wait for no validation improv')


train_arg.add_argument('--lr_core', type=float, default=5e-4,
                       help='Learning rate for classification layers trained from scratch')
train_arg.add_argument('--lr_ft', type=float, default=2.5e-6,
                       help='Learning rate for fine-tuned VGG layers')
train_arg.add_argument('--wd_base', type=float, default=1e-3,
                       help='Weight decay')



# Stochastic softmax - Clip sampling params
sampling_arg = add_argument_group('Sampling Params')
sampling_arg.add_argument('--exploration_epochs', type=int, default=5,
                       help='# of epochs of deterministic clip-sampling for initialization')
sampling_arg.add_argument('--warmup_epochs', type=int, default=3,
                       help='# of epochs of uniform sampling warm-up')
sampling_arg.add_argument('--weight_init_value', type=float, default=0.0,
                       help='NOT USED - Value of initialization of sampling distributions')



# other params
misc_arg = add_argument_group('Misc.')

misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
#misc_arg.add_argument('--random_seed', type=int, default=175,
#                      help='Seed to ensure reproducibility')
#misc_arg.add_argument('--data_dir', type=str, default='./data',
#                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')

misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--yo', type=str, default=' no message from user',
	                      help='Message for log')



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
