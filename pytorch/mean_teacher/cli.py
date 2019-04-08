import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
    parser.add_argument('--dataset', metavar='DATASET', default='conll',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: conll)')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='dev',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default=None, type=str, #metavar='FILE',
                        help='list of image labels (default: based on directory structure) OR \% of labeled data to be used for the NLP task (randomly selected)')
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='simple_MLP_embed',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--checkpoint-epochs', default=10, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--wordemb-size', default=100, type=int,
                        help='size of the word-embeddings to be used in the simple_MLP_embed model (default: 300)')
    parser.add_argument('--hidden-size', default=100, type=int, #was 200
                        help='size of the hidden layer to be used in the simple_MLP_embed model (default: 50)')
    parser.add_argument('--pretrained-wordemb', default=True, type=str2bool, metavar='BOOL',
                        help='Use pre-trained word embeddings to be loaded from disk, if True; else random initialization of word-emb (default: True)')
    parser.add_argument('--pretrained-wordemb-file', type=str, default='glove.6B.100d.txt',
                        help='pre-trained word embeddings file')
    parser.add_argument('--update-pretrained-wordemb', default=False, type=str2bool, metavar='BOOL',
                        help='Update the pre-trained word embeddings during training, if True; else keep them fixed (default: False)')
    parser.add_argument('--random-initial-unkown', default=False, type=str2bool, metavar='BOOL',
                        help='Randomly initialize unkown words embedding. It only works when --pretrained-wordemb is True')
    parser.add_argument('--word-frequency', default='2', type=int,
                        help='only the word with higher frequency than this number will be added to vocabulary')
    parser.add_argument('--random-seed', default='20', type=int,
                        help='random seed')
    parser.add_argument('--run-name', default='', type=str, metavar='PATH',
                        help='Name of the run used in storing the results for post-precessing (default: none)')
    parser.add_argument('--word-noise', default='drop:1', type=str,
                        help='What type of noise should be added to the input (NLP) and how much; format= [(drop|replace):X], where replace=replace a random word with a wordnet synonym, drop=random word dropout, X=number of words (default: drop:1) ')
    parser.add_argument('--save-custom-embedding', default=True, type=str2bool, metavar='BOOL',
                        help='Save the custom embedding generated from the LSTM-based custom_embed model (default: True)')
    parser.add_argument('--max-entity-len', default='8', type=int,
                        help='maximum number of words in entity, extra words would be truncated')
    parser.add_argument('--max-inbetween-len', default='50', type=int,
                        help='maximum number of words in between of two entities, extra words would be truncated')
    parser.add_argument('--ckpt-file', type=str, default='best.ckpt', help='best checkpoint file')
    parser.add_argument('--ckpt-path', type=str, default='', help='path where best checkpoint file locates')
    parser.add_argument('--subset-labels', type=str, default='None', help='if not \'None\', only datpoints with the specified subset of test labels are considered, for both train/dev/test; currently only implemented for fullyLex and headLex of Riedel')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
