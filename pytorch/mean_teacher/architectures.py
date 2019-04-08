import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import export, parameter_count
###############
# https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits

###############
#### TODO: Add the emboot objective functions as another parameter -- NO longer necessary at least for the COLING submission
###############
# @export
# def pushpull_embed(pretrained=True, **kwargs):
#
#
# Look at pytorch implementations like https://github.com/fanglanting/skip-gram-pytorch/blob/master/model.py
# class PushPullSkipGram(nn.Module):
#
#     def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz):
#         super().__init__()


##############################################
##### More advanced architecture where the entity and pattern embeddings are computed by a Sequence model (like a biLSTM) and then concatenated together
##############################################
@export
def custom_embed(pretrained=True, word_vocab_size=7970, wordemb_size=300, hidden_size=300, num_classes=4, word_vocab_embed=None, update_pretrained_wordemb=False):

    lstm_hidden_size = 100
    model = SeqModelCustomEmbed(word_vocab_size, wordemb_size, lstm_hidden_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model


# todo: Is padding the way done here ok ?
class SeqModelCustomEmbed(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_size, lstm_hidden_size, hidden_size, output_size, word_vocab_embed, update_pretrained_wordemb): #todo: add lstm parameters
        super().__init__()
        self.embedding_size = word_embedding_size
        self.entity_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)
        self.pat_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)

        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.entity_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            self.pat_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:  # NOTE: do not update the embeddings
                print("NOT UPDATING the word embeddings ....")
                self.entity_word_embeddings.weight.detach_()
                self.pat_word_embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")

        # todo: keeping the hidden sizes of the LSTMs of entities and patterns to be same. To change later ?
        self.lstm_entities = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)
        self.lstm_patterns = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)

        # UPDATE: NOT NECASSARY .. we can directly return from forward method the values that we want,
        #  in this case `entity_lstm_out` and `pattern_lstm_out`
        # Note: saving these representations, so that when they are computed during forward, we can use these variables
        # to dump the custom entity and pattern embeddings
        # self.entity_lstm_out = None
        # self.pattern_lstm_out = None

        # Note: Size of linear layer = [(lstm_hidden_size * 2) bi-LSTM ] * 2 --> concat of entity and context lstm out
        self.layer1 = nn.Linear(lstm_hidden_size * 2 * 2, hidden_size, bias=True)  # concatenate entity and pattern embeddings and followed by a linear layer;
        self.activation = nn.ReLU()  # non-linear activation
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True) # second linear layer from hidden layer to the output logits

    # todo: Is padding the way done here ok ? should I explicitly tell what the pad value is ?
    def forward(self, entity, pattern):
        entity_word_embed = self.entity_word_embeddings(entity).permute(1, 0, 2)  # compute the embeddings of the words in the entity (Note the permute step)
        pattern_split = torch.unbind(pattern, dim=1)  # split the 2D list of list of set of patterns into individual patterns of words
        pattern_word_embed_list = [self.pat_word_embeddings(p).permute(1, 0, 2)  # Note: the permute step is to make it compatible to be input to LSTM (seq of words,  batch, dimensions of each word)
                                  for p in pattern_split] # list of pattern word embeddings (one for each pattern of words)

        ###############################################
        # bi-LSTM computation here

        # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        self.lstm_entities.flatten_parameters()
        self.lstm_patterns.flatten_parameters()

        _, (entity_lstm_out, _) = self.lstm_entities(entity_word_embed)  # bi-LSTM over entities, hidden state is initialized to 0 if not provided
        entity_lstm_out = torch.cat([entity_lstm_out[0], entity_lstm_out[1]], 1) # roll out the 2 tuple output each of the LSTMs

        pattern_lstm_out_list = list()
        for pattern_embed in pattern_word_embed_list:
            _, (pattern_lstm_out, _) = self.lstm_patterns(pattern_embed)  # hidden state is initialized to 0 if not provided
            pattern_lstm_out_list.append(torch.unsqueeze(torch.cat([pattern_lstm_out[0], pattern_lstm_out[1]], 1), 0)) # roll out the 2 tuple output each of the LSTMs and add to the list
        pattern_lstm_out = torch.cat(pattern_lstm_out_list, 0)

        # compute the average of all the pattern lstms outputs
        pattern_lstm_out_avg = torch.mean(pattern_lstm_out, 0)

        # concatenate the entity_lstm and avgeraged pattern_lstm representations
        entity_and_pattern_lstm_out = torch.cat([entity_lstm_out, pattern_lstm_out_avg], dim=1)

        ###############################################
        # print("###############################################")
        # print("entity_word_embed = " + str(entity_word_embed.size()))
        # print("pattern_word_embed_list = " + str(len(pattern_word_embed_list)))
        # print("pattern_word_embed_list[i] = " + str(pattern_word_embed_list[0].size()))
        # print("entity_lstm_out = " + str(entity_lstm_out.size()))
        # print("pattern_lstm_out = " + str(pattern_lstm_out.size()))
        # print("pattern_lstm_out_avg = " + str(pattern_lstm_out_avg.size()))
        # print("entity_and_pattern_lstm_out = " + str(entity_and_pattern_lstm_out.size()))
        # print("###############################################")

        ###############################################

        res = self.layer1(entity_and_pattern_lstm_out)
        res = self.activation(res)
        res = self.layer2(res)
        return res, entity_lstm_out, pattern_lstm_out

##############################################
##### Simple architecture where the entity and pattern embeddings are computed by an average
##############################################
@export
def simple_MLP_embed(pretrained=True, num_classes=4, word_vocab_embed=None, word_vocab_size=7970, wordemb_size=300, hidden_size=50, update_pretrained_wordemb=False):

    # Note: custom embeddings sz in Emboot was 15 (used in conjunction of gigaword init embeddings as features in the classifier). This is similar to ladder networks

    model = FeedForwardMLPEmbed(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model


class FeedForwardMLPEmbed(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed, update_pretrained_wordemb):
        super().__init__()
        self.embedding_size = embedding_size
        self.entity_embeddings = nn.Embedding(word_vocab_size, embedding_size)
        self.pat_embeddings = nn.Embedding(word_vocab_size, embedding_size)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
        if word_vocab_embed is not None: # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.entity_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            self.pat_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:
                # NOTE: do not update the emebddings
                # https://discuss.pytorch.org/t/how-to-exclude-embedding-layer-from-model-parameters/1283
                print ("NOT UPDATING the word embeddings ....")
                self.entity_embeddings.weight.detach_()
                self.pat_embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")

        ## Intialize the embeddings if pre-init enabled ? -- or in the fwd pass ?
        ## create : layer1 + ReLU
        self.layer1 = nn.Linear(embedding_size*2, hidden_sz, bias=True) ## concatenate entity and pattern embeddings
        self.activation = nn.ReLU()
        ## create : layer2 + Softmax: Create softmax here
        self.layer2 = nn.Linear(hidden_sz, output_sz, bias=True)
        # self.softmax = nn.Softmax(dim=1) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function

    def forward(self, entity, pattern):
        entity_embed = torch.mean(self.entity_embeddings(entity), 1)             # Note: Average the word-embeddings
        pattern_flattened = pattern.view(pattern.size()[0], -1)                  # Note: Flatten the list of list of words into a list of words
        pattern_embed = torch.mean((self.pat_embeddings(pattern_flattened)), 1)  # Note: Average the words in every pattern in the list of patterns
        # print (entity_embed.size())
        # print (pattern_embed.size())
        concatenated = torch.cat([entity_embed, pattern_embed], 1)
        res = self.layer1(concatenated)
        res = self.activation(res)
        res = self.layer2(res)
        # print (res)
        # print (res.shape)
        # res = self.softmax(res) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function
        # print ("After softmax : " + str(res))
        return res

##############################################
##### Simple architecture for relation extraction. the entity and sentence embeddings are computed by an average
##############################################
@export
def simple_MLP_embed_RE(word_vocab_size, num_classes, wordemb_size, pretrained=True, word_vocab_embed=None, hidden_size=200, update_pretrained_wordemb=False):

    model = FeedForwardMLPEmbed_RE(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model


class FeedForwardMLPEmbed_RE(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed, update_pretrained_wordemb):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(word_vocab_size, embedding_size)
        print("word_vocab_size=", word_vocab_size)

        if word_vocab_embed is not None: # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:
                # NOTE: do not update the emebddings
                # https://discuss.pytorch.org/t/how-to-exclude-embedding-layer-from-model-parameters/1283
                print ("NOT UPDATING the word embeddings ....")
                self.embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")

        ## Intialize the embeddings if pre-init enabled ? -- or in the fwd pass ?
        ## create : layer1 + ReLU
        self.layer1 = nn.Linear(embedding_size, hidden_sz, bias=True) ## concatenate entity and pattern embeddings
        self.activation = nn.ReLU()
        ## create : layer2 + Softmax: Create softmax here
        self.layer2 = nn.Linear(hidden_sz, output_sz, bias=True)
        # self.softmax = nn.Softmax(dim=1) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function

    def forward(self, input_tuple):
        input = input_tuple[0]
        seq_lengths = input_tuple[1]   # LongTensor, length before padding
        pad_id = input_tuple[2]    # the id corresponds to mask

        # Embed the input
        embedded = self.embeddings(input)   # embedded.shape: torch.Size([256, 66, 100])

        # Make the mask for removing the padded items
        mask = input.ne(pad_id)

        if torch.cuda.is_available():
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        # add an extra dimension, initially of size 1
        # then "expand_as" copies the last dimension into the new dimension
        # This essentially propogates the mask through the final dimension
        # input.shape[0] should be batch size and input.shape[1] should be the num_words
        expanded_mask = mask.view(input.shape[0], input.shape[1], 1).expand_as(embedded)

        # Apply mask (clear out the embeddings of padded items)
        masked_embedded = embedded * expanded_mask

        summation = masked_embedded.sum(1)  # Variable containing torch.FloatTensor of size 256x100

        if torch.cuda.is_available():
            seq_lengths = torch.autograd.Variable(seq_lengths.type(torch.cuda.FloatTensor))
        else:
            seq_lengths = torch.autograd.Variable(seq_lengths.type(torch.FloatTensor))  # Variable containing torch.FloatTensor of size 256x100

        seq_lengths = seq_lengths.view(-1, 1).expand_as(summation)

        avg = summation / seq_lengths

        res = self.layer1(avg)
        res = self.activation(res)
        res = self.layer2(res)

        return res

##############################################
##### More advanced architecture where the entity and words in-between embeddings are computed by a Sequence model (like a biLSTM) and then concatenated together
##############################################
@export
def lstm_RE(word_vocab_size, num_classes, wordemb_size, hidden_size, pretrained=True, word_vocab_embed=None, update_pretrained_wordemb=False):
    lstm_hidden_size = 50 # was 100
    model = SeqModel_RE(word_vocab_size, wordemb_size, lstm_hidden_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model

class SeqModel_RE(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_size, lstm_hidden_size, hidden_size, output_size, word_vocab_embed, update_pretrained_wordemb): #todo: add lstm parameters
        super().__init__()
        self.embedding_size = word_embedding_size
        self.embeddings = nn.Embedding(word_vocab_size, word_embedding_size)
        print("word_vocab_size=", word_vocab_size)

        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:  # NOTE: do not update the embeddings
                print("NOT UPDATING the word embeddings ....")
                self.embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")

        self.lstm = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)

        # *2 is for bidir
        self.layer1 = nn.Linear(lstm_hidden_size * 2, hidden_size, bias=True)  # concatenate entity and pattern embeddings and followed by a linear layer;
        self.activation = nn.ReLU()  # non-linear activation
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)

    # todo: Is padding the way done here ok ? should I explicitly tell what the pad value is ?
    def forward(self, input_tuple):
        input = input_tuple[0]
        seq_lengths = input_tuple[1]

        # https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
        # seq_lengths = torch.LongTensor([x for x in lengths])
        # #print("shape of seq_lengths:", seq_lengths.shape)

        sorted_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        input_sorted = input[perm_idx]

        embed = self.embeddings(input_sorted).permute(1, 0, 2)  # compute the embeddings of the words in the entity (Note the permute step)
        #print("shape of embed:", embed.shape)

        packed = pack_padded_sequence(embed, sorted_lengths.cpu().numpy(), batch_first=False)

        # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        self.lstm.flatten_parameters()

        _, (lstm_out, _) = self.lstm(packed)  # bi-LSTM over entities, hidden state is initialized to 0 if not provided
        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], 1) # roll out the 2 tuple output each of the LSTMs

        res = self.layer1(lstm_out)
        res = self.activation(res)
        res = self.layer2(res)
        return res, perm_idx


@export
def simple_MLP(pretrained=True, num_classes=10):

    ## Hard-coding the parameters
    input_sz = 900
    hidden_sz = 400
    output_sz = num_classes
    model = FeedForwardMLP(input_sz, hidden_sz, output_sz)

    return model

class FeedForwardMLP(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        ## Write code to initialize the MLP module
        super().__init__()
        self.layer1 = nn.Linear(input_sz, hidden_sz, bias=True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sz, output_sz, bias=True)
        # self.softmax = nn.Softmax() ## IMPT NOTE: Removing the softmax from here as it is done in the loss function

    def forward(self, x):
        ## code to to the forward pass of the MLP module
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        # x = self.softmax(x) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function
        return x

@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model


@export
def resnext152(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet224x224(BottleneckBlock,
                          layers=[3, 8, 36, 3],
                          channels=32 * 4,
                          groups=32,
                          downsample='basic', **kwargs)
    return model



class ResNet224x224(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 4
        self.downsample_mode = downsample
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, channels * 8, groups, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(
            planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(residual + a)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1,
                                                              grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
