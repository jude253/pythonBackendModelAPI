import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Set the seed for PyTorch random number generator
torch.manual_seed(1)

# If gpu is supported, then seed the gpu random number generator as well
gpu_available = torch.cuda.is_available()
if gpu_available:
    torch.cuda.manual_seed(1)

import sys
import os

print('You\'re running python %s' % sys.version.split(' ')[0])
print("GPU is available:", gpu_available)


cwd = os.getcwd()
VOCABFILENAME = 'Gender_age_classifier/corpustestMF.csv'
MODELFILENAME = "Gender_age_classifier/model.pt"
vocab_size = 12820
embedding_size = 320
# label 0 == female
# label 1 == male
label_meaning = ['Female', 'Male']


# Create a Deep Averaging network model class
# embedding_size is the size of the word_embedding we are going to learn
class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_size=32):
        super().__init__()

        # Create a word-embedding of dimension embedding_size
        # self.embeds is now the matrix E, where each column corresponds to the embedding of a word
        self.embeds = torch.nn.Parameter(torch.randn(vocab_size, embedding_size))
        self.embeds.requires_grad_(True)
        # add a final linear layer that computes the 2d output from the averaged word embedding
        self.fc = nn.Linear(embedding_size, 2)

    def average(self, x):
        '''
        This function takes in multiple inputs, stored in one tensor x. Each input is a bag of word representation of reviews.
        For each review, it retrieves the word embedding of each word in the review and averages them (weighted by the corresponding
        entry in x).

        Input:
            x: nxd torch Tensor where each row corresponds to bag of word representation of a review

        Output:
            n x (embedding_size) torch Tensor for the averaged reivew
        '''

        # YOUR CODE HERE
        n, d = x.shape
        E = self.embeds
        sumInnerProducts = 0

        # this is the matrix Multiplication part
        sumInnerProducts = torch.matmul(x, E)

        # this is the sum part:
        sumXs = torch.sum(x, axis=1)

        oneOverSumXs = 1 / torch.sum(x, axis=1)
        # create a tensor of the same shape as sumInnerProducts because I cannot figure out how to do
        # piecewise division of sumInnerProducts/sumXs
        emb = torch.zeros(sumInnerProducts.shape)

        # this is the piecewise wise division of the two with a for loop:

        # expand oneOverSumXs to have the same shape as sumInnerProducts, so I can do piece wise multiplication:
        pieceWiseMultTensor = oneOverSumXs.unsqueeze(-1).expand_as(sumInnerProducts)

        # piece-wise multiplication-- this is more optimized than a for loop.
        emb = pieceWiseMultTensor * sumInnerProducts

        return emb

    def forward(self, x):
        '''
        This function takes in a bag of word representation of reviews. It calls the self.average to get the
        averaged review and pass it through the linear layer to produce the model's belief.

        Input:
            x: nxd torch Tensor where each row corresponds to bag of word representation of reviews

        Output:
            nx2 torch Tensor that corresponds to model belief of the input. For instance, output[0][0] is
            is the model belief that the 1st review is negative
        '''
        review_averaged = self.average(x)

        out = None
        # YOUR CODE HERE

        # run the averaged reviews through the layers in self.fc
        out = self.fc(review_averaged)
        return out


def generate_featurizer(vocabulary):
    return CountVectorizer(vocabulary=vocabulary)


def csv_to_corpus_dict(filename):
    '''input:
            filename: name of file to storing corpus

        output:
            corpus: dictionary with each word in blog as key and an index as a value
    '''
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        corpus = {}
        index = 0
        for line in file:
            corpus[line.rstrip()] = index
            index += 1
    return corpus


def generate_PyTorch_Dataset(string_entry, vocab, label=0):
    '''
    Input:
        string_entry: a string probably from user input
        vocab: the vocab/corpus from the training dataset
        label: the gender of the user inputing data if, they please
    Output:
        testset_input: the pytorch test that can be evaled by the model, the string entry is vectorized
    '''
    # creates bag of words featurizer
    bow_featurizer = generate_featurizer(vocab)

    # convert the entry to bow representation and torch Tensor
    X_test_input = torch.Tensor(bow_featurizer.transform([string_entry]).toarray())
    y_test_input = torch.LongTensor(np.array(label).flatten())
    # Note: for Y_test_input I am just putting 0 for default.  I am not sure how to create a pytorch set without
    # the label simply for evaluating probability yet.

    # Generate PyTorch Dataset (of one data point)
    testset_input = torch.utils.data.TensorDataset(X_test_input, y_test_input)

    return testset_input


# load the model from file
def load_model(MODELFILENAME, vocab_size, embedding_size=320):
    '''
    Input:
        MODELFILENAME: the filename/path of the model. this might be an enviroment variable in my web app
        vocab_size: the number of words from the vocab/copus from the training set
        embedding_size: the length of the word embeddings
    Output:
        model: the deep averaging network model that was saved in file MODELFILENAME
    '''
    #     print(MODELFILENAME)
    model = DAN(vocab_size, embedding_size)
    model.load_state_dict(torch.load(MODELFILENAME))
    return model


def prediction(model, testset_input, label_meaning):
    # evaluate the model:
    output = {}
    # def eval_model_with():
    # this is going to be for testing if the input entry works

    # index of my input in its list (of 1 element):
    target = 0

    # gets the post text and label from the test set:
    # post_target, label_target = test_entry, test_label

    # does something with "testset" and the target
    if gpu_available:
        bog_target = testset_input[target][0].unsqueeze(0).cuda()
    else:
        bog_target = testset_input[target][0].unsqueeze(0)

    # evaluates the model or puts the model into evluate mode:
    model.eval()

    # this is from the pytorch docs:
    '''
    Disabling gradient calculation is useful for inference, 
    when you are sure that you will not call Tensor.backward(). 
    It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    '''
    with torch.no_grad():

        # calculates the logits of bog_target, or basically I think does a foward pass and get the likelyhood of being
        # male or female as output
        logits_target = model(bog_target)

    # the prediction is whichever of the output logits are higher.  Index 0 = Female, Index 1 = Male
    pred = torch.argmax(logits_target, dim=1)

    # soft max of logit values to find the probability that this classification is correct
    probability = torch.exp(logits_target.squeeze()) / torch.sum(torch.exp(logits_target.squeeze()))

    prediction = label_meaning[pred.item()]
    certainty = 100.0 * probability[pred.item()]
    #     print('Post: ', post_target)
    #     print('Ground Truth: ', label_meaning[int(label_target)])
    print('Prediction: %s (Certainty %2.2f%%)' % (prediction, certainty))

    return prediction, probability[0].item(), probability[1].item()


def gender_text_classifier(input_string):
    global VOCABFILENAME, MODELFILENAME, vocab_size, label_meaning, embedding_size

    vocab = csv_to_corpus_dict(VOCABFILENAME)
    testset_input = generate_PyTorch_Dataset(input_string, vocab)
    model = load_model(MODELFILENAME, vocab_size, embedding_size)
    pred, certainty_female, certainty_male = prediction(model, testset_input, label_meaning)
    certainty_female = str(round(100*certainty_female,2)) + '%'
    certainty_male = str(round(100*certainty_male, 2)) + '%'
    dict_resp = {'prediction': pred, 'certainty': {'female': certainty_female, 'male': certainty_male}}
    return dict_resp
