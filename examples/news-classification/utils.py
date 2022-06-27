#  Implements data-reading and modelling utils etc.

import csv
import math
import re
import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from random import shuffle



class DataReader(object):
    def __init__(self):
        self.already_labeled = {}
        self.feature_index = {}

    def load_data(self, path_to_file, skip_labeled=False):
        """
        Reads the csv file in the format of [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        """
        data = []
        with open(path_to_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if skip_labeled and row[0] in self.already_labeled:
                    continue
                if len(row) < 3:  # if no-label, append empty label
                    row.append("")
                if len(row) < 4:  # if no-sampling strategy, append empty sampling strategy
                    row.append("")
                if len(row) < 5:  # if no confidence yet, append 0 to modify later
                    row.append(0)
                data.append(row)

                label = str(row[2])
                if row[2] != "":
                    textid = row[0]
                    self.already_labeled[textid] = label
        return data

    @classmethod
    def append_data(cls, path_to_file, data):
        with open(path_to_file, 'a', errors='replace') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    @classmethod
    def write_data(cls, path_to_file, data):
        with open(path_to_file, 'w', errors='replace') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)


class FeatureExtractor(object):
    """
    Create features
    """

    def __init__(self,
                 feature_index=None,
                 minword=3,
                 ):
        if feature_index is None:
            feature_index = {}
        self.minword = minword
        self.feature_index = feature_index

    def create_features(self,
                        data_to_label,
                        training_data,
                        ):
        total_training_words = {}
        for item in data_to_label + training_data:
            text = item[1]
            for word in text.split():
                total_training_words[word] = total_training_words.get(word, 0) + 1

        self._modify_feature_index(data_to_label + training_data, total_training_words)
        return len(self.feature_index)

    def _modify_feature_index(self,
                              data,
                              total_training_words
                              ):
        for item in data:
            text = item[1]
            for word in text.split():
                # import ipdb
                # ipdb.set_trace()
                if word not in self.feature_index and total_training_words[word] >= self.minword:
                    self.feature_index[word] = len(self.feature_index)

    def get_feature_vector(self, features):
        vec = torch.zeros(len(self.feature_index))
        for feat in features:
            if feat in self.feature_index:
                vec[self.feature_index[feat]] += 1
        return vec.view(1, -1)


class SimpleTextClassifier(nn.Module):
    """
    1 hidden layer MLP classifier
    """
    hidden_size = 128

    def __init__(self, num_labels, vocab_size) -> None:
        super(SimpleTextClassifier, self).__init__()
        self.linear1 = nn.Linear(vocab_size, SimpleTextClassifier.hidden_size)
        self.linear2 = nn.Linear(SimpleTextClassifier.hidden_size, num_labels)

    def forward(self, feature_vec):
        h1 = self.linear1(feature_vec)
        h1 = F.relu(h1)
        output = self.linear2(h1)
        return F.log_softmax(output, dim=1)


def train_model(training_data,
                feature_index,
                validation_data='',
                evaluation_data='',
                num_labels=2,
                vocab_size=0,
                ):
    model = SimpleTextClassifier(num_labels, vocab_size)
    feature_extractor = FeatureExtractor(feature_index=feature_index)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1
    select_per_epoch = 200  # number to select per epoch per label

    for epoch in range(epochs):
        print('Epoch: ' + str(epoch))

        shuffle(training_data)
        related = [row for row in training_data if '1' == row[2]]
        not_related = [row for row in training_data if '0' == row[2]]

        epoch_data = related[:select_per_epoch]
        epoch_data += not_related[:select_per_epoch]
        shuffle(epoch_data)

        for item in epoch_data:
            features = item[1].split()
            label = int(item[2])
            model.zero_grad()

            feature_vec = feature_extractor.get_feature_vector(features)
            target = torch.LongTensor([int(label)])
            log_probs = model(feature_vec)

            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

    fscore, auc = evaluate_model(model, evaluation_data, feature_extractor)
    fscore = round(fscore, 3)
    auc = round(auc, 3)
    print(fscore, auc)

    # save model to path that is alphanumeric and includes number of items and accuracies in filename
    timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":",
                                                                                                                 "")
    training_size = "_" + str(len(training_data))
    accuracies = str(fscore) + "_" + str(auc)

    model_path = 'models/' + timestamp + accuracies + training_size + '.params'
    # torch.save(model.state_dict(), model_path)
    # return model_path  # TODO: save model and return model_path?
    return model


def evaluate_model(model, evaluation_data, feature_extractor):
    """
    Compute f-score for disaster-related and the AUC
    """
    related_confs = []  # related items and their confidence of being related
    not_related_confs = [] # related items and their confidence of being related

    true_pos, false_pos, false_neg = 0.0, 0.0, 0.0  # confussion matrix items

    with torch.no_grad():
        for item in evaluation_data:
            _, text, label, _, _, = item
            feature_vec = feature_extractor.get_feature_vector(text.split())
            log_probs = model(feature_vec)

            prob_related = math.exp(log_probs.data.tolist()[0][1])

            if label == '1':
                related_confs.append(prob_related)
                if prob_related > 0.5:
                    true_pos += 1.0
                else:
                    false_neg += 1.0
            else:
                not_related_confs.append(prob_related)
                if prob_related > 0.5:
                    false_pos += 1.0
    # compute F-Score
    if true_pos == 0.0:
        fscore = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = (2*precision*recall) / (precision+recall)
    # get auc
    not_related_confs.sort()
    total_greater = 0
    for conf in related_confs:
        for conf2 in not_related_confs:
            if conf < conf2:
                break
            else:
                total_greater += 1
    denom = len(not_related_confs) * len(related_confs)
    auc = total_greater / denom
    return (fscore, auc)