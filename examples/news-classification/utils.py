#  Implements data-reading and modelling utils etc.

import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

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
