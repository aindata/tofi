#  Implements data-reading and modelling utils etc.

import csv


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





