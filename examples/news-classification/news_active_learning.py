from utils import *


def main():
    # # TODO: put data to public drive folder
    datadir = '/Users/omerkirnap/Documents/Developer/pytorch_active_learning/'
    reader = DataReader()

    training_related_data = datadir + 'training_data/related.csv'
    training_not_related_data = datadir + 'training_data/not_related.csv'

    training_data = reader.load_data(training_related_data) + reader.load_data(training_not_related_data)
    training_count = len(training_data)

    validation_related_data = datadir + 'validation_data/related.csv'
    validation_not_related_data = datadir + 'validation_data/not_related.csv'
    validation_data = reader.load_data(validation_related_data) + reader.load_data(validation_not_related_data)
    validation_count = len(validation_data)

    evaluation_related_data = datadir + 'evaluation_data/related.csv'
    evaluation_not_related_data = datadir + 'evaluation_data/not_related.csv'
    evaluation_data = reader.load_data(evaluation_related_data) + reader.load_data(evaluation_not_related_data)
    evaluation_count = len(evaluation_data)

    unlabeled_datadir = datadir + 'unlabeled_data/unlabeled_data.csv'
    unlabeled_data = reader.load_data(unlabeled_datadir, skip_labeled=True)

    feature_extractor = FeatureExtractor()
    vocab_size = feature_extractor.create_features(unlabeled_data, training_data)


if __name__ == '__main__':
    main()
