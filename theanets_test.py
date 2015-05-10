import climate
import theanets
from sklearn import cross_validation
import pandas as pd
import numpy as np
from sklearn import preprocessing
from ml_metrics import *
# try scaling http://sebastianraschka.com/Articles/2014_about_feature_scaling.html


def load_test_data(std_scale):
    raw_testing_data = pd.read_csv('test.csv')
    raw_testing_data = raw_testing_data.astype('float32')  # convert types to float32

    # Get the features and the classes
    test_features = raw_testing_data.iloc[:, 1:94].values  #get the feature vectors

    test_features = std_scale.transform(test_features)  #scale the features
    return test_features


def save_results(file, test_features, results):
    # create colums and headers
    index = range(1, len(test_features) + 1)
    columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']

    #create data frame
    res = pd.DataFrame.from_records(results, index=index, columns=columns)
    res.index.name = 'id'

    #save results
    res.to_csv(file)


def class_to_int(class_string):
    return int(class_string[6]) - 1


def load_training_data():
    raw_training_data = pd.read_csv('train.csv')

    # convert types to ints
    raw_training_data['target'] = raw_training_data['target'].apply(class_to_int)
    raw_training_data = raw_training_data.astype('float32')
    raw_training_data['target'] = raw_training_data['target'].astype('int32')

    #Get the features and the classes
    features = raw_training_data.iloc[:, 1:94].values
    classes = raw_training_data['target'].values

    print np.unique(classes)

    #split train/validate
    feat_train, feat_test, class_train, class_test = cross_validation.train_test_split(features, classes, test_size=0.3,
                                                                                       random_state=0)

    feat_train, feat_val, class_train, class_val = cross_validation.train_test_split(feat_train, class_train,
                                                                                     test_size=0.3,
                                                                                     random_state=0)

    #scale the features
    std_scale = preprocessing.StandardScaler().fit(feat_train)
    feat_train = std_scale.transform(feat_train)
    feat_val = std_scale.transform(feat_val)
    feat_test = std_scale.transform(feat_test)

    #convert to np array for theanets
    training_data = [feat_train, class_train]
    validation_data = [feat_val, class_val]
    test_data = [feat_test, class_test]

    return training_data, validation_data, test_data, std_scale


def main():
    training_data, validation_data, test_data, std_scale = load_training_data()
    climate.enable_default_logging()

    trainers = ['nag', 'sgd', 'rprop', 'rmsprop', 'adadelta', 'esgd', 'hf', 'sample', 'layerwise', 'pretrain']
    # layers = [(93, 256, 128, 9), (93, 128, 64, 32, 9)]
    layers = [(93, 256, 128, 9)]
    for l in layers:
        exp = theanets.Experiment(
            theanets.Classifier,
            layers=l
        )

        #loop through trainers
        for t in trainers:

            train_result = exp.train(training_data,
                                     validation_data,
                                     algorithm=t,
                                     hidden_l1=0.001,
                                     train_batches=300,
                                     )


            #get an prediction of the accuracy from the test_data
            test_results = exp.network.predict(test_data[0])
            loss = multiclass_log_loss(test_data[1], test_results)

            print 'Test multiclass log loss:', loss

            out_file = 'results/' + str(loss) + t + str(l)
            exp.save(out_file + '.pkl')


            #save the kaggle results
            kaggle_test_features = load_test_data(std_scale)
            results = exp.network.predict(kaggle_test_features)
            save_results(out_file + '.csv', kaggle_test_features, results)


if __name__ == "__main__":
    main()