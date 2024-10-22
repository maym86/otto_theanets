import climate
import theanets
from sklearn import cross_validation
from sklearn.utils import compute_class_weight
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import preprocessing
from ml_metrics import *
import math
# try scaling http://sebastianraschka.com/Articles/2014_about_feature_scaling.html


def load_test_data(std_scale):
    raw_testing_data = pd.read_csv('test.csv')
    raw_testing_data = raw_testing_data.astype('float32')  # convert types to float32

    # Get the features and the classes
    test_features = np.log(raw_testing_data.iloc[:, 1:94] + 1).values#apply log functio

    test_features = std_scale.transform(test_features)  # scale the features
    return test_features


def save_results(file, test_features, results):
    # create colums and headers
    index = range(1, len(test_features) + 1)
    columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']

    # create data frame
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

    raw_training_data = raw_training_data.iloc[np.random.permutation(len(raw_training_data))] #shuffle data
    # Get the features and the classes
    features = np.log(raw_training_data.iloc[:, 1:94] + 1).values # apply log function

    classes = raw_training_data['target'].values

    print np.unique(classes)

    #split train/validate
    feat_train, feat_test, class_train, class_test = cross_validation.train_test_split(features, classes,
                                                                                       test_size=0.3,
                                                                                       random_state=1232)

    feat_train, feat_val, class_train, class_val = cross_validation.train_test_split(feat_train, class_train,
                                                                                     test_size=0.3,
                                                                                     random_state=1232)


    #scale the features
    std_scale = preprocessing.StandardScaler().fit(feat_train)
    feat_train = std_scale.transform(feat_train)
    feat_val = std_scale.transform(feat_val)
    feat_test = std_scale.transform(feat_test)

    #class weights
    weights = compute_class_weight('auto', np.unique(classes), class_train)
    weights = weights.astype('float32')
    print weights
    train_weights = []
    val_weights = []
    for i in class_train:
        train_weights.append(weights[i])

    for i in list(class_val):
        val_weights.append(weights[i])

    #convert to np array for theanets
    training_data = [feat_train, class_train, np.array(train_weights)]
    validation_data = [feat_val, class_val, np.array(val_weights)]
    test_data = [feat_test, class_test]

    return training_data, validation_data, test_data, std_scale


def main():
    training_data, validation_data, test_data, std_scale = load_training_data()
    climate.enable_default_logging()

    targets = ['esgd','layerwise','rmsprop','nag','rprop','sgd','sample','adadelta']
    layers = [(93,  dict(size=512, sparsity=0.2, activation='relu'),
                    dict(size=512, sparsity=0.2, activation='relu'),
                    dict(size=512, sparsity=0.2, activation='relu'),
                    9)]

    for l in layers:
        for t in targets:
            exp = theanets.Experiment(
                theanets.Classifier,
                layers=l,
                weighted=True,
                output_activation='softmax'
            )

            exp.train(training_data,
                        validation_data,
                        optimize=t,
                      )

            exp.train(training_data,
                        validation_data,
                        optimize=t,

                      )
            exp.train(training_data,
                        validation_data,
                        optimize=t,

                      )

            #get an prediction of the accuracy from the test_data
            test_results = exp.network.predict(test_data[0])
            loss = multiclass_log_loss(test_data[1], test_results)

            print 'Test multiclass log loss:', loss

            out_file = 'results/' + str(loss) + t
            exp.save(out_file + '.pkl')


            #save the kaggle results
            kaggle_test_features = load_test_data(std_scale)
            results = exp.network.predict(kaggle_test_features)
            save_results(out_file + '.csv', kaggle_test_features, results)


if __name__ == "__main__":
    main()