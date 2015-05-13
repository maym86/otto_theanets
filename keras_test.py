__author__ = 'gleesonm'


import climate

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

from sklearn import cross_validation
import pandas as pd
from sklearn import preprocessing
from ml_metrics import *

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


def class_list_to_matrix(classes):
    matrix = np.zeros([classes.shape[0], len(set(classes))])
    rows = matrix.shape[0]
    matrix[np.arange(rows), classes.astype(int)] = 1
    return matrix


def load_training_data():
    raw_training_data = pd.read_csv('train.csv')

    # convert types to ints
    raw_training_data['target'] = raw_training_data['target'].apply(class_to_int)
    raw_training_data = raw_training_data.astype('float32')
    raw_training_data['target'] = raw_training_data['target'].astype('int32')

    # Get the features and the classes
    features = np.log(raw_training_data.iloc[:, 1:94] + 1).values # apply log function
    classes = raw_training_data['target'].values

    print np.unique(classes)
    #split train/validate
    feat_train, feat_test, class_train, class_test = cross_validation.train_test_split(features,
                                                                                       class_list_to_matrix(classes),
                                                                                       test_size=0.2,
                                                                                       random_state=1232)

    feat_train, feat_val, class_train, class_val = cross_validation.train_test_split(feat_train,
                                                                                       class_train,
                                                                                       test_size=0.2,
                                                                                       random_state=1232)

    #scale the features
    std_scale = preprocessing.StandardScaler().fit(feat_train)
    feat_train = std_scale.transform(feat_train)
    feat_val = std_scale.transform(feat_val)
    feat_test = std_scale.transform(feat_test)

    #convert to np array for theanets
    training_data = [feat_train, class_train]
    test_data = [feat_test, class_test]
    validation_data = [feat_val, class_val]

    return training_data, validation_data ,test_data, std_scale

#use a min improvement method of training
def trainer(model, training_data, validation_data, batch_size, min_improvement=0.01, patience=4, epochs_before_eval=10):
    loss = 1000000
    count = 0
    best_model = []
    while True:
        model.fit(training_data[0], training_data[1],  nb_epoch=epochs_before_eval,
                  batch_size=batch_size, shuffle=True, show_accuracy=True)

        new_loss = model.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)

        if new_loss < (loss - min_improvement):
            loss = new_loss
            best_model = model
            count = 0
        else:
            count += 1

        print 'Validation loss:', new_loss, 'Current Best:', loss, 'Count:', count

        if count == patience:
            break

    return best_model


def main():
    training_data, validation_data, test_data, std_scale = load_training_data()

    layers = [(512,512,512), (512,256,128), (256,256,256)]
    optimizers = ['sgd', 'adam', 'adagrad' ,'adadelta' ,'rmsprop']
    for l in layers:
        for o in optimizers:
            model = Sequential()
            model.add(Dense(93, l[0], init='glorot_uniform'))
            model.add(PReLU((l[0],)))
            model.add(BatchNormalization((l[0],)))
            model.add(Dropout(0.5))

            model.add(Dense(l[0], l[1], init='glorot_uniform'))
            model.add(PReLU((l[1],)))
            model.add(BatchNormalization((l[1],)))
            model.add(Dropout(0.5))

            model.add(Dense(l[1], l[2], init='glorot_uniform'))
            model.add(PReLU((l[2],)))
            model.add(BatchNormalization((l[2],)))
            model.add(Dropout(0.5))

            model.add(Dense(l[2], 9, init='glorot_uniform'))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=o)

            batch_size=128
            model = trainer(model, training_data, validation_data, batch_size)

            loss = model.evaluate(test_data[0], test_data[1], batch_size=batch_size)
            print 'Test multiclass log loss:', loss

            out_file = 'keras/' + str(loss) + str(o) + str(l)
            model.save_weights(out_file + '.hdf5')
            #save the kaggle results
            kaggle_test_features = load_test_data(std_scale)
            results = model.predict(kaggle_test_features, batch_size=batch_size)
            save_results(out_file + '.csv', kaggle_test_features, results)


if __name__ == "__main__":
    main()