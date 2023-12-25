import argparse
import csv
import gc
from itertools import product

import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from sklearn import metrics
import tensorflow as tf
import os

from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dropout

from utils import parse_data, OptimizerFactory
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from utils import set_seed

# Set GPU device and disable eager execution
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# physical_devices = tf.config.list_physical_devices('GPU')

config = dict()

XGlobal = list()
YGlobal = list()

XValGlobal = list()
YValGlobal = list()

XTestGlobal = list()
YTestGlobal = list()

SavedParameters = list()
SavedParametersAE = list()
Mode = str()
Name = str()
SAVE_PATH_ = str()
result_path = str()
CHECKPOINT_PATH_ = str()

tid = 0
best_loss = float('inf')
best_val_acc = 0
best_ae = None
best_params = dict()
load_previous_result = True
continue_loading = True

set_seed(seed=0)


def train_ae(params, dataset_name):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global best_loss
    global best_params
    global best_ae
    global SavedParametersAE

    ae_time = 0

    hidden_size = params['hidden_size']
    opt = params['optimizer']
    activation = 'tanh'
    loss_fn = 'mse'
    ae_epoch = 150
    ae_filename = f'{dataset_name}_H-{hidden_size}_O-{opt}_A-{activation}_L-{loss_fn}_E{ae_epoch}'
    BASE_DIR = Path(__file__).resolve().parent
    ae_path = os.path.join(BASE_DIR, 'trained_ae', f'{ae_filename}.keras')

    X_train = np.array(XGlobal)
    X_test = np.array(XTestGlobal)
    X_val = np.array(XValGlobal)

    if os.path.isfile(ae_path):
        ae = tf.keras.models.load_model(ae_path)
        print(f'AE loaded from {ae_path}')
        tr_loss = ae.evaluate(X_train, X_train)
        val_loss = ae.evaluate(X_val, X_val)
        print(f'loss: {tr_loss}, val_loss: {val_loss}')

    else:
        ae = Sequential()
        ae.add(LSTM(hidden_size,
                    input_shape=(1, X_train.shape[2]),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    return_sequences=True))
        ae.add(LSTM(X_train.shape[2],
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    return_sequences=True))

        opt_factory_ae = OptimizerFactory(opt=opt,
                                          lr_schedule=True,
                                          len_dataset=len(X_train),
                                          epochs=150,
                                          batch_size=32,
                                          init_lr=0.1,
                                          final_lr=0.001)

        ae.compile(optimizer=opt_factory_ae.get_opt(), loss='mse')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH_, 'best_ae.h5'),
            monitor='val_loss',
            mode='auto',
            save_best_only=True)
        ae_start = time.time()
        history = ae.fit(X_train, X_train,
                         validation_data=(X_val, X_val),
                         epochs=150,
                         batch_size=32,
                         callbacks=[early_stop, model_checkpoint],
                         verbose=2)
        ae_time = (time.time() - ae_start)

        stopped_epoch = early_stop.stopped_epoch

        tr_loss = history.history['loss'][stopped_epoch]
        val_loss = history.history['val_loss'][stopped_epoch]
        ae.save(ae_path)

    res = {"loss": tr_loss,
           "val_loss": val_loss,
           "hidden_size": params['hidden_size'],
           "optimizer": params['optimizer'],
           "train_time": ae_time
           }

    SavedParametersAE.append(res)
    # Save model
    if SavedParametersAE[-1]["val_loss"] < best_loss:
        print("new saved model:" + str(SavedParametersAE[-1]))
        best_ae = ae
        best_params = res
        ae.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "ae_model.h5")))
        del ae
        best_loss = SavedParametersAE[-1]["val_loss"]

    SavedParametersAE = sorted(SavedParametersAE, key=lambda i: i['val_loss'])

    try:
        with open((os.path.join(SAVE_PATH_, 'best_result_ae.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParametersAE[0].keys())
            writer.writeheader()
            writer.writerows(SavedParametersAE)
    except IOError:
        print("I/O error")
    gc.collect()


def train_cf(params):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global tid
    global best_ae

    global best_val_acc
    global SavedParameters

    global result_path
    global load_previous_result
    global continue_loading

    if (result_path is not None) and continue_loading:
        result_table = pd.read_csv(result_path)

        tid += 1
        selected_row = result_table[round(result_table['learning_rate'], 5) == round(params['learning_rate'], 5)]
        print(selected_row)
        loss_hp = selected_row['F1_val'].values[0]
        loss_hp = -loss_hp
        if tid == len(result_table):
            continue_loading = False

        if load_previous_result:
            best_val_acc = result_table['F1_val'].max()

            result_table = result_table.sort_values('F1_val', ascending=False)
            SavedParameters = result_table.to_dict(orient='records')
            with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)

            load_previous_result = False

    else:
        tid += 1
        tf.keras.backend.clear_session()
        model_input = Input(shape=(1, XGlobal.shape[2]))
        y = LSTM(units=params['unit1'],
                 activation='tanh',
                 kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 return_sequences=True)(model_input)
        y = Dropout(params['dropout'])(y)
        y = LSTM(units=params['unit2'],
                 activation='tanh',
                 kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                 bias_initializer=tf.keras.initializers.Zeros())(y)
        output = Dense(YGlobal.shape[1],
                       kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0),
                       bias_initializer=tf.keras.initializers.Zeros(),
                       activation='softmax')(y)
        model = tf.keras.Model(inputs=model_input, outputs=output)

        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_variables])
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH_, 'best_model.h5'),
            save_weights_only=True,
            monitor='val_loss',
            mode='auto',
            save_best_only=True
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True
        )
        cf_start = time.time()
        model.fit(XGlobal, YGlobal,
                  validation_data=(XValGlobal, YValGlobal),
                  epochs=50,
                  batch_size=params['batch_size'],
                  callbacks=[model_checkpoint, early_stop],
                  verbose=2)
        cf_time = (time.time() - cf_start)

        Y_predicted = model.predict(XValGlobal, workers=4, verbose=2)

        y_val = np.argmax(YValGlobal, axis=1)
        Y_predicted = np.argmax(Y_predicted, axis=1)

        cf_val = metrics.confusion_matrix(y_val, Y_predicted)
        acc_val = metrics.accuracy_score(y_val, Y_predicted)
        precision_val = metrics.precision_score(y_val, Y_predicted, average='binary')
        recall_val = metrics.recall_score(y_val, Y_predicted, average='binary')
        f1_val = metrics.f1_score(y_val, Y_predicted, average='binary')

        test_start_time = time.time()
        pred = model.predict(XTestGlobal, workers=4, verbose=2)
        test_elapsed_time = time.time() - test_start_time

        pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(YTestGlobal, axis=1)

        cf_test = metrics.confusion_matrix(y_eval, pred)
        acc_test = metrics.accuracy_score(y_eval, pred)
        precision_test = metrics.precision_score(y_eval, pred, average='binary')
        recall_test = metrics.recall_score(y_eval, pred, average='binary')
        f1_test = metrics.f1_score(y_eval, pred, average='binary')

        result = {
            "tid": tid,
            "n_params": trainable_params,
            "unit1": params["unit1"],
            "unit2": params["unit2"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "dropout": params["dropout"],
            "cf_time": cf_time,
            "TP_val": cf_val[0][0],
            "FP_val": cf_val[0][1],
            "TN_val": cf_val[1][1],
            "FN_val": cf_val[1][0],
            "OA_val": acc_val,
            "P_val": precision_val,
            "R_val": recall_val,
            "F1_val": f1_val,
            "test_time": int(test_elapsed_time),
            "TP_test": cf_test[0][0],
            "FP_test": cf_test[0][1],
            "FN_test": cf_test[1][0],
            "TN_test": cf_test[1][1],
            "OA_test": acc_test,
            "P_test": precision_test,
            "R_test": recall_test,
            "F1_test": f1_test,
        }

        SavedParameters.append(result)

        if SavedParameters[-1]["F1_val"] > best_val_acc:
            print("new model saved:" + str(SavedParameters[-1]))
            model.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "_model.h5")))
            del model
            best_val_acc = SavedParameters[-1]["F1_val"]

        SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

        try:
            with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)
        except IOError:
            print("I/O error")

        loss_hp = -result["F1_val"]
        gc.collect()
    return {'loss': loss_hp, 'status': STATUS_OK}


def hyperparameter_tuning(dataset_name):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global best_ae
    global best_params

    global SAVE_PATH_
    global CHECKPOINT_PATH_

    BASE_DIR = Path(__file__).resolve().parent
    BASE_DIR.joinpath('session').mkdir(exist_ok=True)
    BASE_DIR.joinpath('trained_ae').mkdir(exist_ok=True)

    i = 1
    flag = True

    while flag:

        TEMP_FILENAME = f"{dataset_name}-binary-LSTM-{i}"
        TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
            SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

            os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/")

            print(f'MODEL SESSION: {SAVE_PATH_}')

    # Load and preprocess the training and testing data
    train = pd.read_csv(os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'train_binary.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'test_binary.csv'))

    XGlobal, YGlobal = parse_data(train, dataset_name=dataset_name, mode='np',
                                  classification_mode='binary')
    XTestGlobal, YTestGlobal = parse_data(test, dataset_name=dataset_name, mode='np',
                                          classification_mode='binary')

    XGlobal = np.reshape(XGlobal, (XGlobal.shape[0], 1, XGlobal.shape[1])).astype('float32')
    XTestGlobal = np.reshape(XTestGlobal, (XTestGlobal.shape[0], 1, XTestGlobal.shape[1])).astype('float32')

    YGlobal = tf.keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = tf.keras.utils.to_categorical(YTestGlobal, num_classes=2)

    XGlobal, XValGlobal, YGlobal, YValGlobal = train_test_split(XGlobal,
                                                                YGlobal,
                                                                test_size=0.2,
                                                                stratify=YGlobal,
                                                                random_state=0
                                                                )

    print('train set:', XGlobal.shape, YGlobal.shape)
    print('validation set:', XValGlobal.shape, YValGlobal.shape)
    print('test set:', XTestGlobal.shape, YTestGlobal.shape)

    ae_hyperparameters_to_optimize = {
        # "hidden_size": [32, 64, 128],
        # "optimizer": ['sgd', 'adam']
        "hidden_size": [128],
        "optimizer": ['sgd']
    }
    keys = list(ae_hyperparameters_to_optimize.keys())
    values = list(ae_hyperparameters_to_optimize.values())
    for combination in product(*values):
        params_ae = {keys[i]: combination[i] for i in range(len(keys))}
        train_ae(params_ae, dataset_name)

    output_of_latent = best_ae.layers[1].output
    encoder_ae = Model(inputs=best_ae.input, outputs=output_of_latent)

    XGlobal = encoder_ae.predict(XGlobal, workers=4, verbose=2)
    XValGlobal = encoder_ae.predict(XValGlobal, workers=4, verbose=2)
    XTestGlobal = encoder_ae.predict(XTestGlobal, workers=4, verbose=2)

    cf_hyperparameters = {
        "unit1": hp.choice("unit1", [512, 1024]),
        "unit2": hp.choice("unit2", [512, 1024]),
        "batch_size": hp.choice("batch_size", [128, 256, 512]),
        'dropout': hp.uniform("dropout", 0, 1),
        "learning_rate": hp.uniform("learning_rate", 0.00001, 0.01)
    }
    trials = Trials()
    # spark_trials = SparkTrials()
    fmin(train_cf, cf_hyperparameters,
         trials=trials,
         algo=tpe.suggest,
         max_evals=30,
         rstate=np.random.default_rng(0))
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW_NB15', required=True,
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')
    parser.add_argument('--result', type=str, required=False,
                        help='path of hyper-parameter training result table .csv file')

    args = parser.parse_args()

    if args.result is not None:
        result_path = args.result
    else:
        result_path = None

    hyperparameter_tuning(args.dataset)
