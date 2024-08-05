import sys, os, getopt, re
# unsupervised greedy layer-wise pretraining for blobs classification problem
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np 
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import initializers
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from data import read_gene_data, read_phenotype_data
from data_v2 import read_aug_phenotype_data
import random 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing
import argparse


# define, fit and evaluate the base autoencoder
def base_autoencoder(trainX, val_X, sample_number,noise_number,dropout_number,file_number,aug_type,dimensions,squash_func,weight_initial,weight_mean,weight_stddev,learning_rate,momentum,run_ID):
    # define noise factor
    noise_factor = noise_number
    X_train_noisy = trainX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=trainX.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    # Without Noise
    #X_train_noisy = trainX
    if weight_initial==1:
        print(weight_initial)
        initializer = initializers.RandomNormal(mean=weight_mean, stddev=weight_stddev)
    model = Sequential()
    if weight_initial==1:
        model.add(Dense(dimensions, input_dim=len(trainX[1]), activation=squash_func,kernel_initializer=initializer))
    else:
        model.add(Dense(dimensions, input_dim=len(trainX[1]), activation=squash_func)) 
    # if applying regularization to the first layer
    #model.add(Dense(300, input_dim=len(trainX[1]), activation='relu',activity_regularizer=regularizers.l1(1e-3))) 
    model.add(Dropout(dropout_number))
    if weight_initial==1: 
        model.add(Dense(len(trainX[1]), activation=squash_func,kernel_initializer=initializer))
    else:    
        model.add(Dense(len(trainX[1]), activation=squash_func))
    # compile model
    model.compile(loss='mse', optimizer=SGD(lr=learning_rate, momentum=momentum))
    model_name = "/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/autoencoder_pretraining/ae_models/model_"+run_ID +  ".h5"

    #patience and early stopping
    es = EarlyStopping(monitor='loss',patience=20)
    mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min',verbose=1)

    # fit model
    history = model.fit(X_train_noisy, trainX, epochs=300, verbose=1,batch_size=32,shuffle=True,callbacks=[es,mcp_save],validation_data=(val_X,val_X))

    # evaluate reconstruction loss
    train_mse = model.evaluate(X_train_noisy, trainX, verbose=1)
    val_mse = model.evaluate(val_X, val_X, verbose=1)
    print('> reconstruction error train=%.3f, val=%.3f' % (train_mse, val_mse))

    #Save the Model
    model.save(model_name)
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer("dense_1").output)
    encoded_train = intermediate_layer_model.predict(trainX)
    encoded_val = intermediate_layer_model.predict(val_X)
    np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/encoded_train/encoded_val_' + run_ID + '.txt', encoded_val, delimiter=',')
    np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/encoded_train/encoded_train_' + run_ID + '.txt', encoded_train, delimiter=',')
    return history


def plot_history(history_aug, sample_number,noise_number,dropout_number,file_number,aug_type,dimensions,squash_func,weight_initial,weight_mean,weight_stddev,learning_rate,momentum,run_ID):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(history_aug.history['loss'])
    plt.plot(history_aug.history['val_loss'])
    plt.title('Autoencoder Pre-training loss', fontsize=22)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks([0,0.04,0.08,0.12,0.16,0.2],fontsize=15)
    plt.legend(['aug_train','aug_val'], loc='upper right', fontsize=18)
    fig.subplots_adjust(left=0.16,bottom=0.16)
    #plotName="autoencoder_pretraining_"+aug_type+"_F"+str(file_number)+ "S" + str(sample_number) + "N" + str(noise_number) + "D" + str(dropout_number) + "Dim" + str(dimensions) +".pdf"
    plotName = run_ID + '.pdf'
    fig.savefig(str(plotName))

 
def format_expression_data(expression_data):
    formatted_expression_data = []
    for ID in expression_data.keys():
        for line in expression_data[ID]:
            x = [float(x) for x in line[1:]]
            formatted_expression_data.append(x)
    return np.array(formatted_expression_data)
                
def normalize_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data


def main():

    # read in user input
    parser = argparse.ArgumentParser(description = 'Train a model on a dataset with certain parameters')
    parser.add_argument('train_X_file',type = str, help = 'Train X File')
    parser.add_argument('val_X_file',type = str, help = 'Val X File')
    parser.add_argument('aug_train_X_file',type = str, help = 'Augmented Train X File')
    parser.add_argument('file_number',type = int, help = 'Replicate File Number')
    parser.add_argument('sample_number',type = int, help = 'Number of Added Augmented Samples')
    parser.add_argument('noise_factor',type = float, help = 'Percentage of Noise Added to Samples')
    parser.add_argument('dropout_rate',type = float, help = 'Rate of Layers Dropped Out')
    parser.add_argument('aug_type',type = str, help = 'Presence of Augmented Data for Training')
    parser.add_argument('dimensions',type = int, help = 'Number of Dimensions Reduced to')
    parser.add_argument('squash_func',type = str, help = 'Squashing Function')
    parser.add_argument('weight_initial',type = int, help = 'Using an Alternative Weight Initialization')
    parser.add_argument('weight_mean',type = float, help = 'weight initialization mean')
    parser.add_argument('weight_stddev',type = float, help = 'weight initialization standard deviation')
    parser.add_argument('learning_rate',type = float, help = 'learning rate')
    parser.add_argument('momentum',type = float, help = 'momentum')
    parser.add_argument('run_ID',type=str, help = 'ID of all the run parameters')
    args = parser.parse_args()

    # assign input to variables
    train_X_file = args.train_X_file
    val_X_file = args.val_X_file
    aug_train_X_file = args.aug_train_X_file
    file_number = args.file_number
    sample_number = args.sample_number
    noise_number = args.noise_factor
    dropout_number = args.dropout_rate
    aug_type = args.aug_type
    dimensions = args.dimensions
    squash_func = args.squash_func
    weight_initial = args.weight_initial
    weight_mean = args.weight_mean
    weight_stddev = args.weight_stddev
    learning_rate = args.learning_rate
    momentum = args.momentum
    run_ID = args.run_ID

    # reading the expression datafiles
    train_X, feature_headers_unlabeled = read_gene_data(train_X_file)
    aug_train_X, aug_feature_headers = read_gene_data(aug_train_X_file)
    val_X, val_headers = read_gene_data(val_X_file)

    # format data for training
    train_X = format_expression_data(train_X)
    aug_train_X = format_expression_data(aug_train_X)
    val_X = format_expression_data(val_X)
    
    # Normalize the data
    train_X = normalize_data(train_X)
    aug_train_X = normalize_data(aug_train_X)
    val_X = normalize_data(val_X)

    #Combine augmented and unlabeled data for pre-training the ae
    if sample_number != 0:
        aug_train_X = random.sample(list(aug_train_X),int(sample_number))
        X = np.concatenate([aug_train_X,train_X])
    else:
        X = train_X

    #Train the autoencoder 
    #history_train = base_autoencoder(train_X,val_X,sample_number,noise_number,dropout_number,file_number,"train")
    history_aug = base_autoencoder(X,val_X,sample_number,noise_number,dropout_number,file_number,aug_type,dimensions,squash_func,weight_initial,weight_mean,weight_stddev,learning_rate,momentum,run_ID)
    # plot the training history
    plot_history(history_aug, sample_number,noise_number,dropout_number,file_number,aug_type,dimensions,squash_func,weight_initial,weight_mean,weight_stddev,learning_rate,momentum,run_ID)

if __name__ == "__main__":
    main()



