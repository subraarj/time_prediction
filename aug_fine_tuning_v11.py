import sys, os, getopt, re
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing
import math
from collections import defaultdict
import argparse
import tensorflow as tf
import keras.losses
import matplotlib as mpl
from matplotlib import cm
#tf.enable_eager_execution()

# needs to be defined as activation class otherwise error
# AttributeError: 'Activation' object has no attribute '__name__'    
class Sin(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Sin, self).__init__(activation, **kwargs)
        self.__name__ = 'sin'

# define activation
def sin(x):
  return K.sin(x)
  
get_custom_objects().update({'sin': Sin(sin)})

# def read_gene_data(filename):
#     f = open(filename, "r")
#     data = {}
#     header = []
#     values = []
#     for line in f:
#         if not line.startswith('sampleID') and not line.startswith('gene_name'):
#             terms = line.strip().split()
#             sample_ID = terms[0]
#             human_ID = int(sample_ID.split(".")[1])
#             # formatted already and loaded
#             data[sample_ID] = [float(x) for x in terms[1:]]
#         else:
#             terms = line.strip().split()
#             header =  ["gene_name"] + terms
#     return data, header

def read_gene_data(filename):
    f = open(filename, "r")
    data = defaultdict(list)

    sample_IDs = []
    values = []
    for line in f:
        line = line.strip().split()
        if "BA" in line[0]:
            sample_IDs.append(line[0])
            values.append(line[1:])
        else:
            values.append(line)

    header = values[0]
    values = values[1:]
    if len(header) == len(values[0]):
        header = ["gene_name"] + header
    header = tuple(header)

    for sample_ID,value in zip(sample_IDs,values):
        human_ID = int(sample_ID.split(".")[1])
        #value = [float(x) for x in value]
        value.insert(0, sample_ID)
        data[human_ID].append(value)
    return data, header

def read_aug_phenotype_data(filename):
    lines = [line.strip().split() for line in open(filename, "r").readlines()]
    header = tuple(lines[0])
    lines = lines[1:]
    data = {}
    for line in lines:
        line = {label:value for label,value in zip(header,line)}
        line["ID"] = int(line["ID"])
        if line["TOD"] != "NA":
            line["TOD"] = float(line["TOD"])
        data[line["ID"]] = line
    return data, header

def read_phenotype_data(filename):
    phenotype_data = {}
    with open(filename,"r") as phen_file:
        for i,line in enumerate(phen_file):
           line = line.strip().split("\t")
           if i != 0:
               ID = str(line[ID_index])
               if line[TOD_index] != "NA" and "BA_11."+ID not in phenotype_data.keys():
                  TOD = float(line[TOD_index])
                  phenotype_data["BA11."+ID] = TOD
                  phenotype_data["BA47."+ID] = TOD  
           else:
               header = list(line)
               TOD_index = header.index("TOD")
               ID_index = header.index("ID")
    return phenotype_data

def read_phenotype_data_old(filename):
    lines = [line.strip().split() for line in open(filename, "r").readlines()]
    header = tuple(lines[0])
    lines = lines[1:]
    data = {}
    for line in lines:
        line = {label:value for label,value in zip(header,line)}
        line["ID"] = int(line["ID"])
        line["Age"] = int(line["Age"])
        line["PMI"] = float(line["PMI"])
        line["RIN"] = float(line["RIN"])
        line["pH"] = float(line["pH"])
        if line["TOD"] != "NA":
            line["TOD"] = float(line["TOD"])
        data[line["ID"]] = line
    #print(data)
    return data, header

#custom loss function
def custom_loss_fn(y_true, y_pred):
    #dot_prod = tf.math.reduce_sum(tf.math.multiply(y_pred,y_true), axis=1)
    #norm_pred = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred), axis=1, keepdims=True))
    norm_y_true = tf.nn.l2_normalize(y_true,axis=1)
    norm_y_pred = tf.nn.l2_normalize(y_pred,axis=1)
    dot_prod = tf.math.reduce_sum(tf.math.multiply(norm_y_pred,norm_y_true), axis=1)
    array_of_ones = tf.constant([[1.0] for i in range(K.int_shape(y_pred)[1])])
    pairwise_angles = tf.multiply(tf.constant(2.0),(tf.subtract(array_of_ones,dot_prod)))
    
    return tf.reduce_mean(tf.transpose(pairwise_angles))


# evaluate the autoencoder as a classifier
def evaluate_autoencoder_as_classifier(model, trainX, trainy, valX, valy, file_number,learning_rate,decays,momentums,aug_type,dropout_rate,loss_fn,noise_factor,replicate,run_ID,trainX_org,trainX_aug,testX,testy):
    history = 1
    # output_layer = model.layers[-1]
    # # # remove the output layer
    # model.pop()
    # model.add(Dropout(dropout_rate,name="dropout_2"))
    # model.add(Dense(2, activation=sin, use_bias=True, name="dense_out"))
    # # mark all remaining layers as trainable bcz we want to fine-tune
    # for layer in model.layers:
    #     layer.trainable = True
    
    # # print("valy",valy)
    # epochs =400
    # sgd = SGD(lr=learning_rate, decay=decays, momentum=momentums, nesterov=True)
    # #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse']) # used for non custom loss
    # keras.losses.custom_loss = custom_loss_fn
    # model.compile(loss=custom_loss_fn, optimizer=sgd)
    model_name = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/autoencoder_pretraining/ae_models/fine_tuning/ae_model_' + run_ID+ 'young0.hdf5'
    # #model_name = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/autoencoder_pretraining/ae_models/fine_tuning/ae_model_' + run_ID+ 'young0.hdf5'
    #  # fit model with high patience 
    # # it took a little while for the model to generalize
    # es = EarlyStopping(monitor='loss',patience=30,min_delta=0.01)
    # mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min',verbose=1)
    # # add noise, original was just trainX
    # X_train_noisy = trainX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=trainX.shape) #try 0.1 or 0.05 for scale
    # X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    # history = model.fit(X_train_noisy, trainy, epochs=epochs, verbose=1, batch_size=16, shuffle=True, callbacks=[es,mcp_save], validation_data=(valX,valy))
    # # Load the best model that saved as part of the checkpoint for further evaluation
    get_custom_objects().update({'custom_loss_fn': custom_loss_fn})
    model = load_model(model_name)
    #print('dimensions',model)    
    print(model.metrics_names)
    # test the model
    train_loss = model.evaluate(trainX, trainy, verbose=2)
    val_loss = model.evaluate(valX, valy, verbose=2)
    test_loss = model.evaluate(testX,testy, verbose=2)
    y_pred_train = model.predict(trainX_org)
    y_pred_val = model.predict(valX)
    y_pred_aug = model.predict(trainX_aug)
    y_pred_test = model.predict(testX)
    #save the layer, changed dense_1 to dense_out
    # intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer("dense_out").output)

    # encoded_train = intermediate_layer_model.predict(trainX)
    # encoded_val = intermediate_layer_model.predict(valX)
    # print('train',encoded_train)
    # print('val',encoded_val)
    
    #SAVE THE LOWER REPRESENTATION OF THE DATA
    #np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/encoded_train/encoded_train_'+ run_ID + '.txt', encoded_train, delimiter=',')
    #np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/train_y/train_y_' + run_ID + '.txt',trainy,delimiter = ',')
    #np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/encoded_test/encoded_val_'+run_ID+'.txt', encoded_val, delimiter=',')
    #np.savetxt('/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/test_y/val_y_'+run_ID+'.txt',valy,delimiter = ',')


    # save the model into a file
    #model.save("../2D/model_custom_loss"+str(run_ID) +"_.h5")
    return train_loss, val_loss,test_loss, history, y_pred_train,y_pred_val,y_pred_aug,y_pred_test


def plot_history(history_aug, file_number,lr,decay,momen,aug_type,dropout_rate,noise_factor,replicate,run_ID):
    plt.figure()
    plt.plot(history_aug.history['loss'])
    plt.plot(history_aug.history['val_loss'])
    plt.title('Fine Tuning Loss', fontsize=22)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(['train','val'], loc='upper right', fontsize=18)    
    plotName="fine_tuning_loss" + run_ID + ".pdf"
    plt.savefig(plotName)
    plt.clf()
# use this function if using the original read_phenotype_functions
# def create_data_label(data,phenotype_data):
#     X=[]
#     Y=[]
#     for ID in data.keys():
#         x = [float(x) for x in data[ID]]
#         # append the expression data
#         X.append(x)
#         y = phenotype_data[ID]
#         x_1,y_1 = convert_to_2D(y)
#         print(ID,y,x_1,y_1)
#         # append the time data
#         Y.append([x_1,y_1])
#     return X,Y

def create_data_label(data,phenotype_data):
    X=[]
    Y=[]
    for ID in data.keys():
        print(phenotype_data[0][ID]['TOD'])
        try:
            if phenotype_data[0][ID]['TOD'] != "NA":
                for line in data[ID]:
                    x = [float(x) for x in line[1:]]
                    X.append(x)
                    y = phenotype_data[0][ID]['TOD']
                    x_1,y_1 = convert_to_2D(y)
                    Y.append([x_1,y_1])
                    #Y = np.append(Y,y)
        except (IndexError, KeyError):
            pass
    return X,Y

#Referenced from Jeffs Code
def convert_to_2D(TOD):
    #TOD = TOD + 6.0
    if TOD < 0:
        TOD = TOD + 24
    rads = (math.pi/12)*TOD
    radsCorrected = (math.pi/2)-rads
    x = math.cos(radsCorrected)
    y = math.sin(radsCorrected)
    return x,y
 
def convert_to_TOD(x,y):
    TOD = 0
    TODs = []
    for i in np.arange(0,len(x)):
        if x[i] > 0 and y[i] > 0:
            TOD=(12/np.pi)*np.arctan(x[i]/y[i])
        if x[i] > 0 and y[i] < 0 or x[i] < 0 and y[i] < 0:
            TOD=((12/np.pi)*np.arctan(x[i]/y[i])) + 12
        if x[i] < 0 and y[i] > 0:
            TOD=((12/np.pi)*np.arctan(x[i]/y[i])) + 24
        TODs.append(TOD)
    return TODs

def normalize_data(data):
     min_max_scaler = preprocessing.MinMaxScaler()
     data = min_max_scaler.fit_transform(data)
     return data

def read_val_phenotype_data(filename):
    phenotype_data = {}
    with open(filename,"r") as phen_file:
        for i,line in enumerate(phen_file):
           line = line.strip().split("\t")
           if i != 0:
               ID = str(line[ID_index])
               if line[TOD_index] != "NA":
                  TOD = float(line[TOD_index])
                  phenotype_data["BA_old."+ID] = TOD
           else:
               header = list(line)
               TOD_index = header.index("TOD")
               ID_index = header.index("ID")
    return phenotype_data

def convert_to_circle(x,y):
    t = []
    r = []
    for i in range(len(x)):
        r.append(1.5)
        theta = math.atan2(y[i], x[i])
        t.append(theta)
    return t,r

def rmse(list1, list2):
  """
  This function calculates the root mean squared error (RMSE) between two lists.

  Args:
      list1: The first list.
      list2: The second list.

  Returns:
      The RMSE between the two lists.
  """

  # Check if the lists have the same length
  if len(list1) != len(list2):
    raise ValueError("Lists must have the same length")

  # Calculate the squared differences
  squared_differences = [(a - b) ** 2 for a, b in zip(list1, list2)]

  # Calculate the mean of the squared differences
  mean_squared_difference = np.mean(squared_differences)

  # Take the square root of the mean squared difference
  rmse = np.sqrt(mean_squared_difference)

  return rmse

# X_train,X_val,X_aug,Y_train,Y_val,Y_aug,y_pred_train,y_pred_val,y_pred_aug
def plotter(Y_train,Y_val,Y_aug,Y_test,y_pred_train,y_pred_val,y_pred_aug,y_pred_test,run_ID):
    x_train = y_pred_train[:,0]
    y_train = y_pred_train[:,1]
    x_aug = y_pred_aug[:,0]
    y_aug = y_pred_aug[:,1]
    x_val = y_pred_val[:,0]
    y_val = y_pred_val[:,1]
    x_test = y_pred_test[:,0]
    y_test = y_pred_test[:,1]

    y_t_train = np.array(convert_to_TOD(Y_train[:,0],Y_train[:,1]))
    y_t_val = np.array(convert_to_TOD(Y_val[:,0],Y_val[:,1]))
    y_t_aug = np.array(convert_to_TOD(Y_aug[:,0],Y_aug[:,1]))
    y_t_test = np.array(convert_to_TOD(Y_test[:,0],Y_test[:,1]))

    fig, axs = plt.subplots(1, 2,subplot_kw=dict(projection='polar'))
    s = np.pi*90/180
    e = np.pi*90/180+2*np.pi
    norm = mpl.colors.Normalize(s, e)
    colormap = plt.get_cmap('twilight_r')
    # Scatter plots with color mapping
    #s1 = axs[0, 0].scatter(Y_aug[:,0], Y_aug[:,1], marker="1", label="aug", c=convert_to_TOD(x_aug, y_aug), cmap='hsv') actual point

    # x_c_aug,y_c_aug = convert_to_circle(x_aug,y_aug)
    # s1 = axs[0, 0].scatter(x_c_aug, y_c_aug, c=y_t_aug,cmap='twilight',alpha=0.5,edgecolors='black')
    # axs[0, 0].set_title('Augmented',pad=20,fontsize=12)

    x_c_val,y_c_val = convert_to_circle(x_val,y_val)
    print(len(x_c_val),len(y_c_val))
    axs[0].scatter(x_c_val, y_c_val, c=y_t_val,cmap='twilight',edgecolors='black',alpha=0.7,vmin=0, vmax=24) 
    axs[0].set_title('Validation',pad=20,fontsize=12)

    # x_c_train,y_c_train = convert_to_circle(x_train,y_train)
    # axs[0, 1].scatter(x_c_train, y_c_train, c=y_t_train, cmap='twilight',alpha=0.5,edgecolors='black')
    # axs[0, 1].set_title('Train',pad=20,fontsize=12)

    x_c_test,y_c_test = convert_to_circle(x_test,y_test)
    print(len(x_c_test))
    axs[1].scatter(x_c_test, y_c_test, c=y_t_test, cmap='twilight',edgecolors='black',alpha=0.7,vmin=0, vmax=24)
    axs[1].set_title('Test',pad=20,fontsize=12)
    
    # Set labels and legend
    n = 1000  #the number of secants for the mesh
    t = np.linspace(s,e,n)   #theta values
    r = np.linspace(2,2.3)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = tg                         #define color values as theta value
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i in [0,1]:
        axs[i].pcolormesh(t, r, c.T,norm=norm, cmap=colormap)
        axs[i].set_yticklabels([])                   #turn of radial tick labels (yticks)
        axs[i].set_thetagrids(np.linspace(0, 360, 12, endpoint=False),   ['6','4', '2', '24', '22', '20','18','16','14','12', '10','8'])
        axs[i].grid(False)
    fig.tight_layout(pad=1)
    plt.savefig('circle_'+run_ID+".pdf")
    plt.clf()

    y_p_train = np.array(convert_to_TOD(x_train,y_train))
    y_p_val = np.array(convert_to_TOD(x_val,y_val))
    y_p_aug = np.array(convert_to_TOD(x_aug,y_aug))
    y_p_test = np.array(convert_to_TOD(x_test,y_test))
    print("4",len(y_p_val),len(y_t_val))
    fig, axs = plt.subplots(2, 2)
    x_line = np.linspace(0, 24, 100)
    y_line = x_line
    # Scatter plots
    
    axs[0, 0].scatter(y_t_aug, y_p_aug,alpha=0.5,edgecolors='black',c='#DF0069')
    axs[0,0].plot(x_line, y_line, color='black')
    axs[0, 0].set_title('Augmented',fontsize=14)
    
    axs[1, 0].scatter(y_t_val, y_p_val,alpha=0.5,edgecolors='black',c='#DF0069')
    axs[1,0].plot(x_line, y_line, color='black')
    axs[1, 0].set_title('Validation',fontsize=14)

    axs[0, 1].scatter(y_t_train, y_p_train,alpha=0.5,edgecolors='black',c='#DF0069')
    axs[0,1].plot(x_line, y_line, color='black')
    axs[0, 1].set_title('Train',fontsize=14)

    axs[1, 1].scatter(y_t_test, y_p_test,alpha=0.5,edgecolors='black',c='#DF0069')
    axs[1,1].plot(x_line, y_line, color='black')
    axs[1, 1].set_title('Test',fontsize=14)

    print("Augmented PCoef: ",np.corrcoef(y_t_aug, y_p_aug)[0, 1])
    print("Validation PCoef: ",np.corrcoef(y_t_val, y_p_val)[0, 1])
    print("Train PCoef: ",np.corrcoef(y_t_train, y_p_train)[0, 1])
    print("Test PCoef: ",np.corrcoef(y_t_test, y_p_test)[0, 1])
    print("RMSE: ",rmse(y_t_test, y_p_test))

    # Set labels and legend
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15) 
    fig.text(0.5, 0.04, 'Labeled Time of Day (Hours)', ha='center',fontsize=14)
    fig.text(0.02, 0.5, 'Predicted Time of Day (Hours)', va='center', rotation='vertical',fontsize=14)
    fig.tight_layout(pad=3)
    plt.savefig('labeled_vs_pred_'+run_ID+".pdf")
    plt.clf()
def main():
    usage = 'Usage: ' + sys.argv[0] + '<train data> <phenotype train data> <aug_train data> <val data> <aug_phenotype train file> <phenotype val file> <pre-trained model> <file-number> <learning rate> <decay> <momentum> <aug_type> <dropout_rate> <noise_factor> <replicate> <run_ID> <test data> <phenotype test data>'
    if len(sys.argv) != 19:
      print(usage)
      sys.exit()

    parser = argparse.ArgumentParser(description = 'Fine Tune a model on a dataset with certain parameters')
    parser.add_argument('train_X_FileName',type = str, help = 'Train X File')
    parser.add_argument('phenotype_train_filename',type = str, help = 'Phenotype Training Data File')
    parser.add_argument('aug_train_X_FileName',type = str, help = 'Augmented Train X File')
    parser.add_argument('aug_val_X_FileName',type = str, help = 'Val X File (not augmented)')
    parser.add_argument('aug_phenotype_train_filename',type = str, help = 'Augmented Phenotype Training Data File')
    parser.add_argument('phenotype_val_filename',type = str, help = 'Phenotype Validation File')
    parser.add_argument('pre_trained_model',type = str, help = 'Trained Model from Pretrainer')
    parser.add_argument('file_number',type = float, help = 'Replicate File Number')
    parser.add_argument('learning_rate',type = float, help = 'Learning Rate')
    parser.add_argument('decay',type = float, help = 'Decay Rate')
    parser.add_argument('momentum',type = float, help = 'Momentum Rate')
    parser.add_argument('aug_type',type = str, help = 'Presence of Augmented Data for Training')
    parser.add_argument('dropout_rate',type = float, help = 'dropout_rate')
    parser.add_argument('noise_factor',type = float, help = 'Noise Factor')
    parser.add_argument('replicate',type = float, help = 'Replicate')
    parser.add_argument('run_ID',type = str, help = 'ID of all the run parameters')
    parser.add_argument('test_X_FileName',type = str, help = 'Test X File')
    parser.add_argument('phenotype_test_filename',type = str, help = 'Phenotype Test Data File')
    args = parser.parse_args()

    train_X_FileName = args.train_X_FileName
    phenotype_train_filename = args.phenotype_train_filename
    aug_train_X_FileName = args.aug_train_X_FileName
    aug_val_X_FileName =  args.aug_val_X_FileName
    aug_phenotype_train_filename = args.aug_phenotype_train_filename
    phenotype_val_filename = args.phenotype_val_filename
    pre_trained_model = args.pre_trained_model
    file_number = args.file_number
    learning_rate = args.learning_rate
    decay = args.decay
    momentum = args.momentum
    aug_type = args.aug_type
    dropout_rate = args.dropout_rate
    noise_factor = args.noise_factor
    replicate = args.replicate
    run_ID = args.run_ID
    test_X_FileName = args.test_X_FileName
    phenotype_test_filename = args.phenotype_test_filename

    #print(custom_loss_fn(y_true,y_pred))
    # Reading in Files
    phenotype_train_data = read_aug_phenotype_data(phenotype_train_filename)
    aug_phenotype_train_data = read_aug_phenotype_data(aug_phenotype_train_filename)
    phenotype_val_data = read_aug_phenotype_data(phenotype_val_filename)
    training_data, feature_headers = read_gene_data(train_X_FileName)
    phenotype_test_data = read_aug_phenotype_data(phenotype_test_filename)
    test_data, feature_headers = read_gene_data(test_X_FileName)
    #print("training_data",training_data)
    aug_training_data, aug_feature_headers = read_gene_data(aug_train_X_FileName)
    #print("aug_training_data",aug_training_data)
    val_data, val_headers = read_gene_data(aug_val_X_FileName)
    #print("val_data",val_data)
    # Proccessing Files
    #print("phenotype train", phenotype_train_data)
    X_train,Y_train = create_data_label(training_data,phenotype_train_data)
    X_aug_train,Y_aug_train = create_data_label(aug_training_data,aug_phenotype_train_data)
    X_val,Y_val =  create_data_label(val_data,phenotype_val_data)
    X_test,Y_test =  create_data_label(test_data,phenotype_test_data)
    # Concatenating Files
    X_train_combined = np.concatenate([X_train,X_aug_train])
    Y_train_combined = np.concatenate([Y_train,Y_aug_train])
    #X_train_combined = normalize_data(X_train) # change for org / aug
    #Y_train_combined = Y_train
    # Normalize the data
    if aug_type == "aug": 
        X_train_combined = normalize_data(X_train_combined) # change for org/ aug
        Y_train_combined = np.array(Y_train_combined)
    else:
        X_train_combined = normalize_data(X_train)
        Y_train_combined = np.array(Y_train)
    X_train = normalize_data(X_train)
    Y_train = np.array(Y_train)
    X_val = normalize_data(X_val)
    Y_val = np.array(Y_val)
    X_aug = normalize_data(X_aug_train)
    Y_aug = np.array(Y_aug_train)
    X_test = normalize_data(X_test)
    Y_test = np.array(Y_test)
    print("Y_train_combined",Y_train_combined)
    print(X_train_combined.shape,Y_train_combined.shape,X_val.shape,Y_val.shape) 
    # get the base pre-trained autoencoder model
    model = load_model(pre_trained_model)

    # Fine Tune and Evaluate the base model
    #print(custom_loss_fn(X_train_combined,X_train_combined))
    train_acc_aug, val_acc_aug,test_acc_aug, history_aug,y_pred_train,y_pred_val,y_pred_aug,y_pred_test = evaluate_autoencoder_as_classifier(model,X_train_combined,Y_train_combined, X_val,Y_val, file_number,learning_rate,decay,momentum,aug_type,dropout_rate,custom_loss_fn,noise_factor,replicate,run_ID,X_train,X_aug,X_test,Y_test)
    #train_acc_org, val_acc_org, history_org = evaluate_autoencoder_as_classifier(model,X_train,Y_train, X_val,Y_val, file_number,learning_rate,decay,momentum)
    #print("Original",train_acc_org, val_acc_org)
    print(train_acc_aug, val_acc_aug,test_acc_aug)
    #plot_history(history_aug, file_number,learning_rate,decay,momentum,aug_type,dropout_rate,noise_factor,replicate,run_ID)
    plotter(Y_train,Y_val,Y_aug,Y_test,y_pred_train,y_pred_val,y_pred_aug,y_pred_test,run_ID)

if __name__ == "__main__":
    main()
