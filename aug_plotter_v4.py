import sys, os, getopt, re
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from data import read_gene_data, read_phenotype_data
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
from collections import defaultdict

def usage():
    script = os.path.basename(__file__)
    print ("\n\nUsage:  " + script + " <Encoded_Train> <Train_Y> <Encoded_Test> <Test_Y> + <file_number>")
    sys.exit()


# standardize data for SVM model
def standardize_data(data):
    data = StandardScaler().fit_transform(data)
    return data

def convert_to_2D(TOD):
    #TOD = TOD + 6.0
    if TOD < 0:
        TOD = TOD + 24
    rads = (math.pi/12)*TOD
    radsCorrected = (math.pi/2)-rads
    x = math.cos(radsCorrected)
    y = math.sin(radsCorrected)
    return x,y


def plotter(X_train,Y_train,X_test,Y_test,X_val,Y_val,plot_name):
    #X = TSNE(n_components=2).fit_transform(X)
    x_train = X_train[:,0]
    y_train = X_train[:,1]
    #x_test = X_test[:,0]
    #y_test = X_test[:,1]
    x_val = X_val[:,0]
    y_val = X_val[:,1]
    print(len(x_val),len(y_val),len(Y_val))
    ##plt.scatter(x_val,y_val,c=Y_val,cmap='hsv',marker = "D",label = "val")
    print(len(X_train),len(Y_train)) 
    ##plt.scatter(x_train,y_train,c=Y_train,cmap='hsv', marker = "o",label = "train")
    plt.scatter(x_val,y_val,marker = "D",label = "val")
    plt.scatter(x_train,y_train, marker = "o",label = "train")
    #plt.scatter(x_test,y_test,c=Y_test,cmap='hsv',marker = "D",label = "test")
    #plt.colorbar(label="TOD")
    #plt.xlim(-1.5,2)
    #plt.ylim(-1.5,2)
    plt.xlabel('Predicted x')
    plt.ylabel('Predicted y')
    plt.legend()
    plt.savefig('circle_'+plot_name)
    plt.clf()


def plot_tsne(X,Y,plot_name):
    X = TSNE(n_components=2).fit_transform(X)
    X = np.array(X)
    # Creating scatter plot
    plt.figure()
    #plt.scatter(X[:,0],X[:,1],c=Y)
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='hsv')
    plt.colorbar(label='TOD')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig(plot_name)
    plt.close()

def plot_all_tsne(X,Y,plot_name,X_org,Y_org,X_aug,Y_aug):
    X = np.array(TSNE(n_components=2,perplexity=30.0).fit_transform(X))
    X_org = np.array(TSNE(n_components=2,perplexity=30.0).fit_transform(X_org))
    X_aug = np.array(TSNE(n_components=2,perplexity=30.0).fit_transform(X_aug))

    max_abs_x_org = int(max(abs(X_org[:, 0].min()), abs(X_org[:, 0].max())))+1
    max_abs_x = int(max(abs(X[:, 0].min()), abs(X[:, 0].max())))+1
    max_abs_x_aug = int(max(abs(X_aug[:, 0].min()), abs(X_aug[:, 0].max())))+1

    # Creating scatter plot
    fig, axs = plt.subplots(1, 3,figsize=(12, 6))
    print(len(X_org[:,0]),len(X_org[:,1]))
    colormap = plt.get_cmap('twilight_shifted')
    axs[0].scatter(X_org[:,0],X_org[:,1],c=Y_org,cmap=colormap)
    axs[0].set_xticks(np.linspace(-max_abs_x_org, max_abs_x_org, 3),labelsize=15)
    axs[0].set_title("No Autoencoder",fontsize=22)
    axs[1].scatter(X[:,0],X[:,1],c=Y,cmap=colormap)
    axs[1].set_xticks(np.linspace(-max_abs_x, max_abs_x, 3),labelsize=15)
    axs[1].set_title("Train Autoencoder",fontsize=22)
    axs[2].scatter(X_aug[:,0],X_aug[:,1],c=Y_aug,cmap=colormap)
    axs[2].set_xticks(np.linspace(-max_abs_x_aug, max_abs_x_aug, 3))
    axs[2].set_title("Augmented Autoencoder",fontsize=22)
    fig.supxlabel('Dimension 1',fontsize=22)
    fig.supylabel('Dimension 2',fontsize=22)

    #cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    #cbar = fig.colorbar(sc2,cax=axs[2], label='Time of Day (Hours)')
    #cbar.ax.tick_params(labelsize=15)
    plt.tight_layout(pad=1)
    plt.savefig("autoencoder.pdf")
    plt.close()


def read_val_phenotype_data(filename):
    lines = [line.strip().split() for line in open(filename, "r").readlines()]
    header = tuple(lines[0])
    lines = lines[1:]
    data = {}
    for line in lines:
        line = {label:value for label,value in zip(header,line)}
        line["ID"] = int(line["ID"])
        #line["Age"] = int(line["Age"])
        if line["TOD"] != "NA":
            line["TOD"] = float(line["TOD"])
        data[line["ID"]] = line
    return data, header

def create_data_label(data,phenotype_data):
    X=[]
    Y=[]
    Age=[]
    for ID in data.keys():
        try:
            if phenotype_data[ID]["TOD"] != "NA":
                for line in data[ID]:
                    x = [float(x) for x in line[1:]]
                    X.append(x)
                    y = phenotype_data[ID]["TOD"]
                    #age = phenotype_data[ID]["Age"]
                    if y < 0:
                        y = y + 24
                    Y = np.append(Y,y)
                    #Age.append(age)
        except (IndexError, KeyError):
            pass
    return X,Y

def create_aug_data_label(data,phenotype_data):
    X=[]
    Y=[]
    for ID in data.keys():
        x = [float(x) for x in data[ID]]
        # append the expression data
        X.append(x)
        y = phenotype_data[0]
        y = y[ID]['TOD']
        if y < 0:
            y = y + 24
        Y.append(y) # remove and uncomment the rest to switch to 2D
        #x_1,y_1 = convert_to_2D(y)
        #print(ID,y,x_1,y_1)
        ## append the time data
        #Y.append([x_1,y_1])
    return X,Y

def read_aug_gene_data(filename):
    f = open(filename, "r")
    data = {}
    header = []
    values = []
    for line in f:
        if not line.startswith('sampleID') and not line.startswith('gene_name'):
            terms = line.strip().split()
            sample_ID = terms[0]
            human_ID = int(sample_ID.split(".")[1])
            # formatted already and loaded
            data[human_ID] = [float(x) for x in terms[1:]]
        else:
            terms = line.strip().split()
            header =  ["gene_name"] + terms
    return data, header
 
def read_aug_phenotype_data(filename):
    lines = [line.strip().split() for line in open(filename, "r").readlines()]
    header = tuple(lines[0])
    lines = lines[1:]
    data = {}
    for line in lines:
        line = {label:value for label,value in zip(header,line)}
        line["ID"] = int(line["ID"])
        #line["Age"] = int(line["Age"])
        #line["PMI"] = float(line["PMI"])
        #line["RIN"] = float(line["RIN"])
        #line["pH"] = float(line["pH"])
        if line["TOD"] != "NA":
            line["TOD"] = float(line["TOD"])
            #line["Age"] = line["Age"]
        data[line["ID"]] = line
    return data, header
 
 
def main():


    opts, files = getopt.getopt(sys.argv[1:],"hvo:",["help"])
    if len(files) != 14:
        for file in files:
            print(len(files),"\n")
        usage()

        
    encoded_train = files[0]
    train_X_FileName = files[1]
    phenotype_train_filename = files[2]
    phenotype_val_filename = files[3]
    val_X_FileName = files[4]
    encoded_val = files[5]
    output_filename = files[6]
    aug_train_X_FileName = files[7]
    aug_phenotype_train_filename = files[8]
    test_X_FileName = files[9]
    phenotype_test_filename = files[10]
    encoded_test = files[11]
    encoded_aug = files[12]
    org_X_FileName = files[13]

    train_data, feature_headers = read_gene_data(train_X_FileName)
    org_data, feature_headers = read_gene_data(org_X_FileName)
    val_data, val_headers = read_gene_data(val_X_FileName)
    test_data, aug_feature_headers = read_gene_data(test_X_FileName)
    aug_train_data, feature_headers = read_gene_data(aug_train_X_FileName)
    phenotype_train_data, _ = read_aug_phenotype_data(phenotype_train_filename)
    phenotype_val_data, _ = read_aug_phenotype_data(phenotype_val_filename)
    phenotype_test_data, _ = read_aug_phenotype_data(phenotype_test_filename)
    aug_phenotype_train_data, _ = read_aug_phenotype_data(aug_phenotype_train_filename)

    #LOAD DATA
    encoded_train = np.loadtxt(encoded_train,delimiter=',')
    encoded_test = np.loadtxt(encoded_test,delimiter=',')
    encoded_val = np.loadtxt(encoded_val,delimiter=',')
    encoded_aug = np.loadtxt(encoded_aug,delimiter=',')
    
    train_X,train_y = create_data_label(train_data,phenotype_train_data)
    val_X,val_y =  create_data_label(val_data,phenotype_val_data)
    test_X,test_y =  create_data_label(test_data,phenotype_val_data)
    aug_train_X,aug_train_y = create_data_label(aug_train_data,aug_phenotype_train_data)
    org_X,org_y = create_data_label(org_data,phenotype_train_data)

    # Convert data to numpy array                                                                                                                                                                        
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    val_y = np.array(val_y)
    aug_y = np.array(aug_train_y)
    org_y = np.array(org_y)

    #X = np.concatenate([encoded_train,encoded_val,encoded_test]) # encoded train has both original and augmented data
    #age = np.concatenate([train_age,val_age,test_age])
    X  = standardize_data(train_X)
    X_aug_train = standardize_data(aug_train_X)
    X_train = standardize_data(encoded_train)
    X_val = standardize_data(encoded_val)
    X_test = standardize_data(encoded_test)
    X_aug = standardize_data(encoded_aug)
    X_org = standardize_data(org_X)
    print(len(X_org),len(org_y))
    Y = train_y
    #Y = np.concatenate([Y,val_y,test_y])
    plot_all_tsne(X_train,train_y,output_filename,X,train_y,X_aug,train_y)
    #plot_all_tsne(X_aug,aug_y, "aug_"+output_filename,X_org,org_y)


if __name__ == "__main__":
    main()


