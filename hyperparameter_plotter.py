import sys, os, getopt, re
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
import matplotlib.lines as mlines

def print_aggregated_min_metrics(all_loss, all_val_loss, al, avl, all_loss_a, all_val_loss_a, al_a, avl_a):
    # Find the minimum loss from each set
    min_loss = ([np.min(loss) for loss in all_loss])
    min_val_loss = ([np.min(val_loss) for val_loss in all_val_loss])
    min_al = ([np.min(loss) for loss in al])
    min_avl = ([np.min(val_loss) for val_loss in avl])
    min_all_loss_a = ([np.min(loss) for loss in all_loss_a])
    min_all_val_loss_a = ([np.min(val_loss) for val_loss in all_val_loss_a])
    min_al_a = ([np.min(loss) for loss in al_a])
    min_avl_a = ([np.min(val_loss) for val_loss in avl_a])

    # Aggregate metrics using the minimum losses
    print("Aggregated Metrics based on Minimum Losses:")
    print_metrics([min_loss], "Minimum Org Train Loss")
    print_metrics([min_val_loss], "Minimum Org Validation Loss")
    print_metrics([min_al], "Minimum min Train Loss")
    print_metrics([min_avl], "Minimum min Validation Loss")
    print_metrics([min_all_loss_a], "Minimum Aug Train Loss")
    print_metrics([min_all_val_loss_a], "Minimum Aug Validation Loss")
    print_metrics([min_al_a], "Minimum min Aug Train Loss")
    print_metrics([min_avl_a], "Minimum min Aug Validation Loss")

def print_metrics(values, label):
    mean = np.mean(values)
    median = np.median(values)
    minimum = np.min(values)
    maximum = np.max(values)
    std_dev = np.std(values)

    print(f"{label}:")
    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Minimum: {minimum:.4f}")
    print(f"Maximum: {maximum:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")

def find_repeated_values(filename):
    all_loss = []
    all_val_loss = []
    pattern = r'loss: (\d+\.\d+) - val_loss: (\d+\.\d+)'
    print(filename)
    with open(filename, 'r') as file:
        for line in file:   
            line = line.rstrip()   
            #print(line)
            with open(line, 'r') as f:
                loss = []
                val_loss = []
                for line in f:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        loss_value, val_loss_value = match
                        loss.append(float(loss_value))
                        val_loss.append(float(val_loss_value))
                all_loss.append(loss)
                all_val_loss.append(val_loss)
    return all_loss, all_val_loss

def find_min(line):
    all_loss = []
    all_val_loss = []
    pattern = r'loss: (\d+\.\d+) - val_loss: (\d+\.\d+)'
    with open(line, 'r') as f:
        loss = []
        val_loss = []
        for line in f:
            matches = re.findall(pattern, line)
            for match in matches:
                loss_value, val_loss_value = match
                loss.append(float(loss_value))
                val_loss.append(float(val_loss_value))
        all_loss.append(loss)
        all_val_loss.append(val_loss)
    return all_loss, all_val_loss

def plotter(all_loss,all_val_loss,al,avl,all_loss_a,all_val_loss_a,al_a,avl_a):
    fig, axs = plt.subplots(1, 2,figsize=(12, 5))
    for Loss, Val_Loss in zip(all_loss, all_val_loss):
        if len(Loss) < 1000: 
            axs[0].plot(list(range(1, len(Loss) + 1)), Loss, c="Blue",alpha=.05)
            axs[0].plot(list(range(1, len(Val_Loss) + 1)), Val_Loss, c="Red",alpha=.05)
    for Loss, Val_Loss in zip(al, avl):
        axs[0].plot(list(range(1, len(Loss) + 1)), Loss, c="Blue",linewidth=4)
        axs[0].plot(list(range(1, len(Val_Loss) + 1)), Val_Loss, c="Red",linewidth=4)
    for Loss, Val_Loss in zip(all_loss_a, all_val_loss_a):
        if len(Loss) < 1000:
            axs[1].plot(list(range(1, len(Loss) + 1)), Loss, c="Blue",alpha=.05)
            axs[1].plot(list(range(1, len(Val_Loss) + 1)), Val_Loss, c="Red",alpha=.05)
    for Loss, Val_Loss in zip(al_a, avl_a):
        axs[1].plot(list(range(1, len(Loss) + 1)), Loss, c="Blue",linewidth=4)
        axs[1].plot(list(range(1, len(Val_Loss) + 1)), Val_Loss, c="Red",linewidth=4)
    
    #print(np.concatenate(all_loss),np.concatenate(all_val_loss))

    #plt.title('Autoencoder Loss', fontsize=22)
    #blue_line = mlines.Line2D([], [], color='blue', label='Train Loss', alpha=1)
    #red_line = mlines.Line2D([], [], color='red', label='Val Loss', alpha=1)
    #legend = plt.legend(handles=[blue_line, red_line], loc='upper right', fontsize=18)
    #plt.ylabel('Loss (Mean Squared Error)', fontsize=20)
    #plt.xlabel('Epoch', fontsize=20)
    for ax in axs:
        ax.set_ylim(0, 3.5)
        #ax.set_yticks([i / 100 for i in range(2, 11)])
    #     #ax.set_xlim(0,75)
    #     ax.set_xticks([0,10,20,30,40,50,60,70])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
    plt.tight_layout(pad=1)
    #plot_name = "hyperparameter_182_autoencoder_loss.pdf"
    plot_name = "hyperparameter_22k_fine_tuning.pdf"
    plt.savefig(plot_name)
    plt.clf()

# Hyper Parameter:
#Autoencoder
# filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/regression/hyperparameter_tuner.txt'
# filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/smote/hyperparameter_tuner.txt'
# f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/regression/autoencoder_pretrainer_rhythmic_test_Noi0.1_Dro0.1_lr0.9_m0.9_Fil1_r1_org/autoencoder_pretrainer_rhythmic_test_Noi0.1_Dro0.1_lr0.9_m0.9_Fil1_r1_org.o9962581'
# f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/smote/autoencoder_pretrainer_rhythmic_regression_test_Noi0.1_Dro0.1_Dim100_lr0.9_m0.9_Fil1_r4_aug/autoencoder_pretrainer_rhythmic_regression_test_Noi0.1_Dro0.1_Dim100_lr0.9_m0.9_Fil1_r4_aug.o9962612'
#filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/hyperparameter_tuner_aug.txt'
#filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/hyperparameter_tuner_org.txt'
#f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test.o9972919'
#f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test.o9956318'
#Fine Tuning
filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/rhythmic_fine_tuning/hyperparameter_tuning_org.txt'
filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/rhythmic_fine_tuning/hyperparameter_tuning_aug.txt'
f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/rhythmic_fine_tuning/fine_tuning_Noi0.3_Dro0.3_lr0.05_Dec0.001_m0.3_Fil1_r1_org_v11/fine_tuning_Noi0.3_Dro0.3_lr0.05_Dec0.001_m0.3_Fil1_r1_org_v11.o9982298'
f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/rhythmic_fine_tuning/fine_tuning_Noi0.5_Dro0.1_lr0.05_Dec0.01_m0.7_Fil1_r2_aug_v11/fine_tuning_Noi0.5_Dro0.1_lr0.05_Dec0.01_m0.7_Fil1_r2_aug_v11.o9957619'

#filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/22k_fine_tuning/hyperparameter_tuner_aug.txt'
#filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/22k_fine_tuning/hyperparameter_tuner_org.txt'
#f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/22k_fine_tuning/fine_tuning_22k_Noi0.5_Dro0.5_lr0.05_Dec0.001_m0.3_Fil4_r1_org_v11/fine_tuning_22k_Noi0.5_Dro0.5_lr0.05_Dec0.001_m0.3_Fil4_r1_org_v11.o9970193'
#f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/22k_fine_tuning/fine_tuning_22k_Noi0.5_Dro0.5_lr0.05_Dec0.001_m0.3_Fil4_r2_aug_v11/fine_tuning_22k_Noi0.5_Dro0.5_lr0.05_Dec0.001_m0.3_Fil4_r2_aug_v11.o9970222'


#filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/hyperparameter_tuner_aug.txt'
#filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/hyperparameter_tuner_org.txt'
#f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test.o9972919'
#f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test.o9956318'


# Performance:
# filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/regression/182_org_performance.txt '
# filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/smote/182_aug_performance.txt '
# f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/regression/autoencoder_pretrainer_rhythmic_test_Noi0.1_Dro0.1_lr0.9_m0.9_Fil1_r1_org/autoencoder_pretrainer_rhythmic_test_Noi0.1_Dro0.1_lr0.9_m0.9_Fil1_r1_org.o9962581'
# f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/smote/autoencoder_pretrainer_rhythmic_regression_test_Noi0.1_Dro0.1_Dim100_lr0.9_m0.9_Fil1_r4_aug/autoencoder_pretrainer_rhythmic_regression_test_Noi0.1_Dro0.1_Dim100_lr0.9_m0.9_Fil1_r4_aug.o9962612'
#filename_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/22k_aug_performance.txt '
#filename_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/22k_org_performance.txt '
#f_org = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test/autoencoder_pretrainer_Noi0.1_Dro0.1_lr0.9_m0.9_Fil2_r1_org_test.o9972919'
#f_aug = '/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/ExpressionProfiles/gene_expression_data/PLS/autoEncoding_AS/run_output_files/autoencoder_pretraining_parameter_testing/autoencoder_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test/regression_Noi0.1_Dro0.1_Dim300_lr0.9_m0.9_Fil5_r2_aug_test.o9956318'


all_loss_aug, all_val_loss_aug = find_repeated_values(filename_aug)
all_loss_org, all_val_loss_org = find_repeated_values(filename_org)
al_aug, avl_aug = find_min(f_aug)
al_org, avl_org = find_min(f_org)
plotter(all_loss_org,all_val_loss_org,al_org,avl_org,all_loss_aug,all_val_loss_aug,al_aug,avl_aug)
print_aggregated_min_metrics(all_loss_org,all_val_loss_org,al_org,avl_org,all_loss_aug,all_val_loss_aug,al_aug,avl_aug)


