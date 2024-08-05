import sys, os, getopt, re
from sklearn.preprocessing import StandardScaler
from data import read_gene_data, read_phenotype_data
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
from collections import defaultdict
import numpy as np
from scipy.stats import ttest_ind

def calculate_metrics_and_t_tests(group1, group2, label1, label2):
    # Calculate metrics
    min_1 = np.min(group1)
    max_1 = np.max(group1)
    mean_1 = np.mean(group1)
    median_1 = np.median(group1)
    q1_1 = np.percentile(group1, 25)
    q3_1 = np.percentile(group1, 75)
    iqr_1 = q3_1 - q1_1

    min_2 = np.min(group2)
    max_2 = np.max(group2)
    mean_2 = np.mean(group2)
    median_2 = np.median(group2)
    q1_2 = np.percentile(group2, 25)
    q3_2 = np.percentile(group2, 75)
    iqr_2 = q3_2 - q1_2

    # Print metrics
    print("Metrics for", label1 + ":")
    print("Min:", min_1)
    print("Max:", max_1)
    print("Mean:", mean_1)
    print("Median:", median_1)
    print("Q1:", q1_1)
    print("Q3:", q3_1)
    print("IQR:", iqr_1)
    print()

    print("Metrics for", label2 + ":")
    print("Min:", min_2)
    print("Max:", max_2)
    print("Mean:", mean_2)
    print("Median:", median_2)
    print("Q1:", q1_2)
    print("Q3:", q3_2)
    print("IQR:", iqr_2)
    print()

def plotter(org_182,aug_182,org_22k,aug_22k):
    group1_182 = np.array(org_182)
    group2_182 = np.array(aug_182)
    group1_22k = np.array(org_22k)
    group2_22k = np.array(aug_22k)

    plt.figure(figsize=(9,7))
    # Plot the scatter points
    plt.scatter(np.ones_like(group1_22k) + np.random.uniform(-0.03, 0.03, size=group1_22k.shape), group1_22k + np.random.uniform(-0.001, 0.001, size=group1_22k.shape), color='red',edgecolors='black')
    plt.scatter(2 * np.ones_like(group2_22k) + np.random.uniform(-0.03, 0.03, size=group2_22k.shape), group2_22k + np.random.uniform(-0.001, 0.001, size=group2_22k.shape), color='blue',edgecolors='black')
    plt.scatter(3 * np.ones_like(group1_182) + np.random.uniform(-0.03, 0.03, size=group1_182.shape), group1_182 + np.random.uniform(-0.001, 0.001, size=group1_182.shape), color='red',edgecolors='black')
    plt.scatter(4 * np.ones_like(group2_182) + np.random.uniform(-0.03, 0.03, size=group2_182.shape), group2_182 + np.random.uniform(-0.001, 0.001, size=group2_182.shape), color='blue',edgecolors='black')
    plt.boxplot([group1_22k,group2_22k,group1_182,group2_182],labels=["All Genes","a","a","Rhythmic Genes"],positions=[1,2,3,4],showfliers=False)
    
    colors = ['red','blue']
    legend_handles = [plt.Line2D([0], [0],marker='o', color=color, lw=2,linestyle='None') for color in colors]
    legend_labels = ['Train', 'Train+Augmented']
    plt.legend(legend_handles, legend_labels,fontsize='x-large')

    plt.ylabel('Test Loss (Mean Squared Error)',fontsize=15)
    #plt.ylabel('Test Loss (' + r'$1-2\cos(\theta)$'+')',fontsize=15)
    #plt.xlabel('Model Type',fontsize=15)
    #matplotlib.rc('ytick', labelsize=15) 
    #matplotlib.rc('xtick', labelsize=15) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    #plt.ylim(0.035, 0.08)

    plt.tight_layout()
    plt.savefig("autoencoder_boxplot.pdf")
    #plt.savefig("fine_tuning_boxplot.pdf")
    plt.close()
    
    # Calculate and print metrics, and perform t-tests
    calculate_metrics_and_t_tests(group1_182, group2_182, "Train_182", "Train+Augmented_182")
    calculate_metrics_and_t_tests(group1_22k, group2_22k, "Train_22k", "Train+Augmented_22k")
    # Performing t-test
    t_stat, p_value = ttest_ind(group1_182, group2_182)
    print(p_value,"182: org vs aug")
    t_stat, p_value = ttest_ind(group1_22k, group2_22k)
    print(p_value,"22k: org vs aug")
    t_stat, p_value = ttest_ind(group1_182, group1_22k)
    print(p_value,"182 vs 22k: org vs org")
    t_stat, p_value = ttest_ind(group2_182, group2_22k)
    print(p_value,"182 vs 22k: aug vs aug")


def main():
    opts, files = getopt.getopt(sys.argv[1:],"hvo:",["help"])
    if len(files) != 4:
        for file in files:
            print(len(files),"\n")
        usage()

        
    org_file_182 = files[0]
    aug_file_182 = files[1]
    org_file_22k = files[2]
    aug_file_22k = files[3]
    org_test_values_182 = []
    aug_test_values_182 = []
    org_test_values_22k = []
    aug_test_values_22k = []
    with open(org_file_182, 'r') as file:
        for line in file:
            match = re.search(r'test=([\d.]+)', line)
            #org_test_values_182.append(float(line))
            if match:
                org_test_values_182.append(float(match.group(1)))
    with open(aug_file_182, 'r') as file:
        for line in file:
            match = re.search(r'test=([\d.]+)', line)
            #aug_test_values_182.append(float(line))
            if match:
                aug_test_values_182.append(float(match.group(1)))
    with open(org_file_22k, 'r') as file:
        for line in file:
            match = re.search(r'test=([\d.]+)', line)
            #org_test_values_22k.append(float(line))
            if match:
                org_test_values_22k.append(float(match.group(1)))
    with open(aug_file_22k, 'r') as file:
        for line in file:
            match = re.search(r'test=([\d.]+)', line)
            #aug_test_values_22k.append(float(line))
            if match:
                aug_test_values_22k.append(float(match.group(1)))
    plotter(org_test_values_182,aug_test_values_182,org_test_values_22k,aug_test_values_22k)
if __name__ == "__main__":
    main()

