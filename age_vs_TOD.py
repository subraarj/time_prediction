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


def read_aug_phenotype_data(filename):
    lines = [line.strip().split() for line in open(filename, "r").readlines()]
    header = tuple(lines[0])
    lines = lines[1:]
    Age = []
    TOD = []
    All_age = []
    data = {}
    for line in lines:
        line = {label:value for label,value in zip(header,line)}
        #line["Age"] = int(line["Age"])
        All_age.append(float(line["Age"]))
        if line["TOD"] != "NA":
            y = float(line["TOD"])
            if y < 0:
                    y = y + 24
            TOD.append(y)
            Age.append(float(line["Age"]))
    return TOD, Age,All_age

def age_stats(age_data):
    # Calculate average age
    avg_age = np.mean(age_data)
    
    # Calculate median age
    median_age = np.median(age_data)
    
    # Calculate percentage of data age 50 and above
    num_above_50 = np.sum(np.array(age_data) >= 50)
    total_data = len(age_data)
    percentage_above_50 = (num_above_50 / total_data) * 100
    
    # Calculate standard deviation
    std_dev = np.std(age_data)
    
    # Calculate minimum and maximum age
    min_age = np.min(age_data)
    max_age = np.max(age_data)
    
    # Calculate interquartile range (IQR)
    Q1 = np.percentile(age_data, 25)
    Q3 = np.percentile(age_data, 75)
    IQR = Q3 - Q1
    
    # Calculate skewness
    skewness = np.mean((age_data - np.mean(age_data)) ** 3) / np.std(age_data) ** 3
    
    # Calculate kurtosis
    kurtosis = np.mean((age_data - np.mean(age_data)) ** 4) / np.std(age_data) ** 4
    
    # Print the results
    print("Average Age:", avg_age)
    print("Median Age:", median_age)
    print("Percentage of data age 50 and above:", percentage_above_50, "%")
    print("Standard Deviation:", std_dev)
    print("Minimum Age:", min_age)
    print("Maximum Age:", max_age)
    print("Interquartile Range (IQR):", IQR)
    print("Skewness:", skewness)
    print("Kurtosis:", kurtosis)

def plotter(TOD, age, All_age):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))  

    # Subplot 1: Histograms
    ax1 = axes[0]

    # Calculate bin edges starting from a multiple of 10 to cover the range of ages
    min_age = 10 * (min(age) // 10)
    max_age = 10 * (1 + max(age) // 10)
    num_bins = (max_age - min_age) // 10
    bin_edges = np.linspace(int(min_age), int(max_age), int(num_bins) + 1)

    ax1.hist(All_age, bins=bin_edges, color='#FF5733', edgecolor='black', alpha=0.7, label='All Individuals')
    ax1.hist(age, bins=bin_edges, color='#337FFF', edgecolor='black', alpha=0.7, label='Labeled Individuals')

    # Set x-axis tick positions to the center of bins and make them multiples of 10
    ax1.set_xticks((bin_edges[:-1] + bin_edges[1:]) / 2)
    ax1.set_xticklabels([f"{int(b)}" for b in bin_edges[:-1]], rotation=45, fontsize=12)
    ax1.set_yticklabels(np.arange(0, len(All_age)+1, 2), fontsize=12)  # Corrected this line
    ax1.set_xlabel("Age (Years)", fontsize=24)
    ax1.set_ylabel("Count", fontsize=24)
    ax1.legend(fontsize=12)

    # Subplot 2: Age vs TOD
    ax2 = axes[1]

    ax2.plot(age,TOD, marker='o', linestyle='', color='#337FFF', markersize=5)
    ax2.set_xlabel("Age (Years)", fontsize=24)
    ax2.set_ylabel("Time of Day (Hours)", fontsize=24)

    # Set a tight layout
    plt.tight_layout(pad=2)
    
    # Save the plot
    plt.savefig("TvA_subplots.pdf")
    
    # Show the plot
    plt.show()


def main():
    phen_file = "/nfs6/BB/Hendrix_Lab/Human/CircadianRhythms/Chen2016/GSE71620_Phenotype_GEO.txt"
    TOD,age,All_age = read_aug_phenotype_data(phen_file)
    plotter(TOD,age,All_age)
    print("------------TOD--------------")
    age_stats(age)
    print("------------ALL--------------")
    age_stats(All_age)
if __name__ == "__main__":
    main()