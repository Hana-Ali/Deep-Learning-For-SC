from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from .general_helpers import *
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import json
import os

import mat73

# Get the Structural Connectivity Matrices
def get_empirical_SC(path, HCP=False):

    # The SC is different depending on path
    if HCP:
        SC_array = get_empirical_SC_HCP(path)
    else:
        SC_array = get_empirical_SC_CAMCAN(path)

    # Return the SC array
    return SC_array


# Get the Functional Connectivity Matrices
def get_empirical_FC(path, config_path, HCP=False):
    
    # The FC is different depending on path
    if HCP:
        FC_array = get_empirical_FC_HCP(path, config_path)
    else:
        FC_array = get_empirical_FC_CAMCAN(path, config_path)

    # Return the FC array
    return FC_array

# Get the Structural Connectivity Matrices
def get_empirical_SC_HCP(path):
    
    # Define paths, load matrices, and stack into array shape
    SC_path = os.path.join(path, 'Schaefer100_DTI_HCP.mat')
    SC_all = sio.loadmat(SC_path)
    SC_all = np.array(SC_all['DTI_fibers_HCP'])
    SC_all = np.concatenate(SC_all, axis=0)
    SC_all = np.array([subject for subject in SC_all])

    # Consensus averaging
    consensus = 0.5
    SC_consensus = []
    elements = []
    for i in range(0, SC_all.shape[0]):
        for j in range(0, SC_all.shape[1]):
            elements = []
            for k in range(0, SC_all.shape[2]):
                elements.append(SC_all[k][j][i])
                nonZerosCount = np.count_nonzero(elements)
                nonZerosPercent = nonZerosCount / SC_all.shape[2]
            if (nonZerosPercent >= consensus):
                meanValue = np.mean(elements)
                SC_consensus.append(meanValue)
            else:
                SC_consensus.append(0)
    SC_consensus = np.array(SC_consensus)
    SC_consensus = SC_consensus[..., np.newaxis]
    SC_consensus = np.reshape(SC_consensus, (100,100))

    # Filtering outliers and plotting
    SC_consensus = np.reshape(SC_consensus, (-1,1))
    mean = np.mean(SC_consensus)
    std_dev = np.std(SC_consensus)
    threshold = 3 
    outliers = SC_consensus[np.abs(SC_consensus - mean) > threshold * std_dev]
    for idx, element in enumerate(SC_consensus):
        if element in outliers:
            SC_consensus[idx] = threshold * std_dev
    SC_consensus = np.reshape(SC_consensus, (100, 100))
    scaler = MinMaxScaler()
    SC_final = scaler.fit_transform(SC_consensus)

    return SC_final

# Function to get empirical SC - CAMCAN
def get_empirical_SC_CAMCAN(SUBJECT_SC_PATH):

    # Grab all the csv files in the directory
    csv_files = glob_files(SUBJECT_SC_PATH, 'csv')

    # Randomly choose one of the csv files
    csv_file = np.random.choice(csv_files)

    # Read the csv file
    csv_data = np.loadtxt(csv_file, delimiter=",")

    # Get whether it's prob or global SC
    if "prob" in csv_file:
        SC_type = "prob"
    elif "global" in csv_file:
        SC_type = "global"
    elif "det" in csv_file:
        SC_type = "det"
    else:
        raise ValueError("The csv file does not contain a valid SC type")

    # Return the csv data
    return (csv_data, SC_type)

# Get the Functional Connectivity Matrices
def get_empirical_FC_HCP(path, config_path):

    # Check that config path exists
    if not os.path.exists(config_path):
        raise ValueError('The input config_path does not exist')
    
    # Read the JSON file
    with open(config_path) as json_file:
        config = json.load(json_file)

    order = config['order']
    TR = config['TR']
    cutoffLow = config['cutoffLow']
    cutoffHigh = config['cutoffHigh']

    # Define paths, load matrices, and stack into array shape
    FC_path = os.path.join(path, 'Schaefer100_BOLD_HCP.mat')
    FC_all = sio.loadmat(FC_path)
    FC_all = np.array(FC_all['BOLD_timeseries_HCP'])
    FC_all = np.concatenate(FC_all, axis=0)
    FC_all = np.array([subject for subject in FC_all])

    # Correlation matrix
    FC_corr = []
    
    # Process the BOLD signal of every subject, and get correlation
    for subject in FC_all:
        bold_z_score = process_BOLD(subject, order, TR, cutoffLow, cutoffHigh)
        correlation = np.corrcoef(bold_z_score)
        FC_corr.append(correlation)

    # Average the correlation matrices
    FC_corr = np.array(FC_corr)
    FC_final = np.mean(FC_corr, axis=0)   
    # Remove the diagonal 
    np.fill_diagonal(FC_final, 0.0)
    
    # Plot the results
    fig = plt.figure(figsize=(6, 7))
    fig.suptitle('Functional Connectivity', fontsize=20)
    plt.imshow(FC_final, interpolation='nearest', aspect='equal', cmap='jet')
    cb = plt.colorbar(shrink=0.2)
    cb.set_label('Weights', fontsize=14)

    return FC_final

# Function to get empirical FC - CAMCAN
def get_empirical_FC_CAMCAN(SUBJECT_FC_PATH, config_path):

    # Check that config path exists
    if not os.path.exists(config_path):
        raise ValueError('The input config_path does not exist')
    
    # Read the JSON file
    with open(config_path) as json_file:
        config = json.load(json_file)

    # Get needed parameters from the JSON file
    order = config['order']
    TR = config['TR']
    cutoffLow = config['cutoffLow']
    cutoffHigh = config['cutoffHigh']

    # Get the mat file from the FC path
    mat_file = glob_files(SUBJECT_FC_PATH, 'mat')[0]

    # Load the mat file
    mat_data = mat73.loadmat(mat_file)

    # Print the keys of the mat file
    BOLD_data = mat_data['Data']

    # Process the BOLD data and find correlation matrix
    bold_z_score = process_BOLD(BOLD_data, order, TR, cutoffLow, cutoffHigh)
    FC_matrix = np.corrcoef(bold_z_score)

    # Remove the diagonal
    np.fill_diagonal(FC_matrix, 0.0)

    # Return the matrix
    return FC_matrix

# Function to get the BOLD signals
def get_empirical_BOLD(path):
    # Define paths, load matrices, and stack into array shape
    BOLD_path = os.path.join(path, 'Schaefer100_BOLD_HCP.mat')
    BOLD_all = sio.loadmat(BOLD_path)
    BOLD_all = np.array(BOLD_all['BOLD_timeseries_HCP'])
    BOLD_all = np.concatenate(BOLD_all, axis=0)
    BOLD_all = np.array([subject for subject in BOLD_all])

    # Return unprocessed BOLD
    return BOLD_all

# Define function for processing the BOLD signals
def process_BOLD(BOLD_signal, order, TR, cutoffLow, cutoffHigh):

    # Define the butter bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order=5):
        return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
    # Use the butter bandpass filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    # Define the parameters for the filter
    fs = 1 / TR
    lowcut = cutoffLow / (fs / 2)
    highcut = cutoffHigh / (fs / 2)

    BOLD_mean = np.mean(BOLD_signal, axis=0)
    BOLD_mean = np.expand_dims(BOLD_mean, axis=0)
    ones_matrix = np.ones((BOLD_mean.shape[0], 1))
    BOLD_mean = ones_matrix @ BOLD_mean
    BOLD_regressed = BOLD_signal - BOLD_mean
    BOLD_butter = butter_bandpass_filter(BOLD_regressed, lowcut, highcut, fs, order)
    BOLD_z_score = stats.zscore(BOLD_butter)

    return BOLD_z_score

# Determine the R order parameter
def determine_order_R(BOLD_signal, number_of_parcels, start_index):
    """"
    This function determines the order parameter R of the data, which is a measure of the
    synchronization of the data. It is defined as the mean of the absolute values of the
    complex phases of the data.

    Parameters
    ----------
    BOLD_signal : numpy array
        The BOLD signal data, with shape (number of oscillators, number of time points)
    number_of_parcels : int
        The number of parcels to use with the data
    start_index : int
        The index at which to start the analysis
        
    Returns
    -------
    R_mean : float
        The mean of the order parameter R of the data
    R_std : float
        The standard deviation of the order parameter R of the data

    """

    # --------- Check that the input arguments are of the correct type
    check_type(BOLD_signal, np.ndarray, 'BOLD_signal')
    check_type(number_of_parcels, int, 'number_of_parcels')
    print('BOLD_signal', BOLD_signal)

    # --------- Check that the input arguments are of the correct shape
    if not BOLD_signal.shape[0] == number_of_parcels:
        raise ValueError('The input BOLD_signal must have shape (number of oscillators, number of time points), has shape ' + str(BOLD_signal.shape))

    # --------- Calculate the order parameter R
    # Process the simulated BOLD in the same way the empirical is processed
    BOLD_processed = process_BOLD(BOLD_signal)

    # Apply the Hilbert transform to the data
    BOLD_hilbert = signal.hilbert(BOLD_processed)
    phase = np.angle(BOLD_hilbert)
    phase = phase[:, start_index:]

    # Calculate the complex phases of the data
    complex_phase = np.exp(1j * phase)

    # Calculate the order parameter R
    R = np.mean(np.abs(complex_phase), axis=0)

    # Calculate the mean and standard deviation of the order parameter R
    R_mean = np.mean(R)
    R_std = np.std(R, ddof=1)

    return float(R_mean), float(R_std)


def determine_similarity(empFC, simFC, technique="Pearson"):
    """
    This function determines the similarity between the empirical and simulated FC matrices.
    Different similarity measures can be used, including Pearson correlation, Spearman
    correlation, and the Euclidean distance. Others should be researched first

    Parameters
    ----------
    empFC : numpy array
        The empirical FC matrix, with shape (number of oscillators, number of oscillators)
    simFC : numpy array
        The simulated FC matrix, with shape (number of oscillators, number of oscillators)
    technique : str
        The technique to use to determine the similarity. Currently supported are "Pearson",
        "Spearman", and "Euclidean"

    Returns
    -------
    similarity : float
        The similarity between the empirical and simulated FC matrices
    """

    # --------- Check that the input arguments are of the correct type
    check_type(empFC, np.ndarray, 'empFC')
    check_type(simFC, np.ndarray, 'simFC')
    check_type(technique, str, 'technique')

    # --------- Check that the input arguments are of the correct shape
    if not empFC.shape == simFC.shape:
        raise ValueError('The input simFC and empFC must have shape (number of oscillators, number of oscillators), empFC has shape ' + str(empFC.shape) + ', simFC has shape ' + str(simFC.shape))
    
    # --------- Determine the similarity
    if technique == "Pearson":
        similarity = stats.pearsonr(empFC.flatten(), simFC.flatten())[0]
    elif technique == "Spearman":
        similarity = stats.spearmanr(empFC.flatten(), simFC.flatten())[0]
    elif technique == "Euclidean":
        similarity = np.linalg.norm(empFC - simFC)
    else:
        raise ValueError('The input technique must be "Pearson", "Spearman", or "Euclidean", is ' + technique)

    return float(similarity)