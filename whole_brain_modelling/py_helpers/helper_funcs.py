from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import os

# Helpers

# Check type of array and raise error if wrong
def check_type(inp, input_type, input_name):
    if not isinstance(inp, input_type):
        raise TypeError('The input ' + input_name + ' must be a ' + input_type.__name__ + ', is a ' + type(inp).__name__)

# Check shape of array and raise error if wrong
def check_shape(inp, input_shape, input_name):
    if not inp.shape == input_shape:
        raise ValueError('The input ' + input_name + ' must have shape ' + str(input_shape) + ', has shape ' + str(inp.shape))

def check_all_types(inputs):
    for (inputs, input_type, input_name) in inputs:
        check_type(inputs, input_type, input_name)

def check_all_shapes(inputs):
    for (inputs, input_shape, input_name) in inputs:
        check_shape(inputs, input_shape, input_name)

# Get the Structural Connectivity Matrices
def get_empirical_SC(path):
    
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
    fig = plt.figure(figsize=(6, 7))
    fig.suptitle('Structural Connectivity', fontsize=20)
    plt.imshow(SC_final, interpolation='nearest', aspect='equal', cmap='jet')
    cb = plt.colorbar(shrink=0.2)
    cb.set_label('Weights', fontsize=14)

    return SC_final

# Define function for processing the BOLD signals
def process_BOLD(BOLD_signal):

    # Define the butter bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order=5):
        return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
    # Use the butter bandpass filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    # Define the parameters for the filter
    fs = 1 / 0.7
    lowcut = 0.01 / (fs / 2)
    highcut = 0.1 / (fs / 2)

    BOLD_mean = np.mean(BOLD_signal, axis=0)
    BOLD_mean = np.expand_dims(BOLD_mean, axis=0)
    ones_matrix = np.ones((BOLD_mean.shape[0], 1))
    BOLD_mean = ones_matrix @ BOLD_mean
    BOLD_regressed = BOLD_signal - BOLD_mean
    BOLD_butter = butter_bandpass_filter(BOLD_regressed, lowcut, highcut, fs, order=6)
    BOLD_z_score = stats.zscore(BOLD_butter)

    return BOLD_z_score

# Get the Functional Connectivity Matrices
def get_empirical_FC(path):
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
        bold_z_score = process_BOLD(subject)
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