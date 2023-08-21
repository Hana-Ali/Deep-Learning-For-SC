import numpy as np
from .general_helpers import check_type
import scipy.signal as signal
import scipy.stats as stats
from .sc_fc_helpers import process_BOLD


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