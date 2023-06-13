// -------------------- Header file with some helper for the main simulation

#define _USE_MATH_DEFINES

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <boost/any.hpp>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

std::vector<double> ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff);
std::vector<double> TrinomialMultiply(int FilterOrder, std::vector<double> b, std::vector<double> c);
std::vector<double> ComputeNumCoeffs(int FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> DenC);
std::vector<double> ComputeLP(int FilterOrder);
std::vector<double> ComputeHP(int FilterOrder);
std::vector<double> filter(std::vector<double> denom_coeff, std::vector<double> numer_coeff, int number_samples, 
                            std::vector<double> original_signal, std::vector<double> filtered_signal);
std::vector<std::vector<double>> process_BOLD(std::vector<std::vector<double>> BOLD_signal, int num_rows, int num_columns, int order, 
							double samplingFrequency, double cutoffFrequencyLow, double cutoffFrequencyHigh);
std::vector<std::vector<double>> determine_FC(std::vector<std::vector<double>> BOLD_signal);
double pearsoncoeff(std::vector<double> X, std::vector<double> Y);
std::vector<std::vector<double>> determine_FC_nogsl(std::vector<std::vector<double>> BOLD_signal);
std::vector<std::vector<double>> get_emp_FC(std::string &file_path);

// Function to check the data type of a variable
void check_type(boost::any input, const std::string& expected_type, const std::string& input_name) {

    // Get the type of the variable
    std::string true_type = input.type().name();
    printf("The type of %s is %s\n", input_name.c_str(), true_type.c_str());

    // Check if the type is the expected one
    if (true_type != expected_type) {
        throw std::invalid_argument("The type of " + input_name + " is " + true_type + " but it should be " + expected_type);
    }
    else {
        printf("The type of %s is correct\n", input_name.c_str());
    }
}

// Function to check the shape of a variable
// void check_shape(boost::any input, std::vector<int>& expected_shape, std::string& input_name) {

//     // Get the shape of the variable
//     std::vector<int> true_shape = input.shape();

//     // Check if the shape is the expected one
//     if (true_shape != expected_shape) {
//         throw std::invalid_argument("The shape of " + input_name + " is " + std::to_string(true_shape[0]) + "x" + std::to_string(true_shape[1]) + " but it should be " + std::to_string(expected_shape[0]) + "x" + std::to_string(expected_shape[1]));
//     }
// }

#define N 10 //The number of images which construct a time series for each pixel
#define PI 3.1415926535897932384626433832795
std::vector<double> ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff)
{
	int k;            // loop variables
	double theta;     // PI * (Ucutoff - Lcutoff) / 2.0
	double cp;        // cosine of phi
	double st;        // sine of theta
	double ct;        // cosine of theta
	double s2t;       // sine of 2*theta
	double c2t;       // cosine 0f 2*theta
	std::vector<double> RCoeffs(2 * FilterOrder);     // z^-2 coefficients 
	std::vector<double> TCoeffs(2 * FilterOrder);     // z^-1 coefficients
	std::vector<double> DenomCoeffs;     // dk coefficients
	double PoleAngle;      // pole angle
	double SinPoleAngle;     // sine of pole angle
	double CosPoleAngle;     // cosine of pole angle
	double a;         // workspace variables

	cp = cos(PI * (Ucutoff + Lcutoff) / 2.0);
	theta = PI * (Ucutoff - Lcutoff) / 2.0;
	st = sin(theta);
	ct = cos(theta);
	s2t = 2.0*st*ct;        // sine of 2*theta
	c2t = 2.0*ct*ct - 1.0;  // cosine of 2*theta

	for (k = 0; k < FilterOrder; ++k)
	{
		PoleAngle = PI * (double)(2 * k + 1) / (double)(2 * FilterOrder);
		SinPoleAngle = sin(PoleAngle);
		CosPoleAngle = cos(PoleAngle);
		a = 1.0 + s2t*SinPoleAngle;
		RCoeffs[2 * k] = c2t / a;
		RCoeffs[2 * k + 1] = s2t*CosPoleAngle / a;
		TCoeffs[2 * k] = -2.0*cp*(ct + st*SinPoleAngle) / a;
		TCoeffs[2 * k + 1] = -2.0*cp*st*CosPoleAngle / a;
	}

	DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs);

	DenomCoeffs[1] = DenomCoeffs[0];
	DenomCoeffs[0] = 1.0;
	for (k = 3; k <= 2 * FilterOrder; ++k)
		DenomCoeffs[k] = DenomCoeffs[2 * k - 2];

	for (int i = DenomCoeffs.size() - 1; i > FilterOrder * 2 + 1; i--)
		DenomCoeffs.pop_back();

	return DenomCoeffs;
}

std::vector<double> TrinomialMultiply(int FilterOrder, std::vector<double> b, std::vector<double> c)
{
	int i, j;
	std::vector<double> RetVal(4 * FilterOrder);

	RetVal[2] = c[0];
	RetVal[3] = c[1];
	RetVal[0] = b[0];
	RetVal[1] = b[1];

	for (i = 1; i < FilterOrder; ++i)
	{
		RetVal[2 * (2 * i + 1)] += c[2 * i] * RetVal[2 * (2 * i - 1)] - c[2 * i + 1] * RetVal[2 * (2 * i - 1) + 1];
		RetVal[2 * (2 * i + 1) + 1] += c[2 * i] * RetVal[2 * (2 * i - 1) + 1] + c[2 * i + 1] * RetVal[2 * (2 * i - 1)];

		for (j = 2 * i; j > 1; --j)
		{
			RetVal[2 * j] += b[2 * i] * RetVal[2 * (j - 1)] - b[2 * i + 1] * RetVal[2 * (j - 1) + 1] +
				c[2 * i] * RetVal[2 * (j - 2)] - c[2 * i + 1] * RetVal[2 * (j - 2) + 1];
			RetVal[2 * j + 1] += b[2 * i] * RetVal[2 * (j - 1) + 1] + b[2 * i + 1] * RetVal[2 * (j - 1)] +
				c[2 * i] * RetVal[2 * (j - 2) + 1] + c[2 * i + 1] * RetVal[2 * (j - 2)];
		}

		RetVal[2] += b[2 * i] * RetVal[0] - b[2 * i + 1] * RetVal[1] + c[2 * i];
		RetVal[3] += b[2 * i] * RetVal[1] + b[2 * i + 1] * RetVal[0] + c[2 * i + 1];
		RetVal[0] += b[2 * i];
		RetVal[1] += b[2 * i + 1];
	}

	return RetVal;
}

std::vector<double> ComputeNumCoeffs(int FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> DenC)
{
	std::vector<double> TCoeffs;
	std::vector<double> NumCoeffs(2 * FilterOrder + 1);
	std::vector<std::complex<double>> NormalizedKernel(2 * FilterOrder + 1);

	std::vector<double> Numbers;
	for (double n = 0; n < FilterOrder * 2 + 1; n++)
		Numbers.push_back(n);
	int i;

	TCoeffs = ComputeHP(FilterOrder);

	for (i = 0; i < FilterOrder; ++i)
	{
		NumCoeffs[2 * i] = TCoeffs[i];
		NumCoeffs[2 * i + 1] = 0.0;
	}
	NumCoeffs[2 * FilterOrder] = TCoeffs[FilterOrder];

	double cp[2];
	double Bw, Wn;
	cp[0] = 2 * 2.0*tan(PI * Lcutoff / 2.0);
	cp[1] = 2 * 2.0*tan(PI * Ucutoff / 2.0);

	Bw = cp[1] - cp[0];
	//center frequency
	Wn = sqrt(cp[0] * cp[1]);
	Wn = 2 * atan2(Wn, 4);
	double kern;
	const std::complex<double> result = std::complex<double>(-1, 0);

	for (int k = 0; k< FilterOrder * 2 + 1; k++)
	{
		NormalizedKernel[k] = std::exp(-sqrt(result)*Wn*Numbers[k]);
	}
	double b = 0;
	double den = 0;
	for (int d = 0; d < FilterOrder * 2 + 1; d++)
	{
		b += real(NormalizedKernel[d] * NumCoeffs[d]);
		den += real(NormalizedKernel[d] * DenC[d]);
	}
	for (int c = 0; c < FilterOrder * 2 + 1; c++)
	{
		NumCoeffs[c] = (NumCoeffs[c] * den) / b;
	}

	for (int i = NumCoeffs.size() - 1; i > FilterOrder * 2 + 1; i--)
		NumCoeffs.pop_back();

	return NumCoeffs;
}

std::vector<double> ComputeLP(int FilterOrder)
{
	std::vector<double> NumCoeffs(FilterOrder + 1);
	int m;
	int i;

	NumCoeffs[0] = 1;
	NumCoeffs[1] = FilterOrder;
	m = FilterOrder / 2;
	for (i = 2; i <= m; ++i)
	{
		NumCoeffs[i] = (double)(FilterOrder - i + 1)*NumCoeffs[i - 1] / i;
		NumCoeffs[FilterOrder - i] = NumCoeffs[i];
	}
	NumCoeffs[FilterOrder - 1] = FilterOrder;
	NumCoeffs[FilterOrder] = 1;

	return NumCoeffs;
}

std::vector<double> ComputeHP(int FilterOrder)
{
	std::vector<double> NumCoeffs;
	int i;

	NumCoeffs = ComputeLP(FilterOrder);

	for (i = 0; i <= FilterOrder; ++i)
		if (i % 2) NumCoeffs[i] = -NumCoeffs[i];

	return NumCoeffs;
}

std::vector<double> filter(std::vector<double> numer_coeff, std::vector<double> denom_coeff, int number_samples, 
                            std::vector<double> original_signal, std::vector<double> filtered_signal)
{
    int len_x = original_signal.size();
	int len_b = numer_coeff.size();
	int len_a = denom_coeff.size();

	std::vector<double> zi(len_b);

	if (len_a == 1)
	{
		for (int m = 0; m < len_x; m++)
		{
			filtered_signal[m] = numer_coeff[0] * original_signal[m] + zi[0];
			for (int i = 1; i < len_b; i++)
			{
				zi[i - 1] = numer_coeff[i] * original_signal[m] + zi[i];//-coeff_a[i]*filter_x[m];
			}
		}
	}
	else
	{
		for (int m = 0; m<len_x; m++)
		{
			filtered_signal[m] = numer_coeff[0] * original_signal[m] + zi[0];
			for (int i = 1; i < len_b; i++)
			{
				zi[i - 1] = numer_coeff[i] * original_signal[m] + zi[i] - numer_coeff[i] * filtered_signal[m];
			}
		}
	}

    return filtered_signal;
}

// Function to process the BOLD data - same as in Python helper_funcs.py file
std::vector<std::vector<double>> process_BOLD(std::vector<std::vector<double>> BOLD_signal, int num_rows, int num_columns, int order, 
                                                double samplingFrequency, double cutoffFrequencyLow, double cutoffFrequencyHigh)
{   
    // Create the filtered signal object
    std::vector<std::vector<double>> filteredSignal(num_rows, std::vector<double>(num_columns));

    // Create filter objects
    // These values are as a ratio of f/fs, where fs is sampling rate, and f is cutoff frequency
    double FrequencyBands[2] = {
        cutoffFrequencyLow/(samplingFrequency*2.0),
        cutoffFrequencyHigh/(samplingFrequency*2.0)
    };
    //Create the variables for the numerator and denominator coefficients
    std::vector<double> DenC;
    std::vector<double> NumC;

    // Find the mean across the columns
    printf("Finding the mean across the columns\n");
    std::vector<double> mean(num_columns);
    // Calculate the mean across the columns
    for (int row = 0; row < num_rows; row++) {
        double colSum = 0.0;
        for (int col = 0; col < num_columns; col++) {
            colSum += BOLD_signal[row][col];
            // printf("column %d, colSum is: %lf\n", col, colSum);
        }
        printf("Row %i, column sum is %f\n", row, colSum);
        mean[row] = colSum / num_rows;
    }

    // Remove the mean from each column
    printf("Removing the mean from each column\n");
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_columns; col++) {
            BOLD_signal[row][col] -= mean[col];
        }
    }

    // Finding the coefficients of the filter
    printf("Finding the coefficients of the filter\n");
    DenC = ComputeDenCoeffs(order, FrequencyBands[0], FrequencyBands[1]);
    for(int k = 0; k<2*order+1; k++)
        printf("DenC is: %lf\n", DenC[k]);

    NumC = ComputeNumCoeffs(order,FrequencyBands[0],FrequencyBands[1],DenC);
    for(int k = 0; k<2*order+1; k++)
        printf("NumC is: %lf\n", NumC[k]);

	// Printing sizes of both vectors
	printf("BOLD_signal size is %d x %d\n", BOLD_signal.size(), BOLD_signal[0].size());
	printf("filteredSignal size is %d x %d\n", filteredSignal.size(), filteredSignal[0].size());

    // Applying the filter forwards and backwards
    printf("Applying the filter forwards and backwards\n");
    for (int row = 0; row < num_rows; row++)
        filteredSignal[row] = filter(NumC, DenC, num_columns, BOLD_signal[row], filteredSignal[row]);
    
    for (int row = num_rows - 1; row >= 0; row--)
        filteredSignal[row] = filter(NumC, DenC, num_columns, BOLD_signal[row], filteredSignal[row]);

	// Z-scoring the final filtered signal
	printf("Z-scoring the final filtered signal\n");
	for (int row = 0; row < num_rows; row++) {
		double mean = 0.0;
		double std = 0.0;
		for (int col = 0; col < num_columns; col++) {
			mean += filteredSignal[row][col];
		}
		mean /= num_columns;
		for (int col = 0; col < num_columns; col++) {
			std += pow(filteredSignal[row][col] - mean, 2);
		}
		std /= num_columns;
		std = sqrt(std);
		for (int col = 0; col < num_columns; col++) {
			filteredSignal[row][col] = (filteredSignal[row][col] - mean) / std;
		}
	}

    return filteredSignal;
}

// Function to find the functional connectivity matrix from the BOLD signal
std::vector<std::vector<double>> determine_FC(std::vector<std::vector<double>> BOLD_signal)
{
	// Create a correlation matrix of size BOLD_signal.size() x BOLD_signal.size()
	std::vector<std::vector<double>> correlation_matrix(BOLD_signal.size(), std::vector<double>(BOLD_signal.size()));

	// For every row (brain region) of the BOLD_signal
	for (int i = 0; i < BOLD_signal.size(); i++)
	{
		// For every other row (brain region) of the BOLD_signal
		for (int j = 0; j < BOLD_signal.size(); j++)
		{
			// // Convert the signal vectors to the appropriate format for gsl
			// gsl_vector_const_view gsl_BOLD_i = gsl_vector_const_view_array(BOLD_signal[i].data(), BOLD_signal[i].size());
			// gsl_vector_const_view gsl_BOLD_j = gsl_vector_const_view_array(BOLD_signal[j].data(), BOLD_signal[j].size());
			double correlation = gsl_stats_correlation(BOLD_signal[i].data(), 1, BOLD_signal[j].data(), 1, BOLD_signal[i].size());
			correlation_matrix[i][j] = correlation;
			printf("Correlation between %d and %d is %lf\n", i, j, correlation);
		}
	}

	return correlation_matrix;
}

double pearsoncoeff(std::vector<double> X, std::vector<double> Y)
{
	double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;
 
    for (int i = 0; i < X.size(); i++)
    {
        // sum of elements of array X.
        sum_X += X[i];
 
        // sum of elements of array Y.
        sum_Y += Y[i];
 
        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];
 
        // sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i];
        squareSum_Y = squareSum_Y + Y[i] * Y[i];
    }
 
    // use formula for calculating correlation coefficient.
    float corr = (float)(X.size() * sum_XY - sum_X * sum_Y)
                  / sqrt((X.size() * squareSum_X - sum_X * sum_X)
                      * (X.size() * squareSum_Y - sum_Y * sum_Y));
 
    return corr;
}

// Function to find the functional connectivity matrix from the BOLD signal
std::vector<std::vector<double>> determine_FC_nogsl(std::vector<std::vector<double>> BOLD_signal)
{
	// Create a correlation matrix of size BOLD_signal.size() x BOLD_signal.size()
	std::vector<std::vector<double>> correlation_matrix(BOLD_signal.size(), std::vector<double>(BOLD_signal.size()));

	// For every row (brain region) of the BOLD_signal
	for (int i = 0; i < BOLD_signal.size(); i++)
	{
		// For every other row (brain region) of the BOLD_signal
		for (int j = 0; j < BOLD_signal.size(); j++)
		{
			double correlation = pearsoncoeff(BOLD_signal[i], BOLD_signal[j]);
			correlation_matrix[i][j] = correlation;
			printf("Correlation between %d and %d is %lf\n", i, j, correlation);
		}
	}

	return correlation_matrix;
}

// Function to find get the functional connectivity from input files
std::vector<std::vector<double>> get_emp_FC(std::string &file_path)
{
	// Open the file given in the path
	std::ifstream file(file_path);
	if (!file.is_open()) {
		throw std::invalid_argument("The file " + file_path + " could not be opened");
	}
	else {
		printf("The file %s was opened successfully\n", file_path.c_str());
	}

	// Create a vector to store the data
	std::vector<std::vector<double>> data;

	// Read the data from the file
	std::string line;
	while (std::getline(file, line)) {
		// Create a vector to store the data in the line
		std::vector<double> line_data;

		// Create a stringstream of the line
		std::stringstream ss(line);

		// Read the data from the stringstream
		double value;
		while (ss >> value) {
			line_data.push_back(value);
		}

		// Add the line data to the data vector
		data.push_back(line_data);
	}
}

