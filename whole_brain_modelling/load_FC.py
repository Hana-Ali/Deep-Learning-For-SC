import os
import sys
import numpy as np

path = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\whole_brain_modelling\\temp_arrays"
FC_path = os.path.join(path, "avg_emp_FC.csv")

FC = np.loadtxt(FC_path, delimiter = ',')
print(FC.shape)