#####################################################################################
### Script uses QuTIP library to generate random quantum states.
### PPT Criteria is used to characterize entanglement in each 2x2 state.
### Entanglement witness is then one-hot encoded.
#####################################################################################

from qutip import *
import numpy as np
import csv

rho_array = []
witness = []

for i in range(2500):
	rho = rand_dm(4, pure = 'True', dims=[[2,2], [2,2]])
	rho_pt = partial_transpose(rho,[1,0])
	
	output_rho = operator_to_vector(rho).full()
	rho_array.append(output_rho.T)

	# PPT Criteria
	if any(eigenvalue < 0 for eigenvalue in rho_pt.eigenenergies()):
		witness.append([1,0])
	else:
		witness.append([0,1])

for i in range(2500):
	rho = rand_dm(4, pure = 'False', dims = [[2,2], [2,2]])
	rho_pt = partial_transpose(rho,[1,0])

	output_rho = operator_to_vector(rho).full()
	rho_array.append(output_rho.T)

	# PPT Criteria
	if any(eigenvalue<0 for eigenvalue in rho_pt.eigenenergies()):
		witness.append([1,0])
	else:
		witness.append([0,1])

output_data = np.vstack(rho_array)
file_data_store('features.csv', output_data, numtype = 'complex', sep = ',')
np.savetxt('labels.txt',np.c_[witness], fmt = '%d', delimiter = ',', header = 'witness', comments = '#')