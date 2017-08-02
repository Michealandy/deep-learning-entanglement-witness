import numpy as np
import csv

def csv_to_array(filename, type_):
	with open(filename, 'r') as f:
		next(f)
		reader = csv.reader(f)
		list_ = list(reader)
	f.close()
	
	return np.array(list_).astype(type_)