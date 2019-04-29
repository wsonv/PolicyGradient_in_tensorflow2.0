import os
import numpy as np

class Logger():
	def __init__(self):
		self.mean = np.array([])
		self.max = np.array([])
		self.min = np.array([])
		self.std = np.array([])
		self.is_first_row = True
		self.header = []
	def log(self, what, value):
		if what == "max":
			self.max = np.append(self.max, value)
		elif what == "avg":
			self.mean = np.append(self.mean, value)
		elif what == "min":
			self.min = np.append(self.min, value)
		elif what == "std":
			self.std = np.append(self.std, value)

def initiate(exp_name, seed):
	saving_dir = os.path.join('data', exp_name, str(seed))
	if os.path.exists(saving_dir):
		raise ValueError("directory already exists")
	global logger
	logger = Logger()

def log(statement, value):
	print(statement," ", value)
	logger.log(statement[:3], value)


def finished(exp_name, header, seed):
	logger.header = header
	saving_dir = os.path.join('data', exp_name, str(seed))
	#making dir in finished() rather than initiate() for the convenience of debugging of original file. 
	#(error in main file after making dir in initiate() would be not comfortable because you need to 
	# change your directory name in terminal execution command everytime you execute)
	os.makedirs(saving_dir)
	with open(os.path.join(saving_dir, "log.txt"),'w') as f:
		if logger.is_first_row:
			f.write('\t'.join(logger.header))
			f.write('\n')
		for i in range(len(logger.mean)):
			f.write('\t'.join(map(str, [logger.mean[i], logger.std[i], logger.max[i], logger.min[i]])))
			f.write('\n')
