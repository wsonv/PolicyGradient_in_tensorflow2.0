import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import re

class Data():
	def __init__(self, ):
		self.mean = []
		self.max = []
		self.min = []
		self.std = []
		self.is_header = True
		self.header_num = 4
		self.headers = [self.mean, self.std, self.max, self.min]
		#header name must be in the same order with self.headers elements
		self.header_names = ['Average reward', 'STD of reward', 'Max reward', 'Min reward']
parser = argparse.ArgumentParser()
parser.add_argument("location", type = str)
parser.add_argument("-f", "--full", action = "store_true")

args = parser.parse_args()

p = re.compile('\S+')

data_dir = args.location
Datas = []
def collect_data():
	if not os.path.exists(data_dir):
		raise ValueError("No such directory")
	for root, dirs, files in os.walk(data_dir):
		for file in files:
			if file == 'log.txt':
				data_object = Data()
				Datas.append(data_object)
				#if there is no header in files, set self.is_header as False in Data class
				with open(os.path.join(root,file), 'r') as f:
					if data_object.is_header:
						f.readline()
					text = f.read()
					processed_text = np.array(p.findall(text)).astype(np.float32)
					for i, header in enumerate(data_object.headers):
						header.extend(processed_text[[j * data_object.header_num + i  \
							for j in range(len(processed_text)//data_object.header_num)]])
					
def plot():
	#figure arrangement (horizontal, vertical)

	axes = []
	if args.full:
		fig = plt.figure(figsize = (9, 9))
		plot_num = Datas[0].header_num
		fig_size1 = 2
		fig_size2 = 2
	else:
		fig = plt.figure(figsize = (9, 4))
		plot_num = 2
		fig_size1 = 1
		fig_size2 = 2
	for i in range(plot_num):
		axes.append(fig.add_subplot(fig_size1, fig_size2, i + 1))
		axes[i].set_xlabel("Iteration")
		axes[i].set_ylabel(Datas[0].header_names[i])
	for i, data in enumerate(Datas):
		for j, header in enumerate(data.headers[:plot_num]):
			axes[j].plot(header, label = "Exp num {}".format(i + 1))
			axes[j].legend()

	plt.show()

def main():
	collect_data()
	plot()

if __name__ == "__main__":
	main()
				