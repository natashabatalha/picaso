import numpy as np

diamondback_datapath = "/Users/adityasengupta/projects/clouds/picaso/data"

def readInFile(filename):
	f = open(filename)
	line = f.readline()
	filelines = []
	while line != "":
		filelines.append(line)
		try: 
			line = f.readline()
		except UnicodeDecodeError: 
			line='xxx xxx'
	return filelines
