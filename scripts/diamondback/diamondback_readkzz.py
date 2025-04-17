# %%
import numpy as np
# %%
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

# %%
all_lines = readInFile("../../data/diamondback_allmodels/t900g316f3_m0.0_co1.0.out")
for (i, line) in enumerate(all_lines):
    if "kz(cm^2/s)" in line:
        pr = np.array([float(x[1:9]) for x in all_lines[i+1:i+91]])
        tm = np.array([float(x[20:25]) for x in all_lines[i+1:i+91]])
        kz = np.array([float(x[27:34]) for x in all_lines[i+1:i+91]])
        break
# %%
