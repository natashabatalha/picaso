import numpy as np
from matplotlib import pyplot as plt
from os import path
import re
from scripts.diamondback.diamondback_io import readInFile, diamondback_datapath

detached_regex = re.compile(r".+ TOP OF BOTTOM= (\d\d)\n")
attached_regex = re.compile(r" TOP OF CONVECTION ZONE= (\d\d)\n")

def find_rcb_diamondback(teff, grav_ms2, fsed):
    all_lines = readInFile(path.join(diamondback_datapath, f"diamondback_allmodels/t{teff}g{grav_ms2}f{fsed}_m0.0_co1.0.out"))
    all_lines.reverse()
    for line in all_lines:
        if "TOP OF" in line:
            if "UPPER ZONE" in line:
                return int(detached_regex.match(line)[1])
            else:
                return int(attached_regex.match(line)[1])


if __name__ == "__main__":
    for fsed in [1, 3, 8]:
        plt.scatter(teffs := np.arange(900, 2500, 100), [find_rcb_diamondback(teff, 316, fsed) + np.random.uniform(-0.5,0.5) for teff in teffs], label=f"{fsed = }", s=10)
        plt.xlabel("Effective temperature (K)")
        plt.ylabel("Radiative-convective boundary layer")
        plt.gca().invert_yaxis()
        plt.legend()
        
    plt.savefig("../../figures/diamondback_rcb.pdf")

