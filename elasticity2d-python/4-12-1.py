from Elasticity2D import FERun
from Exact import Exact, sigmarr

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

# Json data files for reduced integration
files_1GP = ("4-12_1_1x5.json", "4-12_1_2x10.json")

# Json data files for full integration
files_2GP = ("4-12_2_1x5.json", "4-12_2_2x10.json")

