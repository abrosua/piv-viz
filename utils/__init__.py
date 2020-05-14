import os
import sys

# init for post-processing
from .plot import singleplot, quiver_plot, checkstat
from .postpro import get_label, use_flowviz, velo_mean

# Manage the working directory
maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(maindir)

# Importing from LiteFlowNet
pivdir = os.path.join(os.path.dirname(maindir), "thesis_faber")
sys.path.append(pivdir)
from src.utils_plot import read_flow, read_flow_collection
