import os
import sys

# init for post-processing
from .plot import color_map, vid_flowviz, FlowViz
from .postpro import Label, velo_mean, checkstat, column_level
from .tools import getpair, file_naming, copyfile

# Manage the working directory
maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(maindir)

# Importing from LiteFlowNet
pivdir = os.path.join(os.path.dirname(maindir), "thesis_faber")
sys.path.append(pivdir)
from src.utils_plot import read_flow, read_flow_collection, array_cropper
from inference import Inference, flowname_modifier, write_flow, piv_liteflownet, hui_liteflownet
