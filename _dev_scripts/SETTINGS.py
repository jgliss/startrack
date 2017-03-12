# -*- coding: utf-8 -*-
"""
Global settings for example scripts
"""
from os.path import join, abspath
from optparse import OptionParser

SAVEFIGS = 1 # save plots from this script in SAVE_DIR
DPI = 300 #pixel resolution for saving
FORMAT = "png" #format for saving

SCREENPRINT = 0 #show images on screen when executing script

# Directory where results are stored

SAVE_DIR = abspath(join(".", "scripts_out"))

OPTPARSE = OptionParser()
OPTPARSE.add_option('--show', dest="show", default=SCREENPRINT)

from matplotlib import rcParams
rcParams.update({'font.size': 13})