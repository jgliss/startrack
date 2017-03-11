from pkg_resources import get_distribution
from os.path import abspath, dirname
from .code import *


__version__ = get_distribution('startrack').version
__dir__ =  abspath(dirname(__file__))

from .code import get_image_files, TrailMaker
import code_old