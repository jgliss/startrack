from pkg_resources import get_distribution
from os.path import abspath, dirname

__version__ = get_distribution('startrack').version
__dir__ =  abspath(dirname(__file__))

import io
from .enterprise import TrailMaker
from .image import Image
import helpers
import code_old
