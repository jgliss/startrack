from pkg_resources import get_distribution
from .code import *


__version__ = get_distribution('startrack').version

from .code import get_image_files, TrailMaker