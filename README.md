# Startrack
Python toolbox for startrail photography

Note: Very early stage!

Idea: Image analysis toolbox for automised detection and tracking of stars in night-sky imagery timeseries. The purpose is to create startrail timelapse videos while maintaining the motion in the background scene. This is achieved by parameterising the history of each star-trail in the time series by combining image segmentation (for star detection in individual frames) and optical flow methods (to record the individual trajectories in time).

## Installation
Install startrack:

```bash

python setup.py install
```


## Testing
Run the tests:

```bash

python setup.py test
```

## Docs
The code documentation is hosted on [Read the Docs](http://startrack.readthedocs.io/)

Build the docs:
```bash

python setup.py build_sphinx
```


## Usage
Import the module:

```python

import startrack
print(startrack.__version__)
```
