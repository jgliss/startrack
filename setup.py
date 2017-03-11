from setuptools import setup

with open("VERSION.rst") as f:
    version = f.readline()
    f.close()
    
setup(
    name='startrack',
    version=version,
    author='Jonas Gliss, Axel Donath',
    author_email='jonas.gliss@gmail.com',
    description='Star trail imaging with Python.',
    url='https://github.com/jgliss/startrack',
    packages=['startrack', 'startrack.tests'],
    include_package_data    =   True,  
    package_data=   {'startrack'     :   ['data/*.jpg']},
    install_requires=['numpy', 'scipy'],
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
    

    