from setuptools import setup

setup(
    name='startrack',
    version=0.1,
    description='Star trail imaging with Python.',
    url='https://github.com/jgliss/startrack',
    packages=['startrack', 'startrack.tests'],
    install_requires=['numpy', 'scipy'],
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
