import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="isensing",
    version="0.1.1",
    author="IMDA Digital Services Lab",
    #author_email="",
    description="Intelligent Sensing Toolbox for Multivariate Time Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/imda-dsl/intelligent-sensing-toolbox",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
        'scipy',
        'fastdtw',
        'sklearn',
        'matplotlib',
        'plotly',
        'descartes',
		'shapely'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
)