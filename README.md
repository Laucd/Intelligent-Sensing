# intelligent-sensing-toolbox

Intelligent Sensing Toolbox (Isensing) is a Python package that focuses on multivariate time series analysis. This toolbox includes multiple open-source machine learning algorithms and statistic calculations.

In data analytics, making sense of massive numbers of data requires machine learning to work on datasets from different multiple sources in order to generate insights. For situation where a node that generates data points of multiple features in time series, massive number of nodes will make analysis more challenging.

Isensing provides a list of algorithms that does features extraction, decomposition and anomaly detections.

### Installation

Isensing is built upon Python 3. To install Isensing, make sure Python 3 and pip is installed.

```python
pip install isensing
```

### Dependencies
```
pandas
numpy
scipy
sklearn
statsmodels
matplotlib
plotly
shapely
```
These dependencies will be installed automatically using pip.

### Modules
#### anomaly
```python
# class
AlphaHull
HDR

# functions
outlier_detection()
isensing_anomalies()
```

#### decomposition
```python
# class
RobustPCA
```

#### features_extraction
```python
# functions
multiple_regression()
fast_DTW()
pearsonr_correlation()
```

### Tutorial
[Link](https://gitlab.com/imda-dsl/intelligent-sensing-toolbox/blob/master/demo/Intelligent%20Sensing%20Toolkit%20Tutorial/Intelligent%20Sensing%20Toolbox%20Tutorial.md)

### References
* https://github.com/robjhyndman/anomalous-acm
* http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
* https://feb.kuleuven.be/public/u0017833/Programs/pca/robpca.txt

### License
Apache License 2.0