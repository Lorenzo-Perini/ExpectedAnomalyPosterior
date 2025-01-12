# Expected Anomaly Posterior (EAP)

`ExpectedAnomalyPosterior` is a GitHub repository containing the **EAP** [1] algorithm. It refers to the paper titled 
*Uncertainty-aware Evaluation of Auxiliary Anomalies with the Expected Anomaly Posterior*.

Check out the pdf here: [[pdf](https://openreview.net/pdf?id=Qq4ge9Qe31)].

## Abstract

Anomaly detection is the task of identifying examples that do not behave as expected. Because anomalies are rare and unexpected events, collecting real anomalous examples is often challenging in several applications. In addition, learning an anomaly detector with limited (or no) anomalies often yields poor prediction performance.
One option is to employ auxiliary synthetic anomalies to improve the model training. However, synthetic anomalies may be of poor quality: anomalies that are unrealistic or indistinguishable from normal samples may deteriorate the detector's performance.
Unfortunately, no existing methods quantify the quality of auxiliary anomalies. We fill in this gap and propose the expected anomaly posterior (EAP), an uncertainty-based score function that measures the quality of auxiliary anomalies by quantifying the total uncertainty of an anomaly detector.
Experimentally on 40 benchmark datasets of images and tabular data, we show that EAP outperforms 12 adapted data quality estimators in the majority of cases. Code of EAP is available at: https://github.com/Lorenzo-Perini/ExpectedAnomalyPosterior.

## Contents and usage

This repository is based on the opendataval (https://github.com/opendataval/opendataval) framework, adapted here to support external data for evaluation purposes.
This setup has been customized to integrate our novel methods and provide an evaluation of the approaches discussed in our paper.

Specifically, the repository contains the implementation of our proposed method, where the key files include:

- opendataval/dataval – Contains all methods to evaluate the external data, including existing metrics from the original opendataval repository;

- opendataval/dataval/uncertainty – This directory contains our novel approaches to evaluation, extending the existing framework;

- opendataval/dataloader – Provides tools for importing and reading data through a configurable data loader;

- opendataval/models – Contains models from the original opendataval repository, which are used as predictors depending on the type of data;

- EAP.py – Main implementation file for our method (referred to as EAP in the paper);

- Notebook.ipynb - Provides a simple example to demonstrate how to run the code on a toy dataset. This example is intended to help practitioners understand the basic functionality and setup required to run our evaluation.


## EAP: Uncertainty-aware Evaluation of Auxiliary Anomalies

Given: a dataset D with m << n anomalies, a set of l auxiliary anomalies, and a detector f;

Do: Design a quality score function \phi for the auxiliary anomalies, such that for any realistic anomaly x_r and any unrealistic or indistinguishable anomaly x_u, x_i, 
the realistic anomaly has a higher quality score \phi(x_r) > \phi(x_u),\phi(x_i).

The code can be used as in the Notebook file.

## Dependencies

The `ExpectedAnomalyPosterior` function requires the following python packages to be used:
- [Python 3.9](http://www.python.org)
- [Numpy 1.21.0](http://www.numpy.org)
- [Pandas 1.4.1](https://pandas.pydata.org/)
- [PyOD 1.1.0](https://pyod.readthedocs.io/en/latest/install.html)


## Contact

Contact the author of the paper: [lperini0133704@gmail.com](mailto:lperini0133704@gmail.com).


## References

[1] Perini L, Rudolph M, Schmedding S, Qiu C. Uncertainty-aware Evaluation of Auxiliary Anomalies with the Expected Anomaly Posterior. Transactions on Machine Learning Research. 2024.
