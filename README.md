# DNN-XGB-for-PPI-Prediction

# Deep neural network and extreme gradient boosting based Hybrid classifier for improved prediction of Protein-Protein interaction

This is the repository for PPI prediction using the DNN-XGB classifier. This repository contains the source code and links to some datasets used in the paper.
The DNN-XGB is compared with several traditional prediction approaches, including Nearest Neighbors, SVM, Random Forest, AdaBoost, Naive Bayes, XgBoost, and DNN.

# Abstract
Understanding the behavioral process of life and disease-causing mechanism, knowledge regarding protein-protein interactions (PPI) is essential. In this paper, a novel hybrid approach combining deep neural network (DNN) and extreme gradient boosting classifier (XGB) is employed for predicting PPI. The hybrid classifier (DNN-XGB) uses a fusion of three sequence-based features, amino acid composition (AAC), conjoint triad composition (CT), and local descriptor (LD) as inputs. The DNN extracts the hidden information through a layer-wise abstraction from the raw features that are passed through the XGB classifier. The 5-fold cross-validation accuracy for intraspecies interactions dataset of Saccharomyces cerevisiae (core subset), Helicobacter pylori, Saccharomyces cerevisiae, and Human are 98.35%, 96.19%, 97.37%, and 99.74% respectively. Similarly, accuracies of 98.50% and 97.25% are achieved for interspecies interaction dataset of Human- Bacillus Anthracis and Human- Yersinia pestis datasets, respectively. The improved prediction accuracies obtained on the independent test sets and network datasets indicate that the DNN-XGB can be used to predict cross-species interactions. It can also provide new insights into signaling pathway analysis, predicting drug targets, and understanding disease pathogenesis. Improved performance of the proposed method suggests that the hybrid classifier can be used as a useful tool for PPI prediction.

# Environment

The feature extraction codes are written in the Matlab 2015a environment. 
Fuse the features in the order AAC-CT-LD

# Dependencies

python 3.6
NumPy 1.18.1
scipy 1.3.1
scikit-learn 0.22.1
Tensorflow 1.13.1
Keras 2.3.1
Xgboost 0.90
