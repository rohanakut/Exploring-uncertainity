# Exploring-uncertainity

This project tries to develop an active learning pipeline to auto-train CNN models by integrating uncertainity approximations. TO estimate the uncertainity in depe learning models we have used dropout approximations [1]. 

To test the active learning pipeline I have used three datasets namely: MNIST [2], ISIC [3], CIFAR-10 [4].
The accuracy of the pipeline is compared using three uncertainity metrics, i.e, BALD, Entropy and variational ratio.

The following were the results for every dataset. 

## MNIST
<img src="https://github.com/rohanakut/Exploring-uncertainity/blob/main/mnist_mine.png?raw=true" width="500"/>
<!-- ![alt text](https://github.com/rohanakut/Exploring-uncertainity/blob/main/mnist_mine.png?raw=true | width=100) -->

## ISIC
<img src="https://github.com/rohanakut/Exploring-uncertainity/blob/main/isic_mine.png?raw=true" width="500"/>

## CIFAR-10
<img src="https://github.com/rohanakut/Exploring-uncertainity/blob/main/cifar_mine.png?raw=true" width="500"/>

The report for the code can be found in pdf file.
The npy files indicate the values that are used to construct the figures in the report. 


References

[1] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

[2] https://en.wikipedia.org/wiki/MNIST_database

[3] https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main

[4] https://www.cs.toronto.edu/~kriz/cifar.html
