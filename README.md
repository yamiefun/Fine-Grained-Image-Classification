# Fine-Grained Image Classification
This repository is the project 1 for NCTU course IOC5008: Selected Topics in Visual Recognition using Deep Learning.

0. [Introduction](#Introduction)
1. [How To Use](#How-To-Use)
2. [Result](#Result)
3. [Reference](#Reference)

## Introduction
The purpose of this project is to classify 196 different categories of cars with different brands and types. The training and testing dataset could be found on [kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1/data).

## How To Use
1. Download dataset from [kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1/data) and put them in this repository.
2. You will need to run `generate_label_id.py` if this is your first time running this project. After running `generate_label_id.py`, it will generate a csv file called `label_dict.csv`, which contains unique ids for every category in the dataset.
3. Now you can train and test model with `hw1.py`. You should checkout the details of arguments by tracing the code.

## Result
+ The ground truths of testing dataset are not provided by TA, so the only way to validate the accuracy of your model is to join the kaggle competition.
+ The training loss is shown in the image below. 
![](https://i.imgur.com/sS0GMbu.jpg)
+ The best accuracy is 0.9482.
![](https://i.imgur.com/HfSntjN.png)

## Reference
+ https://www.itread01.com/content/1544541602.html
+ https://rowantseng.medium.com/pytorch-%E8%87%AA%E5%AE%9A%E7%BE%A9%E8%B3%87%E6%96%99%E9%9B%86-custom-dataset-7f9958a8ff15
+ https://hackmd.io/@lido2370/HyLTOlSn4?type=view
+ https://github.com/creafz/pytorch-cnn-finetune/blob/master/examples/cifar10.py?fbclid=IwAR1CFep01WPFy0Fb0lOWz-uP6QA8C4NriuPFwslEcRQb5NOD4PYzXHLc71Q