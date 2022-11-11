# Experiment-5-Implementation-of-XOR-using-RBF

## AIM:
  To classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.
  
## EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
<img width="541" alt="image" src="https://user-images.githubusercontent.com/112920679/201299438-5d1926f9-25e9-4f20-b392-1c112880ef56.png">

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below
<img width="246" alt="image" src="https://user-images.githubusercontent.com/112920679/201299568-d9398233-71d8-41b3-8b08-a39d5b95e3f1.png">

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.


A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.

<img width="261" alt="image" src="https://user-images.githubusercontent.com/112920679/201300944-5510d7f4-ea0f-45ec-875d-87f463927e9d.png">

The RBF of hidden neuron as gaussian function 

<img width="206" alt="image" src="https://user-images.githubusercontent.com/112920679/201302321-a09f72e9-2352-4f88-838c-3324f6c5f57e.png">


## ALGORIHM:
~~~
1.Import the necessary libraries of python.
2.In the end_to_end function, first calculate the similarity between the inputs and the peaks.
3.Then, to find w used the equation Aw= Y in matrix form.
4.Each row of A (shape: (4, 2)) consists of
5.index[0]: similarity of point with peak1
6.index[1]: similarity of point with peak2
7.index[2]: Bias input (1)
8.Y: Output associated with the input (shape: (4, ))
9.W is calculated using the same equation we use to solve linear regression using a closed solution (normal equation).
10.This part is the same as using a neural network architecture of 2-2-1,
11.2 node input (x1, x2) (input layer)
12.2 node (each for one peak) (hidden layer)
13.1 node output (output layer)
14.To find the weights for the edges to the 1-output unit. Weights associated would be:
15.edge joining 1st node (peak1 output) to the output node
16.edge joining 2nd node (peak2 output) to the output node
~~~
## PROGRAM:
~~~
Developed By: Aavula Tharun.
Register No: 212221240003.
~~~
~~~
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
~~~








## RESULT:
Thus Implementation of XOR problem using Radial Basis Function executed successfully.









