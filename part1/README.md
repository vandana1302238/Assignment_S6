# Session 6 Assignment

## Part 1 - Understanding the backpropagation

Considering the below neural network for understanding the backpropagation.

![Framework](./imgs/img.png)

A typical nural network contains, input layer, output layer and hidden layer. 
Each neuron is associated with weights as input along with inputs, to give an 
output with an help of a function called activation function.

Lets understand with the help of above neural network,

- **i1,i2** are two inputs
- **h1,h2** are two hidden layers
- **a_h1, a_h2** are the respective activation function
- **o1,o2** are two outputs
- **a_01, a_o2** are respective activation function of output layers
- **w1,w2,w3,w4,w5,w6,w7,w8** are the respective weights
- **E1,E2** are the errors calculated from the activation functions a_01,a_02 and the target t1,t2 respectively
- where E1 + E2 corresponds to total error **E**

###Stages of training in neural network

####Step 1:
Generally a neural network, while training in **forward pass**, input data is fed,
and the network's weights are used to calculate the output of the network layer by layer, 
from the input layer to the output layer. 
The output is then compared to the desired output to compute the error

####Step 2:
**Backpropagation** is a step in which  the error is propagated backward through the network.
The gradient of the error with respect to each weight in the network is computed using the chain rule of calculus.

####Step 3:
Once the gradients of the error with respect to the weights are known, the **weights are updated in the 
opposite direction of the gradient**, aiming to reduce the error. This is typically done using an optimization algorithm 
such as gradient descent

In gradient descent-based optimization algorithms, the **learning rate** determines how much the model's parameters 
(weights) are adjusted with respect to the gradients of the loss function.

####Step 4:
Steps 1-3 are repeated iteratively for a number of epochs or until the error is minimized to an acceptable level.


## Stages in Backpropagation

As we understand, we have to adjust weights from the direction of estimated error to inputs, below are the stage by stage formulas for estimating it

### block 1

As we track with the connections between the layers, these are the computed values of hidden layers h1,h2 with respective weights and inputs as in the image

```
h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2
```
a_h1 and a_h2 are the activation functions, here sigmoid activation is used.
They introduce non-linearities into the network, allowing it to learn complex patterns and relationships in the data.

```
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2)
```
o1 and o2 are respective outputs calculated as per the neurons connections
```
o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2
````
a_01 and a_o2 are activation functions of the output layer. Again a sigmoid fn is used.
```
a_o1 = σ(o1)
a_o2 = σ(o2)
```
So the values together computed is predicted value, we find (y-y')² to obtain loss.
similarly, 
```
E_total = E1 + E2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
```
½ is used to reduce mathematical complexity while calaculating gradients.

![Framework](./imgs/exp.png)

### block 2
Once the forward pass is done, to adjust weights, we should find change in error E with respective to weights.

change in error - partial derivative ie.gradients
```
∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2

∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
```

The above formula is calucated by chain rule in calculus. We'll see one example of it

![Framework](./imgs/eexp.png)

```
∂E_total/∂w5 = ∂(E1 + E2)/∂w5
∂E_total/∂w5 = ∂E1/∂w5 (since E2 is not involved in this path)
∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5
```

These process, hence keeps on repeating with multiple iterations untill it reaches the reasonable error value near to zero.
To acheive this minmum error value, the factor **LEARNING RATE** plays a major role in achieving it.
η helps how fast we reach the min error value.

After first iteration (one forward pass+ one backpropagation) the new weights are calculated based on the learning rate η 

```
w1(in iteration 2) = w1(in interation 1)- η * ∂E_total/∂w1 (in iteration 1)
```

Thus all the weights are calculated and adjusted at each iteration

##Varying Learning rates

Learning rate should be just right, not too small or large to help reaching the min loss value.
Ideally it is 0.001 or 0.1 or 0.01 in most ofthe cases.

below are the loss curve for various learning rates:

learning rate = 0.1
![Framework](./imgs/n0point1.png)
#
learning rate = 0.2
![Framework](./imgs/n0point2.png)
#
learning rate = 0.5
![Framework](./imgs/n0point5.png)
#
learning rate = 0.8
![Framework](./imgs/n0point8.png)
#
learning rate = 1.0
![Framework](./imgs/n1.png)
#
learning rate = 2.0
![Framework](./imgs/n2.png)
#

Thus the learning rate hyper parameter plays a major role in achieveing the minimum loss. There are few other hyper parameters which impacts performace such as batch size, number of layers, etc.

Backpropagation_attempt.xlsx is attached herewith on replicating the adjusted values