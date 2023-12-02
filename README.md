# Example Illustration

x1 = ([0.5, 0.5], 1.0, True)

x2 = ([1.0, 1.0], -1.0, True)

hidden_layer0 = [x1, x2]

x3 = ([-1.0, 1.0], -1.0, False)      # Don't use relu for outputs

output_layer = [x3]

dnn = [hidden_layer0, output_layer]

Following are the steps of creating the 0-1 Mixed-Integer Linear Programming (MILP) model based on the above DNN architecture :

Step 1: Define Notation

Step 2: Define Variables

Step 3: Formulate ReLU Activation Constraints

Step 4: Formulate Output Layer Constraints

Step 5: Objective Function

Step 6: Bounds

Step 7: Binary Activation Constraints

Step 8: Solve

# Step 1: Define Notation

First we define the notation based on the DNN architecture of the illustration:

- K = 1 (as there is one hidden layer and one output layer)
  
- n<sub>0</sub> = 2 (number of input units)
  
- n<sub>1</sub> = 2 (number of units in the hidden layer)
  
- n<sub>2</sub> = 1 (number of output units)

# Step 2: Define Variables

For each layer k, we have variables x<sub>k</sub> (output vector) and z<sub>k</sub> (binary activation variables):

- x<sub>0</sub> for the input layer
  
- x<sub>1</sub>, z<sub>1</sub> for the hidden layer
  
- x<sub>2</sub>, z<sub>2</sub> for the output layer

# Step 3: Formulate ReLU Activation Constraints

For the hidden layer (ReLU activation), the constraints are:

x<sub>1</sub> = ReLU(W<sub>0</sub>x<sub>0</sub> + b<sub>0</sub>)

where W<sub>0</sub> is the weight matrix, b<sub>0</sub> is the bias vector

Since ReLU is x = max(0,w<sup>T</sup>y + b), we can formulate the constraints using binary activation variables z<sub>1</sub>:

w<sub>11</sub>x<sub>0,1</sub> + w<sub>12</sub>x<sub>0,2</sub> + b<sub>1</sub> − M<sub>1</sub>(1−z<sub>1,1</sub>) ≤ x<sub>1,1</sub> ≤ M<sub>1</sub>z<sub>1,1</sub>

​w<sub>21</sub>x<sub>0,1</sub> + w<sub>22</sub>x<sub>0,2</sub> + b<sub>2</sub> − M<sub>2</sub>(1−z<sub>1,2</sub>) ≤ x<sub>1,2</sub> ≤ M<sub>2</sub>z<sub>1,2</sub>

where w<sub>0</sub>, b<sub>0</sub> and M<sub>1</sub>, M<sub>2</sub> are weight matrices,bias and big-M constants respectively. 

The values M<sub>1</sub> and M<sub>2</sub> are used in the ReLU constraints to linearize the ReLU activation function. These values should be chosen such that they are "big enough" to ensure that the linearized constraints accurately represent the ReLU function.

Here, w<sub>0</sub> = [[0.2, -0.3],[0.4, 0.5]], b<sub>0</sub> = [[0.1],[-0.2]] and M<sub>1</sub> = 10 and  M<sub>2</sub> = 10, using these values of the above equations can be rewritten as:

0.2x<sub>0,1</sub> − 0.3x<sub>0,2</sub> + 0.1 − 10(1−z<sub>1,1</sub>) ≤ x<sub>1,1</sub> ≤ 10z<sub>1,1</sub>

0.4x<sub>0,1</sub> + 0.5x<sub>0,2</sub> − 0.2 − 10(1−z<sub>1,2</sub>) ≤ x<sub>1,2</sub> ≤ 10z<sub>1,2</sub>

# Step 4: Formulate Output Layer Constraints

For the output layer, there is no ReLU activation, so the constraints are:

x<sub>2</sub> = W<sub>1</sub>x<sub>1</sub> + b<sub>1</sub>

where W<sub>1</sub> is the weight matrix, b<sub>1</sub> is the bias vector

Here W<sub>1</sub> = [-0.1 0.2] and b<sub>1</sub> = -0.5, using these values the above equation can be rewritten as

−0.1x<sub>1,1</sub> + 0.2x<sub>1,2</sub> − 0.5 ≤ x<sub>2,1</sub> ≤ 10

# Step 5: Objective Function

The objective function is the minimization of the output values. Here, for the DNN the objective function is 

min c<sub>2,1</sub>x<sub>2,1</sub>

where the term c<sub>2,1</sub> represents the coefficient associated with the decision variable x<sub>2,1</sub>.It's a weight or cost factor that determines the contribution of x<sub>2,1</sub> to the overall objective value.  The objective function implies that minimizing c<sub>2,1</sub>x<sub>2,1</sub> is the goal of your optimization problem.

# Step 6: Bounds

Apply lower and upper bounds on input and output variables:

In this step, an algorithm is used in which it checks for each itreation the values of a particular variable such as x<sub>0,1</sub> for the each values subsituted and compared to caluate the upper bound and lower bound. Similar process is followed for other variables pending. The algorithm is explained in detailed in next section.

On calculating the values we get: x<sub>0,1</sub> = (-1, 1), x<sub>0,2</sub> = (-1, 1),  x<sub>1,1</sub> = (0, 10), x<sub>1,2</sub> = (0, 10), x<sub>2,1</sub> =(-1, 1).

The results can be summarised as below:

-1 ≤ x<sub>0,1</sub> , x<sub>0,2</sub> ≤ 1

0 ≤ x<sub>1,1</sub> , x<sub>1,2</sub> ≤ 10

−1 ≤ x<sub>2,1</sub> ≤1

