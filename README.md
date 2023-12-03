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

−1 ≤ x<sub>2,1</sub> ≤ 1

# Step 7: Binary Activation Constraints

For each ReLU unit in the hidden layer:

z<sub>1,1</sub> = 1 → x<sub>1,1</sub> ≤ 0

z<sub>1,1</sub> = 0 → x<sub>1,1</sub> ≥ 0

z<sub>1,2</sub> = 1 → x<sub>1,2</sub> ≤ 0

z<sub>1,2</sub> = 0 → x<sub>1,2</sub> ≥ 0


# Step 8: Solve

Solve the MILP model using an optimization solver. 

With the above parameters we can use an MILP solver such as Gurobi, CPLEX, and PuLP (for Python) to find the optimal solution for the given DNN. The solver will find the optimal values for the decision variables x<sub>0,1</sub> , x<sub>0,2</sub> , x<sub>1,1</sub> , x<sub>1,2</sub> , x<sub>2,1</sub> and binary variables z<sub>1,1</sub> , z<sub>1,2</sub> that minimize the objective function c<sub>2,1</sub>x2,1</sub>

# Algorithm Description

1.) Algorithm to calculate upper and lower bound

    function calculate_bounds(x):
      lbest = -1  // initialize best known lower bound on x
      ubest = 1   // initialize best known upper bound on x
    
    for each f in fs:
        // Tighten upper bound
        u = f(x, boundType='upper')
        ubest = min(ubest, u)
        
        // Early return if x is non-positive
        if ubest <= 0:
            return (lbest, ubest)
        
        // Tighten lower bound
        l = f(x, boundType='lower')
        lbest = max(lbest, l)
        
        // Early return if x is non-negative
        if lbest >= 0:
            return (lbest, ubest)
    
    // Return the final bounds
    return (lbest, ubest)

Explanation of the algorithm with intial value of x = 3

Initialization:

x = 3

lbest = -1

ubest = 1

Iterative Process:

Iteration 1 (f1):

Tighten Upper Bound: u = f1(3, 'upper') evaluates to 6, so ubest becomes 1 (min of current ubest and u).

Tighten Lower Bound: l = f1(3, 'lower') evaluates to 3, so lbest becomes 3 (max of current lbest and l).

Iteration 2 (f2):

Tighten Upper Bound: u = f2(3, 'upper') evaluates to 9, so ubest becomes 1 (min of current ubest and u).

Tighten Lower Bound: l = f2(3, 'lower') evaluates to -3, so lbest remains 3 (no change).

Iteration 3 (f3):

Tighten Upper Bound: u = f3(3, 'upper') evaluates to 8, so ubest becomes 1 (min of current ubest and u).

Tighten Lower Bound: l = f3(3, 'lower') evaluates to -6, so lbest remains 3 (no change).

Final Result:

Return the final bounds: (lbest, ubest) = (3, 1)

In this example, the algorithm has determined that the variable x lies in the range [3, 1]. The bounds were iteratively tightened by evaluating the provided functions for both upper and lower bounds. The early returns help optimize the process if it is determined that x is non-positive or non-negative during the iterations.

2.) Algorithm to create a 0-1 Mixed Integer Linear Programming Model for Optimization

  Example MILP model for a simple DNN with one hidden layer
  
  using a Python modeling language like PuLP

    from pulp import LpVariable, LpProblem, LpBinary, lpSum

  Step 1: Define Notation
  
    N = 3  # Number of input nodes
  
    M = 2  # Number of hidden nodes
  
    K = 1  # Number of output nodes

  Step 2: Define Variables
  
    x = LpVariable.dicts("x", ((i, j) for i in range(1, N+1) for j in range(1, M+1)), cat=LpBinary)
  
    y = LpVariable.dicts("y", (j for j in range(1, M+1)), cat=LpBinary)

  Step 3: Formulate ReLU Activation Constraints
  
    a = [[1, -1], [-2, 1], [1, 0]]  # Example input to hidden weights
  
    b = [[2, -1]]  # Example hidden to output weights
  
    M = 1000  # A large constant

  Formulate ReLU activation constraints for hidden layer

    for i in range(1, N+1):

      for j in range(1, M+1):
          prob += a[i-1][j-1] * x[i, j] <= z[i, j]
          
          prob += z[i, j] <= a[i-1][j-1] + M * (1 - x[i, j])

  Formulate ReLU activation constraints for output layer
  
    for k in range(1, K+1):
  
      for j in range(1, M+1):
      
          prob += b[j-1][k-1] * y[j] <= z[j, k]
          
          prob += z[j, k] <= b[j-1][k-1] + M * (1 - y[j])

  Step 4: Formulate Output Layer Constraints
  
    threshold = [1]  # Example threshold for the output node
  
    for k in range(1, K+1):
  
      prob += lpSum(z[j, k] for j in range(1, M+1)) >= threshold[k-1]

  Step 5: Objective Function
  
    prob += lpSum(x[i, j] for i in range(1, N+1) for j in range(1, M+1)) + lpSum(y[j] for j in range(1, M+1))

  Step 6: Bounds (Binary variables)

  Using the algorithm-1 to calculate bounds

  Step 7: Binary Activation Constraints
  
    Hidden layer
  
    for j in range(1, M+1):
  
      prob += lpSum(x[i, j] for i in range(1, N+1)) == 1

  Output layer
  
    for k in range(1, K+1):
  
      prob += lpSum(y[j] for j in range(1, M+1)) == 1

  Step 8: Solve

    prob.solve()

  Display results
  
    print("Objective Value:", value(prob.objective))
  
    for v in prob.variables():
  
      print(v.name, "=", v.varValue)

The algorithm follows the steps as explained in the example illustration.

# Evaluation

1. Benchmarks used in the paper
   
  a.) MNIST:

  - LPd-CNNB: Large MNIST classifier for l<sub>∞</sub> norm-bound ε = 0.1

  - LPd-CNNA: MNIST classifier with asymmetric bounds for l<sub>∞</sub> norm-bound ε = 0.1

  - Adv-CNNA: Adversarially trained MNIST classifier for l<sub>∞</sub> norm-bound ε = 0.1

  - Adv-MLPB: Adversarially trained MLP for l<sub>∞</sub> norm-bound ε = 0.1

  - SDPd-MLPA: MNIST classifier for l<sub>∞</sub> norm-bound ε = 0.1

  - LPd-CNNA: MNIST classifier for l<sub>∞</sub> norm-bound ε = 0.2, 0.3, 0.4
    
  b.) CIFAR-10:

  - LPd-CNNA: Small CIFAR classifier for l<sub>∞</sub> norm-bound ε = 2/255
   
  - LPd-RES: ResNet CIFAR classifier for l<sub>∞</sub> norm-bound ε = 8/255

2. Briefly describe the tools/techniques that were used for comparison

   The paper discusses the formulation of robustness evaluation for classifiers using Mixed Integer Linear Programming (MILP) and focuses on two specific evaluation metrics: adversarial accuracy and mean minimum adversarial distortion. Here are the tools and techniques discussed:

  1.) Tool: Mixed Integer Linear Programming (MILP

  Technique: The formulation involves defining a region G(x) in the input domain for allowable perturbations and determining robustness by comparing predicted probabilities for true and other labels. MILP is employed to express the feasibility problem and constraints associated with the robustness evaluation.
Mean Minimum Adversarial Distortion Evaluation:

  2.) Tool: MILP for optimization

  Technique: The minimum adversarial distortion is determined using an optimization problem, minimizing the distance metric between the original input and the perturbed input subject to constraints. The formulation involves expressing the problem as an MILP, allowing for efficient computation of the minimum adversarial distortion. Piecewise-Linear Functions in MILP Framework

  3.) Tool: MILP

  Technique: Formulating ReLU (Rectified Linear Unit) and maximum functions using MILP. The ReLU formulation includes the use of indicator decision variables and linear constraints, ensuring tight formulations for efficient MILP solving. Similar formulations are provided for the maximum function. Progressive Bounds Tightening.

  4.) Tool: Interval Arithmetic (IA), Linear Programming (LP)

  Technique: To enhance problem tractability, progressive bounds tightening is employed. Coarse bounds are initially determined using fast procedures (IA), and bounds are refined using procedures with higher computational complexity (LP) only when necessary. This approach optimizes the trade-off between build times and solve times in MILP.

  The tools include MILP for robustness evaluation and optimization, while techniques encompass formulating piecewise-linear functions, progressive bounds tightening, and efficient optimization strategies within the MILP framework.

4. Main evaluation results

  Table 1: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-1.png
  
   a.) Performance Comparisons with Other MILP-Based Complete Verifiers (Table-1):
   
    - The MILP approach with optimizations outperforms others in terms of runtime.
  
    - Ablation tests show the significance of each optimization: progressive tightening, restricted input domain, and asymmetric bounds.

  Figure1: https://github.com/atharvasalehittal/MIPVerify/blob/main/Figure-1.png

  Figure2: https://github.com/atharvasalehittal/MIPVerify/blob/main/Figure-2.png

   b.) Comparisons to Other Complete and Incomplete Verifiers (Figures-1 and 2):

     - Verification times for determining minimum targeted adversarial distortions on MNIST samples are two to three orders of magnitude faster than the state-of-the-art complete verifier Reluplex.
  
     - The MILP verifier provides better bounds than incomplete verifiers on minimum targeted adversarial distortions.

  Table2: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-2.png

   c.) Determining Adversarial Accuracy of MNIST and CIFAR-10 Classifiers (Table-2):

     - Adversarial accuracy of various classifiers is determined for different perturbation bounds.
  
     - Lower and upper bounds on adversarial error are improved compared to existing methods.
  
     - The verifier performs well on both MNIST and CIFAR-10 datasets.

  Table3: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-3.png

  Table6: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-6.png

   d.) Observations on Determinants of Verification Time (Table-3 and Table-6):

     - Verification time is correlated with the number of ReLUs that are not provably stable and the number of labels that cannot be eliminated.
  
     - The restricted input domain significantly impacts verification time, proving stability for many ReLUs and eliminating labels.

  Table4: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-4.png
  
   e.) Performance of Verifier with Other MILP Solvers (Table-4):

     - Verifier performance varies with different MILP solvers.
  
     - Even with open-source solvers like Cbc, the verifier outperforms existing lower and upper bounds.

  Table5: https://github.com/atharvasalehittal/MIPVerify/blob/main/Table-5.png

   f.) Additional Solve Statistics (Table 5):

     - Information on nodes explored during verification provides insights into the efficiency of the MILP search tree exploration.
   








