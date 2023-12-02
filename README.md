# Example Illustration

DNN : 
x1 = ([0.5, 0.5], 1.0, True)

x2 = ([1.0, 1.0], -1.0, True)
hidden_layer0 = [x1, x2]

x3 = ([-1.0, 1.0], -1.0, False)  
output_layer = [x3]

dnn = [hidden_layer0,  output_layer]

# Step 1: Verification Condition : 
- Detailed constraints and conditions formulated for each layer.
- Verification Condition for the Hidden Layer:
a.) Neuron 1 in the Hidden Layer : 0.5 * A1,1 + 0.5 * A1,2 + 1.0 ≥ 0
This inequality represents the condition for the first neuron in the hidden layer, where A1,1 and A1,2 are the input values from the previous layer.
b.) Neuron 2 in the Hidden Layer: 1.0 * A1,1 + 1.0 * A1,2 - 1.0 ≥ 0
This inequality represents the condition for the second neuron in the hidden layer, where A1,1 and A1,2 are the input values from the previous layer.
c.) Verification Condition for the Output Layer : -1.0 * A2,1 + 1.0 * A2,2 - 1.0 = 0
This equation represents the condition for the output layer without applying ReLU. A2,1 and A2,2 are the input values from the hidden layer.

# Step 2: Reduction to Satisfiability Problem :
- In the context of the given DNN verification, the step of "Reduction to Satisfiability Problem" involves transforming the verification conditions, formulated in the previous step, into a form that can be handled by a Satisfiability Modulo Theories (SMT) solver
- Negation of the constraints obtained in the Step-1
- 
