# Example Illustration

x1 = ([0.5, 0.5], 1.0, True)

x2 = ([1.0, 1.0], -1.0, True)

hidden_layer0 = [x1, x2]

x3 = ([-1.0, 1.0], -1.0, False)      # Don't use relu for outputs

output_layer = [x3]

dnn = [hidden_layer0, output_layer]

Following are the steps of creating the 0-1 Mixed-Integer Linear Programming (MILP) model based on the above DNN architecture.
