import torch


#initialize
targets = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,.9]
target_tensor = torch.tensor(targets,dtype=torch.float32)
inputs = torch.rand(1,10)

input_tensor2 = torch.tensor(inputs,dtype=torch.float32)
weight_tensor2 = torch.rand(10,10)-.5

input_tensor1 = torch.tensor(inputs,dtype=torch.float32)
weight_tensor1 = torch.rand(10,10)-.5

ones_square_tensor = torch.ones(10,10)
ones_line_tensor = torch.ones(1,10)

learningRate = .3
runs = 1000
count = 0

while count < runs:
    #feedforward
    output_tensor1 = torch.matmul(input_tensor1,weight_tensor2)
    activated_tensor1 = torch.tanh(output_tensor1)
    input_tensor2 = activated_tensor1

    output_tensor2 = torch.matmul(input_tensor2,weight_tensor2)
    activated_tensor2 = torch.tanh(output_tensor2)

    #calculate error
    total_error2 = torch.sum(torch.pow(torch.subtract(activated_tensor2,target_tensor),2))/2
    print(total_error2)

    #calculate activation derivates
    #layer 2
    a = torch.exp(-output_tensor2)
    b = torch.exp(output_tensor2)
    c = sum(a,b)
    d = torch.pow(c,2)
    delta_activation_tensor2 = 4/d

    #layer 1
    r = torch.exp(-output_tensor2)
    s = torch.exp(output_tensor2)
    t = sum(r,s)
    u = torch.pow(t,2)
    delta_activation_tensor1 = 4/u

    #calculate output error derivatives
    delta_error_tensor2 = torch.subtract(activated_tensor2,target_tensor,alpha=1)
    flattened_L2_weight_tensor = torch.matmul(ones_line_tensor,weight_tensor2)

    #calculate layer 2 weight gradient tensor
    activation_and_error_deltas2 = torch.multiply(delta_error_tensor2,delta_activation_tensor2)
    input_tensor_transpose2 = torch.transpose(input_tensor2,0,1)
    weight_gradient_tensor_from_inputs2 = torch.multiply(input_tensor_transpose2,ones_square_tensor)
    complete_weight_gradient_tensor2 = torch.multiply(torch.multiply(activation_and_error_deltas2,weight_gradient_tensor_from_inputs2),learningRate)

    #calculate layer 1 weight gradient tensor
    activation_and_error_deltas1a = torch.multiply(flattened_L2_weight_tensor,delta_activation_tensor1)
    activation_and_error_deltas1b = torch.multiply(activation_and_error_deltas1a,activation_and_error_deltas2)
    input_tensor_transpose1 = torch.transpose(input_tensor1,0,1)
    weight_gradient_tensor_from_inputs1 = torch.multiply(input_tensor_transpose1,ones_square_tensor)
    complete_weight_gradient_tensor1 = torch.multiply(torch.multiply(activation_and_error_deltas1b,weight_gradient_tensor_from_inputs1),learningRate)

    #update weight tensor
    weight_tensor2 = torch.subtract(weight_tensor2,complete_weight_gradient_tensor2,alpha=1)
    weight_tensor1 = torch.subtract(weight_tensor1,complete_weight_gradient_tensor1,alpha=1)

    count = count + 1
















