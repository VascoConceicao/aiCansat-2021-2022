import numpy as np

def sigmoid(matrix):                          # Sigmoid function
    
    if isinstance(matrix,list) == True:
        matrix = np.asarray(matrix)
        
    if matrix.ndim != 1:                       # for multidimensional matrixes
        for i in range(0,len(matrix)):
            matrix[i][0] = 1/(1+np.exp(-matrix[i][0]))
                
    else:                                      # for 1 dimensional matrixes
        for i in range(0,len(matrix)):
            matrix[i] = 1/(1+np.exp(-matrix[i]))
             
    return matrix

def sum_2_matrix(matrix1,matrix2):
    
    if isinstance(matrix1,list) == True:
        matrix1 = np.asarray(matrix1)
    
    if matrix1.ndim != 1:
        for i in range(0, len(matrix1)):
            matrix1[i][0]+=matrix2[i][0]
    else:
        for i in range(0, len(matrix1)):
            matrix1[i]+=matrix2[i]
        
    return matrix1  


#                          A                                         C                           B              D
def feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array): # function that gets the weighted sum for each node and output

    hidden = np.dot(weights_matrix_between_input_hidden,inputs_array)
    hidden = sum_2_matrix(hidden,bias_hidden_nodes)                             # weighted sum for hidden nodes
    hidden = sigmoid(hidden)
    
    outputs = np.dot(weights_matrix_between_hidden_output,hidden)
    outputs = sum_2_matrix(outputs,bias_outputs)                                  # weighted sum for outputs
    outputs = sigmoid(outputs)

    return outputs,hidden