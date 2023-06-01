import xlsxwriter
import numpy as np
import Data_ as data
import time as time
import math
import random

max_int = 9223372036854775807

learning_rate = 0.0004                         # Inicial learning rate 0000

epochs = 0                              # Number of epochs used in training

momentum = 0.9                                # Alfa used in momentum formula, 0 if you don´t want to use momentum

sequential_or_batch = "sequential"            # "sequential" to use sequential learning, "batch" to use batch learning
bold_or_annealing = "annealing"               # "bold" to use bold driver, "annealing" to use annealing
weight_decay_ = 0                             # 0 to disable weight decay, 1 to enable weight decay

#Annealing
start_parm = learning_rate                    # Start and finish parameters in annealing
end_parm = 0.00004                            #

#Bold driver
update_bold_driver_every_x_epochs = 5000      # frequency of bold driver updates
learning_rate_max = 0.5                       # learning rate limits
learning_rate_min = 0.05                      #

#Validation process
validate_every_epoch = 1000                   # Do the validation process every "x" epochs 
mse_validation = 0.1                          # Stop training when mean squared error is lower than "x"

def sum_of_squares_matrix(matrix):
    
    sum = 0
    
    if (np.ndim(matrix) == 1):
        for i in range(0,len(matrix)):
            
            aux = matrix[i] * matrix[i]
            sum += aux
            
    else:      
        for j in range(0,len(matrix)):
            for i in range(0,len(matrix[j])):

                aux = matrix[j][i] * matrix[j][i]
                sum += aux
                
    return sum

def weight_decay(errors,weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,epochs_so_far):

    n = number_hidden_nodes * number_inputs + number_outputs * number_hidden_nodes + number_hidden_nodes * 1 + number_outputs * 1  # number of weights +  biases
    
    sum = 0
    
    sum = sum + sum_of_squares_matrix(weights_matrix_between_input_hidden)
    sum = sum + sum_of_squares_matrix(weights_matrix_between_hidden_output)             # gets the sum of weights and biases used in weight decay formula
    sum = sum + sum_of_squares_matrix(bias_hidden_nodes)
    sum = sum + sum_of_squares_matrix(bias_outputs)

    omega = sum/(2*n)                                
    v = 1/(learning_rate * epochs_so_far)                                               # weight decay formula, in the report
    aux=v*omega
    errors += aux
    
    return errors

def momentum_(delta_matrix):                    # returns alfa * delta matrix, used in momentum formula
    
    return np.multiply(delta_matrix,momentum)

def annealing(max_epochs,epochs_so_far):        # gets the updated learning rate according to the formula specified in the report
    
    return end_parm + (start_parm - end_parm) * (1-1/(1+np.exp(1)**(10-20*epochs_so_far/max_epochs)))  # this is the formula :)

def bold_driver(learning_rate,old_error,new_error):

    print("\nold error -> %.6f"%old_error)
    print("new error -> %.6f\n"%new_error)
    
    control = 0      # control variable

    if (old_error - new_error) > 0:
            
        if (learning_rate * 1.05 > learning_rate_max): learning_rate = random.uniform(0.075, 0.25)              # weights are accepted, increases learning rate by 5%
        else: learning_rate *= 1.05     
            
        old_error = new_error   # updates new error with the old one
        control = 1
        
    elif (old_error - new_error) < 0:
            
        if (learning_rate * 0.7 < learning_rate_min): learning_rate = random.uniform(0.075, 0.25)               # weights are declined, decrases learning rate by 30%
        else: learning_rate *= 0.7
        
        control = -1
    
    return learning_rate,old_error,control

def destandardise(item,max,min):
    
    return ((item-0.1)/0.8)*(max-min)+min

def get_inputs_organized(inputs,i):
    
    inputs_organized = []
            
    aux = []
    aux.append(inputs[0][i])
    inputs_organized.append(aux)
            
    aux = []
    aux.append(inputs[1][i])
    inputs_organized.append(aux)
            
    #aux = []
    #aux.append(inputs[2][i])
    #inputs_organized.append(aux)
    
    return inputs_organized

def mse_(array1,array2):                # gets mean squared error
    
    difference_array = np.subtract(array1, array2)
    squared_array = np.square(difference_array)
    mse = data.mean(squared_array)
    
    return mse

def msre(array1,array2):                # gets root mean squared error 
    
    difference_array = np.subtract(array1, array2)
    for i in range (0,len(array2)):
        difference_array[i]=difference_array[i]*(1/array2[i])
    squared_array = np.square(difference_array)
    msre_ = data.mean(squared_array)
    
    return msre_

def CE(array1,array2):                # gets coefficient of efficiency
    
    difference_array_1 = np.subtract(array1, array2)
    squared_array_1 = np.square(difference_array_1)
    
    mean = data.mean(array2)
    for i in range (0,len(array2)):
        array2[i] = array2[i] - mean
    squared_array_2 = np.square(array2)
    
    
    sum1 = 0
    sum2 = 0
    for i in range(0,len(squared_array_1)):
        sum1 += squared_array_1[i]
        sum2 += squared_array_2[i]
        
    return 1-(sum1/sum2)

def CD(array1,array2):                # gets coefficient of determination or R-squared (RSqr)
    
    mean1 = data.mean(array1)
    mean2 = data.mean(array2)
    
    for i in range (0,len(array1)):
        array1[i] = array1[i] - mean1
        array2[i] = array2[i] - mean2
    
    sum1 = 0
        
    for i in range (0,len(array1)):
        sum1 = sum1 + (array1[i]*array2[i])
    
    squared_array_1 = np.square(array1)
    squared_array_2 = np.square(array2)
    
    sum2 = 0
    sum3 = 0
    
    for i in range (0,len(array1)):
        sum2 += squared_array_1[i]
        sum3 += squared_array_2[i]
        
    denominator = sum2*sum3
    denominator = math.sqrt(denominator)
        
    return (sum1/denominator)**2
    
def reformat_matrix(matrix,rows,columns):
    
    aux_1 = []
    if rows == 1 and np.ndim(matrix) != 1:
        matrix = matrix.tolist()
        for j in range (0,rows):
            for i in range (0,columns):
                aux_1.append(matrix[j][i])

    elif rows == 1 and columns == 1:
        aux_1.append(matrix[0])
        
    else:
        for j in range (0,rows):
            aux_2 = []
            for i in range (0,columns):
                aux_2.append(matrix[j][i])
            aux_1.append(aux_2)    

    return aux_1

def transpose_matrix (matrix):
    
    matrix_transpose = []
    
    if isinstance(matrix,list) == True:
        matrix = np.asarray(matrix)
        
    if matrix.ndim != 1:
        
        for j in range(0,len(matrix[0])):
            
            aux = []

            for i in range(0,len(matrix)):
                aux.append(matrix[i][j])
                
            matrix_transpose.append(aux)    
        
    else:
        for i in range(0,len(matrix)):
            aux = []
            aux.append(matrix[i])
            matrix_transpose.append(aux)  
        
    return matrix_transpose

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

#                          A                                         C                           B              D
def feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array): # function that gets the weighted sum for each node and output
    
    hidden = np.dot(weights_matrix_between_input_hidden,inputs_array)
    hidden = sum_2_matrix(hidden,bias_hidden_nodes)                             # weighted sum for hidden nodes
    hidden = sigmoid(hidden)
    
    outputs = np.dot(weights_matrix_between_hidden_output,hidden)
    outputs = sum_2_matrix(outputs,bias_outputs)                                  # weighted sum for outputs
    outputs = sigmoid(outputs)

    return outputs,hidden
    
def train(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array,targets,epochs_so_far,learning_rate): # function that trains de algorithm
    
    returned = feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array)                    # gets the weighted sum for each node and output
    outputs = returned[0][0]
    hidden = returned[1]

    weights_matrix_between_hidden_output_transpose = transpose_matrix(weights_matrix_between_hidden_output)   # matrixes used ahead
    inputs_array_transpose = transpose_matrix(inputs_array)
    hidden_transpose = transpose_matrix(hidden)

    output_errors = (targets - outputs)      # (R - u0)
    
    if weight_decay_ == 1: output_errors = weight_decay(output_errors,weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,epochs_so_far)  # updates the outputs weights with weight decay if that's the case        
    
    output_errors = output_errors * (outputs * (1 - outputs))  # gets the output errors, (R - u0) * ( uj * (1 - uj) )
    
    hidden_errors = 1 - hidden                                 # Gradient of error fucntion, uj * (1 - uj)
    hidden_errors *= hidden                                    #
    hidden_errors = np.dot(weights_matrix_between_hidden_output_transpose,output_errors) * hidden_errors             # gets hidden errors, C^T * output errors * gradient calculated before
    
    if sequential_or_batch == "sequential":
        
        delta_weights_matrix_between_input_hidden = learning_rate * (np.dot(hidden_errors,inputs_array_transpose))   # p * hidden errors * inputs^T , p is the learning rate
        delta_weights_matrix_between_hidden_output = learning_rate * (np.dot(output_errors,hidden_transpose))        # p * output errors * hidden^T
        delta_bias_hidden_nodes = learning_rate * hidden_errors * 1                                                  # p * hidden erorrs * 1
        delta_bias_outputs = learning_rate * output_errors * 1                                                       # p * output erorrs * 1
        
        weights_matrix_between_input_hidden = weights_matrix_between_input_hidden + delta_weights_matrix_between_input_hidden + momentum_(delta_bias_hidden_nodes)                          # adds momentum and update weights
        weights_matrix_between_hidden_output = weights_matrix_between_hidden_output + delta_weights_matrix_between_hidden_output + momentum_(delta_weights_matrix_between_hidden_output)
        bias_hidden_nodes = bias_hidden_nodes + delta_bias_hidden_nodes + momentum_(delta_bias_hidden_nodes)
        bias_outputs = bias_outputs + delta_bias_outputs + momentum_(delta_bias_outputs)
        
        weights_matrix_between_input_hidden = reformat_matrix(weights_matrix_between_input_hidden,number_hidden_nodes,number_inputs)       # reformats matrixers to their inicial format
        weights_matrix_between_hidden_output = reformat_matrix(weights_matrix_between_hidden_output,number_outputs,number_hidden_nodes)
        bias_hidden_nodes = reformat_matrix(bias_hidden_nodes,number_hidden_nodes,1)
        bias_outputs = reformat_matrix(bias_outputs,number_outputs,1)
        
        return weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,outputs

    if sequential_or_batch == "batch":
        
        weights_matrix_between_input_hidden_error_gradient = np.dot(hidden_errors,inputs_array_transpose)                                  # gets gradients of error function for each matrix
        weights_matrix_between_hidden_output_error_gradient = np.dot(output_errors,hidden_transpose)
        bias_hidden_nodes_error_gradient = hidden_errors * 1
        bias_outputs_error_gradient = output_errors * 1
        
        return weights_matrix_between_input_hidden_error_gradient,weights_matrix_between_hidden_output_error_gradient,bias_hidden_nodes_error_gradient,bias_outputs_error_gradient
        
def get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,test_data,finish):  # gets destandardised results
    
    output_array = []
    output_target_array = []

    for i in range(0,len(test_data[2])):
        
        inputs_array = get_inputs_organized(test_data,i)
        
        returned = feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array)  # gets the weighted sum for each node and output

        output = destandardise(returned[0][0],max_Skelton,min_Skelton)                                                                                 # destandardises results
        output_target = destandardise(test_data[2][i],max_Skelton,min_Skelton)
        output_array.append(output)
        output_target_array.append(output_target)
        
        if (finish == 1): print("target -> %.5f, guess -> %.5f, i-> %d" %(output_target,output,i))   # prints results if control variable "finish" is equal to 1
        
    return output_array,output_target_array

def use_train(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,learning_rate):
    
    validate_every = -1
    epochs_so_far = 0
    
    if sequential_or_batch == "sequential":
    
        if bold_or_annealing == "bold":
                    
            do_bold_driver = 0
                
            weights_matrix_between_input_hidden_copy = weights_matrix_between_input_hidden                  # Gets a copy of the 4 principal matrixes (matrixes with weights), because if the mean square error increases
            weights_matrix_between_hidden_output_copy = weights_matrix_between_hidden_output                # we have to go back to the matrixes we had, adjusting the learning rate
            bias_hidden_nodes_copy = bias_hidden_nodes                                                      # 
            bias_outputs_copy = bias_outputs                                                                #

            if epochs != 0:
                returned = get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,0)  # Gets the first mean square error to compare with the next one
                old_error = mse_(returned[0],returned[1])                                                                                                        #
        
        for j in range (0,epochs):
            
            epochs_so_far = j+1           # keeps track of epochs so far
            
            validate_every += 1           # used to validate the weights
                
            for i in range(0,len(training_data[0])):
                    
                inputs_array = get_inputs_organized(training_data,i)           # organizes the inputs in this format, [[a],[b],[c],[d]]
                    
                returned = train(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array,training_data[2][i],epochs_so_far,learning_rate)  # function with backpropagation algorithm that trains it
                    
                weights_matrix_between_input_hidden = returned[0]              # updates weights with new ones
                weights_matrix_between_hidden_output = returned[1]
                bias_hidden_nodes = returned[2]
                bias_outputs = returned[3]
                
            if bold_or_annealing == "annealing":
                    
                learning_rate = annealing(epochs,epochs_so_far)                # Annealing -> adjusts the learning rate every epoch
                
            if bold_or_annealing == "bold":
                if do_bold_driver == update_bold_driver_every_x_epochs:        # Bold driver -> adjusts the learning rate every "x" epochs (defined in the beggining of the code)
                        
                    do_bold_driver = 0

                    returned = get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,0)
                    new_error = mse_(returned[0],returned[1])                  # gets the new error, with updated weights
                        
                    returned = bold_driver(learning_rate,old_error,new_error)
                    learning_rate = returned[0]
                    old_error = returned[1]
                    control = returned[2]

                    if control == 1:                                           # if the weights are accepted, replace the copy matrixes with the new weights
                        weights_matrix_between_input_hidden_copy = weights_matrix_between_input_hidden
                        weights_matrix_between_hidden_output_copy = weights_matrix_between_hidden_output 
                        bias_hidden_nodes_copy = bias_hidden_nodes
                        bias_outputs_copy = bias_outputs
                        
                    elif control == -1:                                        # if the weights are declined, replace the matrixes with the old ones

                        weights_matrix_between_input_hidden = weights_matrix_between_input_hidden_copy
                        weights_matrix_between_hidden_output = weights_matrix_between_hidden_output_copy
                        bias_hidden_nodes = bias_hidden_nodes_copy
                        bias_outputs = bias_outputs_copy                 
                            
                do_bold_driver += 1
                    
            if validate_every == validate_every_epoch:
                    
                validate_every = 0
                
                returned = get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,0)   # gets the result with the validtion data subset, used to calculate the mse and stop taining

                output_array = returned[0]
                output_target_array = returned[1]

                #for i in range(0,len(output_array)):
                #    print("target -> %.6f,predicted -> %.6f\n"%(output_target_array[i],output_array[i]))
                
                print("Mean squared error -> %.6f" %mse_(output_target_array,output_array))
                print("Learning rate -> %.6f"%learning_rate)      
                
                f = open("Matrixes.txt", "w")            # writes the new weights to text file
                
                f.write("\n\n")
                        
                for row in weights_matrix_between_input_hidden:
                    np.savetxt(f, row)

                f.write("\n\n")
                            
                for row in weights_matrix_between_hidden_output:
                    f.write(str(row))
                    f.write(" / ")
                            
                f.write("\n\n")
                                                
                for row in bias_hidden_nodes:
                    np.savetxt(f, row)
                            
                f.write("\n\n\n")
                                
                for row in bias_outputs:
                    f.write(str(row))         
                    f.write("\n\n")
                            
                f.close()
                
                if mse_(output_target_array,output_array) < mse_validation: break                                 # stops the training when the mse is lower than certain number
    
    elif sequential_or_batch == "batch":

        for j in range (0,epochs):
            
            weights_matrix_between_input_hidden_error_gradient = np.zeros((number_hidden_nodes,number_inputs))    # initializes gradient matrix for each input, with zeros in every position
            weights_matrix_between_hidden_output_error_gradient = np.zeros((number_outputs,number_hidden_nodes))
            bias_hidden_nodes_error_gradient = np.zeros((number_hidden_nodes,1))
            bias_outputs_error_gradient = np.zeros((number_outputs,1))
            
            epochs_so_far = j+1           # keeps track of epochs so far
            
            validate_every += 1           # used to validate the weights
            
            for i in range(0,len(training_data[0])):
        
                inputs_array = get_inputs_organized(training_data,i)
                        
                returned = train(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array,training_data[2][i],epochs_so_far,learning_rate)  # function with backpropagation algorithm that trains it
                
                weights_matrix_between_input_hidden_error_gradient = np.add(returned[0],weights_matrix_between_input_hidden_error_gradient)                                                             # updates gradient matrixes, adding the gradient of the error function, for each set of inputs
                weights_matrix_between_hidden_output_error_gradient = np.add(returned[1],weights_matrix_between_hidden_output_error_gradient)
                bias_hidden_nodes_error_gradient = np.add(returned[2],bias_hidden_nodes_error_gradient)
                bias_outputs_error_gradient = np.add(returned[3],bias_outputs_error_gradient)
            
            weights_matrix_between_input_hidden = weights_matrix_between_input_hidden + np.dot(learning_rate, weights_matrix_between_input_hidden_error_gradient) / len(training_data[0])               # updates weights and biases matrixes every epoch, formula in report
            weights_matrix_between_hidden_output = weights_matrix_between_hidden_output + np.dot(learning_rate, weights_matrix_between_hidden_output_error_gradient) / len(training_data[0]) 
            bias_hidden_nodes = bias_hidden_nodes + np.dot(learning_rate, bias_hidden_nodes_error_gradient) / len(training_data[0])
            bias_outputs = bias_outputs + np.dot(learning_rate, bias_outputs_error_gradient) / len(training_data[0])
            
            weights_matrix_between_input_hidden = reformat_matrix(weights_matrix_between_input_hidden,number_hidden_nodes,number_inputs)    # reformats matrixes to their initial format
            weights_matrix_between_hidden_output = reformat_matrix(weights_matrix_between_hidden_output,number_outputs,number_hidden_nodes)
            bias_hidden_nodes = reformat_matrix(bias_hidden_nodes,number_hidden_nodes,1)
            bias_outputs = reformat_matrix(bias_outputs,number_outputs,1)
                
            if validate_every == validate_every_epoch:  # the same as in sequential learning
                        
                validate_every = 0
                
                returned = get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,validation_data,0)

                output_array = returned[0]
                output_target_array = returned[1]
                
                print("Mean squared error -> %.6f" %mse_(output_target_array,output_array))
                print("Learning rate -> %.6f"%learning_rate)      
                
                f = open("Matrixes.txt", "w")
                
                f.write("Mean squared error\n -> %.6f" %mse_(output_target_array,output_array))
                f.write("Learning rate\n\n -> %.6f"%learning_rate)
                        
                for row in weights_matrix_between_input_hidden:
                    np.savetxt(f, row)

                f.write("\n\n")
                            
                for row in weights_matrix_between_hidden_output:
                    f.write(str(row))
                    f.write(" / ")
                            
                f.write("\n\n")
                                                
                for row in bias_hidden_nodes:
                    np.savetxt(f, row)
                            
                f.write("\n\n\n")
                                
                for row in bias_outputs:
                    f.write(str(row))         
                    f.write("\n\n")
                            
                f.close()
                
                if mse_(output_target_array,output_array) < mse_validation: break
                
    return weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,learning_rate,epochs_so_far
                

returned = data.get_data_from_excel()                                   # gets data from excel (organized)

training_data = returned[0]                                             # 0 -> Temperatura  1 -> Humidade  2 -> Pressao  3 -> Indice
validation_data = returned[1]
test_data = returned[2]
max_Skelton = returned[3]
min_Skelton = returned[4]

number_inputs = len(training_data) - 1 
number_outputs = len(training_data) - number_inputs
number_hidden_nodes = 4 #int((number_inputs+number_outputs)/2)

min_interval = -2/number_inputs                                                                                                                     # Initializes every matrix with random numbers in the interval [-2/n , 2/n] n -> number of inputs
max_interval = abs(min_interval)                                                                                                                    # 
weights_matrix_between_input_hidden = np.random.uniform(low = min_interval,high = max_interval,size = (number_hidden_nodes,number_inputs))          # weights_matrix_between_input_hidden   -> has weights that go from input to hidden nodes
weights_matrix_between_hidden_output = np.random.uniform(low = min_interval,high = max_interval,size = (number_outputs,number_hidden_nodes))        # weights_matrix_between_hidden_output  -> has weights that go from hidden nodes to outputs
bias_hidden_nodes = np.random.uniform(low = min_interval,high = max_interval,size = (number_hidden_nodes,1))                                        # bias_hidden_nodes                     -> has bias from hidden nodes 
bias_outputs = np.random.uniform(low = min_interval,high = max_interval,size = (number_outputs,1))                                                  # bias_outputs                          -> has bias from outputs

weights_matrix_between_input_hidden = reformat_matrix(weights_matrix_between_input_hidden,number_hidden_nodes,number_inputs)                        # Reformats matrixes to be in the following forms:
weights_matrix_between_hidden_output = reformat_matrix(weights_matrix_between_hidden_output,number_outputs,number_hidden_nodes)                     # -> Dimension = 1 -> [a,b,c,d,e]
bias_hidden_nodes = reformat_matrix(bias_hidden_nodes,number_hidden_nodes,1)                                                                        # -> Dimension = 2 -> [[a,b,c,d] , [e,f,g,h]]
bias_outputs = reformat_matrix(bias_outputs,number_outputs,1)                                                                                       #

weights_matrix_between_input_hidden = [[1.093524224383599908e+01, 1.214035315431151574e+01], [-1.088279743636970309e+01, -1.052648579738700718e+01], [-1.616084379028606932e+01, -1.052999274390015216e+01], [1.594765932816003939e+01, 6.035456604081985255e+00]]
weights_matrix_between_hidden_output = [-6.671369217994886, -4.158976697245206, -9.578999576508618, -7.257751200901053]
bias_hidden_nodes = [[-1.109260993749014546e+01], [8.567703641622069100e+00], [1.333083507989023353e+01], [-1.087154941841754408e+01]]
bias_outputs = [13.575327851663541]

weights_matrix_between_input_hidden = [[1.179254531427120689e+01, 1.264563091847170995e+01], [-1.083832000213494773e+01, -9.791435223970211155e+00], [-1.581817680419104732e+01, -1.068864726790247843e+01], [1.498300209409340589e+01, 5.759502391600410220e+00]]
weights_matrix_between_hidden_output = [-7.342481064839878, -4.193543823694243, -10.481743416889536, -7.593006036882884]
bias_hidden_nodes = [[-1.189752817304256993e+01], [8.353218827017046877e+00], [1.325546480003319694e+01], [-1.024225307434639554e+01]]
bias_outputs = [14.536338707860855]

beggining = time.time()
returned = use_train(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,learning_rate)

end = time.time()
elapsed_time = end - beggining
elapsed_time_minutes = int(elapsed_time / 60)
elapsed_time_seconds = elapsed_time - (elapsed_time_minutes * 60)

weights_matrix_between_input_hidden = returned[0]         # 
weights_matrix_between_hidden_output = returned[1]        #
bias_hidden_nodes = returned[2]                           # use_train() function returns this values
bias_outputs = returned[3]                                # 
learning_rate = returned[4]                               #
it_took_x_epochs = returned[5]                            #

returned = get_results(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,training_data,0)

output_array = returned[0]                                # get_results() function returns this values
output_target_array = returned[1]                         #

for i in range(0,len(output_array)):
    output_array[i] /= 10
    output_target_array[i] /= 10
    print("target -> %.2f, guess -> %.2f" %(output_target_array[i],output_array[i]))

workbook = xlsxwriter.Workbook(r"Results.xlsx")           # writes results to an excel file
worksheet = workbook.add_worksheet()

row = 0

for i in range(0,len(output_target_array)):
    worksheet.write(i, 0, output_target_array[i])
    worksheet.write(i, 1, output_array[i])

workbook.close()

print("\nIt took %d epochs"%it_took_x_epochs,end = '')                                            # Print results
print(" and",elapsed_time_minutes,"minutes and",elapsed_time_seconds,"seconds\n")

mse = mse_(output_target_array,output_array)
msre_ = msre(output_target_array,output_array)
cE_ = CE(output_target_array,output_array)
cD_ = CD(output_target_array,output_array)

print("Mean squared error -> %.4f"%mse)
print("Root mean squared error -> %.4f"%math.sqrt(mse))
print("Mean squared relative error -> %.4f"%msre_)
print("Coefficient of Efficiency -> %.4f"%cE_)
print("Coefficient of Determination -> %.4f"%cD_)

print("\n")
print("A -> ", weights_matrix_between_input_hidden)
print("C -> ", weights_matrix_between_hidden_output)
print("B -> ", bias_hidden_nodes)
print("D -> ", bias_outputs)
print("\n\n")