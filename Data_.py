import pandas as pd
import math
import numpy as np

days_in_months = [    31    ,    28    ,    31    ,    30    ,    31    ,    30    ,    31    ,    31    ,    30    ,    31    ,    30    ,    31    ]
####               January  ,  Febuary ,   March  ,   April  ,    May   ,    June  ,   July   ,  August  , September,  October , November ,  December

weights = [[1],[2/3,1/3],[0.5,0.3,0.2],[0.45,0.3,0.15,0.1],[0.4,0.25,0.2,0.1,0.05]]

def standardise(item,max,min):
    
    return 0.8*((item - min)/(max - min))+0.1

def find_max_in_array(array):
    
    size = len(array)
    
    max = array[0]

    for i in range(1,size):
        
        if array[i]>max: max = array[i]
        
    return max

def find_min_in_array(array):
    
    size = len(array)
    
    min = array[0]

    for i in range(1,size):
        
        if array[i]<min: min = array[i]
        
    return min
    
def lag_by_x(array_1,array_2,lag): 

    for i in range(0,lag):
        array_2.pop(0)
        array_1.pop(len(array_1)-1)

    return array_1,array_2
    
def moving_average_array(array,steps,Indice,control):
    
    if steps == 1: return array,Indice
    
    array_copy = array.copy()
    
    size = len(array)
    
    weights_ = weights[steps-1]

    for j in range(steps-1,size):

        array[j] *= weights_[0]
        for i in range (1,steps):
            array[j] += (weights_[i] * array_copy[j-i])

    for i in range (1, steps):
        array.pop(0)
        Indice.pop(0)
          
    return array,Indice
        

def standard_deviation(array,mean):
    
    sum = 0
    size = len(array)
    
    for i in range(0, size):
        
        difference = array[i] - mean
        difference = abs(difference)
        difference *= difference
        sum+=difference
            
    sum /= (size-1)
    
    return math.sqrt(sum)   
    
def mean(array):
    
    sum = 0
    size = len(array)
    
    for i in range(0,size):
        sum+=array[i]

    return sum/size

def get_data_from_excel():
    
    excel = pd.read_excel(r"C:\Users\VascoConceicao\Desktop\Python\AI CANSAT\Dados.xlsx")
    dimensions = excel.shape
    
    Temperatura = []
    Humidade = []
    #Pressao = []
    Indice = []

    for i in range (0, dimensions[0]):

        if issubclass(type(excel["Temperatura"].iloc[i]), str) or excel["Temperatura"].iloc[i]<0: Temperatura.append(excel["Temperatura"].iloc[i-1])
        else: Temperatura.append(excel["Temperatura"].iloc[i])
        
        if issubclass(type(excel["Humidade"].iloc[i]), str) or excel["Humidade"].iloc[i]<0: Humidade.append(excel["Humidade"].iloc[i-1])
        else: Humidade.append(excel["Humidade"].iloc[i])
        
        #if issubclass(type(excel["Pressao"].iloc[i]), str) or excel["Pressao"].iloc[i]<0: Pressao.append(excel["Pressao"].iloc[i-1])   
        #else: Pressao.append(excel["Pressao"].iloc[i])
        
        if issubclass(type(excel["Indice"].iloc[i]), str) or excel["Indice"].iloc[i]<0: Indice.append(excel["Indice"].iloc[i-1])
        else: Indice.append(excel["Indice"].iloc[i])

    for i in range(0,len(Indice)):
        Indice[i]*=10

    mean_Temperatura = mean(Temperatura)
    mean_Humidade = mean(Humidade)
    #mean_Pressao = mean(Pressao)
    mean_Indice = mean(Indice)
    
    standard_deviation_Temperatura = standard_deviation(Temperatura,mean_Temperatura)
    standard_deviation_Humidade = standard_deviation(Humidade,mean_Humidade)
    #standard_deviation_Pressao = standard_deviation(Pressao,mean_Pressao)
    standard_deviation_Indice = standard_deviation(Indice,mean_Indice)

    for i in range (0,len(Temperatura)):
        
        if standard_deviation_Temperatura*3 + mean_Temperatura < Temperatura[i]:
            Temperatura[i] = Temperatura[i-1]
        if standard_deviation_Humidade*3 + mean_Humidade < Humidade[i]:
            Humidade[i] = Humidade[i-1]
        #if standard_deviation_Pressao*3 + mean_Pressao < Pressao[i]:
            #Pressao[i] = Pressao[i-1]
        if standard_deviation_Indice*3 + mean_Indice < Indice[i]:
            Indice[i] = Indice[i-1]

    steps = 1
    step_final = 1
    lag_final = 0

    while True:
        
        lag  = 1

        Indice_copy = Indice.copy()
        Temperatura_moving_average = Temperatura.copy()
        
        returned = moving_average_array(Temperatura_moving_average,steps,Indice_copy,0)

        Temperatura_moving_average = returned[0]
        Indice_copy = returned[1]

        Temperatura_correlation_current = np.corrcoef(Indice_copy,Temperatura_moving_average)[1][0]
        
        if steps == 1: Temperatura_correlation_max = Temperatura_correlation_current
        else: 
            if Temperatura_correlation_current > Temperatura_correlation_max: 
                step_final = steps
                lag_final = 0
                Temperatura_correlation_max = Temperatura_correlation_current
        
        while True:
            
            Indice_copy_copy = Indice_copy.copy()
            Temperatura_moving_average_copy = Temperatura_moving_average.copy()

            returned = lag_by_x(Temperatura_moving_average_copy,Indice_copy_copy,lag)

            Temperatura_moving_average_copy = returned[0]
            Indice_copy_copy = returned[1]
            
            Temperatura_correlation_current = np.corrcoef(Indice_copy_copy,Temperatura_moving_average_copy)[1][0]
            
            if Temperatura_correlation_current > Temperatura_correlation_max: 
                Temperatura_correlation_max = Temperatura_correlation_current
                step_final = steps
                lag_final = lag
                
            else : 
                if (lag == 6): break
            
            lag += 1

        steps += 1
        
        if steps == 6: break

    ############ 1
    steps_array = []
    lag_array = []
    
    steps_array.append(step_final)
    lag_array.append(lag_final)
    
    step_final = 1
    lag_final = 0

    steps = 1
    
    while True:
        
        lag  = 1

        Indice_copy = Indice.copy()
        Humidade_moving_average = Humidade.copy()
        
        returned = moving_average_array(Humidade_moving_average,steps,Indice_copy,0)
        
        Humidade_moving_average = returned[0]
        Indice_copy = returned[1]
        
        Humidade_correlation_current = np.corrcoef(Indice_copy,Humidade_moving_average)[1][0]
        
        if steps == 1: Humidade_correlation_max = Humidade_correlation_current
        else: 
            if Humidade_correlation_current > Humidade_correlation_max:
                Humidade_correlation_max = Humidade_correlation_current 
                step_final = steps
                lag_final = 0
                 
        while True:
            
            Indice_copy_copy = Indice_copy.copy()
            Humidade_moving_average_copy = Humidade_moving_average.copy()

            returned = lag_by_x(Humidade_moving_average_copy,Indice_copy_copy,lag)

            Humidade_moving_average_copy = returned[0]
            Indice_copy_copy = returned[1]
                    
            Humidade_correlation_current = np.corrcoef(Indice_copy_copy,Humidade_moving_average_copy)[1][0]
            
            if Humidade_correlation_current > Humidade_correlation_max: 
                Humidade_correlation_max = Humidade_correlation_current
                step_final = steps
                lag_final = lag
                
            else : 
                if (lag == 6): break
            
            lag += 1

        steps += 1
        
        if steps == 6: break
    
        
    #################### 2
    
    steps_array.append(step_final)
    lag_array.append(lag_final)
    
    step_final = 1
    lag_final = 0

    steps = 1
    """""
    while True:
        
        lag  = 1

        Indice_copy = Indice.copy()
        Pressao_moving_average = Pressao.copy()
        
        returned = moving_average_array(Pressao_moving_average,steps,Indice_copy,0)
        
        Pressao_moving_average = returned[0]
        Indice_copy = returned[1]
        
        Pressao_correlation_current = np.corrcoef(Indice_copy,Pressao_moving_average)[1][0]
        
        if steps == 1: Pressao_correlation_max = Pressao_correlation_current
        else: 
            if Pressao_correlation_current > Pressao_correlation_max: 
                Pressao_correlation_max = Pressao_correlation_current
                step_final = steps
                lag_final = 0          
        
        while True:
            
            Indice_copy_copy = Indice_copy.copy()
            Pressao_moving_average_copy = Pressao_moving_average.copy()

            returned = lag_by_x(Pressao_moving_average_copy,Indice_copy_copy,lag)

            Pressao_moving_average_copy = returned[0]
            Indice_copy_copy = returned[1]
                    
            Pressao_correlation_current = np.corrcoef(Indice_copy_copy,Pressao_moving_average_copy)[1][0]
                        
            if Pressao_correlation_current > Pressao_correlation_max: 
                Pressao_correlation_max = Pressao_correlation_current
                step_final = steps
                lag_final = lag
                
            else : 
                if (lag == 6): break
            
            lag += 1

        steps += 1
        
        if steps == 6: break
            
    ###################### 3
    
    steps_array.append(step_final)
    lag_array.append(lag_final)
    """""

    Indice_copy = Indice.copy()
    returned = moving_average_array(Temperatura,steps_array[0],Indice_copy,0)
    Temperatura = returned[0]
    Indice_copy = returned[1]
    returned = lag_by_x(Temperatura,Indice_copy,lag_array[0])
    Temperatura = returned[0]
    
    Indice_copy = Indice.copy()
    returned = moving_average_array(Humidade,steps_array[1],Indice_copy,0)
    Humidade = returned[0]
    Indice_copy = returned[1]
    returned = lag_by_x(Humidade,Indice_copy,lag_array[1])
    Humidade = returned[0]
    
    #Indice_copy = Indice.copy()
    #returned = moving_average_array(Pressao,steps_array[2],Indice_copy,1)
    #Pressao = returned[0]
    #Indice_copy = returned[1]
    #returned = lag_by_x(Pressao,Indice_copy,lag_array[2])
    #Pressao = returned[0]
    
    sum_of_steps_lag = []
    
    for i in range(0,len(steps_array)):
        sum_of_steps_lag.append(steps_array[i]+lag_array[i])
        
    max = find_max_in_array(sum_of_steps_lag)
    
    gap = max
    day = gap                                                  #### assumed that the gap isnt more than 30, which is impossible
    month = 6
    year = 2020
  
    Temperatura_winters = []
    Temperatura_springs = []
    Temperatura_summers = []
    Temperatura_autumns = []
    
    Humidade_winters = []
    Humidade_springs = []
    Humidade_summers = []
    Humidade_autumns = []
    
    Pressao_winters = []
    Pressao_springs = []
    Pressao_summers = []
    Pressao_autumns = []
    
    Indice_winters = []
    Indice_springs = []
    Indice_summers = []
    Indice_autumns = []

    for i in range(0,len(Humidade)):
        
        if month == 1: 
            Temperatura_winters.append(Temperatura[i+4])
            Humidade_winters.append(Humidade[i])
            #Pressao_winters.append(Pressao[i])
            Indice_winters.append(Indice[i+4])
            
        if month == 2:
            Temperatura_winters.append(Temperatura[i])
            Humidade_winters.append(Humidade[i])
            #Pressao_winters.append(Pressao[i])
            Indice_winters.append(Indice[i+4])
            
        if month == 3:
            
            if (day<=20):
                Temperatura_winters.append(Temperatura[i+4])
                Humidade_winters.append(Humidade[i])
                #Pressao_winters.append(Pressao[i])
                Indice_winters.append(Indice[i+4])
                
            else:
                Temperatura_springs.append(Temperatura[i+4])
                Humidade_springs.append(Humidade[i])
                #Pressao_springs.append(Pressao[i])
                Indice_springs.append(Indice[i+4])
                
        if month == 4:
            Temperatura_springs.append(Temperatura[i+4])
            Humidade_springs.append(Humidade[i])
            #Pressao_springs.append(Pressao[i])
            Indice_springs.append(Indice[i+4])
            
        if month == 5:
            Temperatura_springs.append(Temperatura[i+4])
            Humidade_springs.append(Humidade[i])
            #Pressao_springs.append(Pressao[i])
            Indice_springs.append(Indice[i+4])
            
        if month == 6:
            
            if (day<=21):
                Temperatura_springs.append(Temperatura[i+4])
                Humidade_springs.append(Humidade[i])
                #Pressao_springs.append(Pressao[i])
                Indice_springs.append(Indice[i+4])
            
            else:
                Temperatura_summers.append(Temperatura[i+4])
                Humidade_summers.append(Humidade[i])
                #Pressao_summers.append(Pressao[i])
                Indice_summers.append(Indice[i+4])   
               
        if month == 7:
            Temperatura_summers.append(Temperatura[i+4])
            Humidade_summers.append(Humidade[i])
            #Pressao_summers.append(Pressao[i])
            Indice_summers.append(Indice[i+4])
            
        if month == 8:
            Temperatura_summers.append(Temperatura[i+4])
            Humidade_summers.append(Humidade[i])
            #Pressao_summers.append(Pressao[i])
            Indice_summers.append(Indice[i+4])
            
        if month == 9:
            
            if (day<=21):
                Temperatura_summers.append(Temperatura[i+4])
                Humidade_summers.append(Humidade[i])
                #Pressao_summers.append(Pressao[i])
                Indice_summers.append(Indice[i+4])
                
            else:
                Temperatura_autumns.append(Temperatura[i+4])
                Humidade_autumns.append(Humidade[i])
                #Pressao_autumns.append(Pressao[i])
                Indice_autumns.append(Indice[i+4])
                
        if month == 10:
            Temperatura_autumns.append(Temperatura[i+4])
            Humidade_autumns.append(Humidade[i])
            #Pressao_autumns.append(Pressao[i])
            Indice_autumns.append(Indice[i+4])
            
        if month == 11:
            Temperatura_autumns.append(Temperatura[i+4])
            Humidade_autumns.append(Humidade[i])
            #Pressao_autumns.append(Pressao[i])
            Indice_autumns.append(Indice[i+4])
            
        if month == 12:
            
            if (day<=21):
                Temperatura_autumns.append(Temperatura[i+4])
                Humidade_autumns.append(Humidade[i])
                #Pressao_autumns.append(Pressao[i])
                Indice_autumns.append(Indice[i+4])
                
            else: 
                Temperatura_winters.append(Temperatura[i+4])
                Humidade_winters.append(Humidade[i])
                #Pressao_winters.append(Pressao[i])
                Indice_winters.append(Indice[i+4])       
        
        if day == days_in_months[month-1]:
                
            if month == 10 and year == 2020:
                year+=1
                month = 3
                day = 1
                
            else:
                day = 1
                month+=1
            
        else:
            day += 1
    
    max_Temperatura = []
    max_Humidade = []
    #max_Pressao = []
    max_Indice = []
    
    min_Temperatura = []
    min_Humidade = []
    #min_Pressao = []
    min_Indice = []
    
    max_Temperatura.append(find_max_in_array(Temperatura_winters))
    max_Temperatura.append(find_max_in_array(Temperatura_summers))
    max_Temperatura.append(find_max_in_array(Temperatura_springs))
    
    min_Temperatura.append(find_min_in_array(Temperatura_winters))
    min_Temperatura.append(find_min_in_array(Temperatura_summers))
    min_Temperatura.append(find_min_in_array(Temperatura_springs))
    
    max_Temperatura_ = find_max_in_array(max_Temperatura)
    min_Temperatura_ = find_min_in_array(min_Temperatura)
    
    max_Humidade.append(find_max_in_array(Humidade_winters))
    max_Humidade.append(find_max_in_array(Humidade_summers))
    max_Humidade.append(find_max_in_array(Humidade_springs))
    
    min_Humidade.append(find_min_in_array(Humidade_winters))
    min_Humidade.append(find_min_in_array(Humidade_summers))
    min_Humidade.append(find_min_in_array(Humidade_springs))
    
    max_Humidade_ = find_max_in_array(max_Humidade)
    min_Humidade_ = find_min_in_array(min_Humidade)
    """""
    max_Pressao.append(find_max_in_array(Pressao_winters))
    max_Pressao.append(find_max_in_array(Pressao_autumns))
    max_Pressao.append(find_max_in_array(Pressao_springs))
    
    min_Pressao.append(find_min_in_array(Pressao_winters))
    min_Pressao.append(find_min_in_array(Pressao_autumns))
    min_Pressao.append(find_min_in_array(Pressao_springs))
    
    max_Pressao_ = find_max_in_array(max_Pressao)
    min_Pressao_ = find_min_in_array(min_Pressao)
    """""

    max_Indice.append(find_max_in_array(Indice_winters))
    max_Indice.append(find_max_in_array(Indice_summers))
    max_Indice.append(find_max_in_array(Indice_springs))
    
    min_Indice.append(find_min_in_array(Indice_winters))
    min_Indice.append(find_min_in_array(Indice_summers))
    min_Indice.append(find_min_in_array(Indice_springs))
    
    max_Indice_ = find_max_in_array(max_Indice)
    min_Indice_ = find_min_in_array(min_Indice)
    
    size = len(Temperatura_winters)
    
    for i in range(0,size):
        
        Temperatura_winters[i] = standardise(Temperatura_winters[i],max_Temperatura_,min_Temperatura_)
        Humidade_winters[i] = standardise(Humidade_winters[i],max_Humidade_,min_Humidade_)
        #Pressao_winters[i] = standardise(Pressao_winters[i],max_Pressao_,min_Pressao_)       
        Indice_winters[i] = standardise(Indice_winters[i],max_Indice_,min_Indice_) 
    
    size = len(Temperatura_springs)
    
    for i in range(0,size):
        
        Temperatura_springs[i] = standardise(Temperatura_springs[i],max_Temperatura_,min_Temperatura_)
        Humidade_springs[i] = standardise(Humidade_springs[i],max_Humidade_,min_Humidade_)
        #Pressao_springs[i] = standardise(Pressao_springs[i],max_Pressao_,min_Pressao_)
        Indice_springs[i] = standardise(Indice_springs[i],max_Indice_,min_Indice_)
        
    size = len(Temperatura_summers)
    
    for i in range(0,size):
        
        Temperatura_summers[i] = standardise(Temperatura_summers[i],max_Temperatura_,min_Temperatura_)
        Humidade_summers[i] = standardise(Humidade_summers[i],max_Humidade_,min_Humidade_)
        #Pressao_summers[i] = standardise(Pressao_summers[i],max_Pressao_,min_Pressao_)
        Indice_summers[i] = standardise(Indice_summers[i],max_Indice_,min_Indice_)
    
    size = len(Temperatura_autumns)
    
    for i in range(0,size):
        
        Temperatura_autumns[i] = standardise(Temperatura_autumns[i],max_Temperatura_,min_Temperatura_)
        Humidade_autumns[i] = standardise(Humidade_autumns[i],max_Humidade_,min_Humidade_)
        #Pressao_autumns[i] = standardise(Pressao_autumns[i],max_Pressao_,min_Pressao_)
        Indice_autumns[i] = standardise(Indice_autumns[i],max_Indice_,min_Indice_)
    
    training_data = []
    
    Temperatura_summers += Temperatura_springs
    Humidade_summers += Humidade_springs
    #Pressao_summers += Pressao_springs
    Indice_summers += Indice_springs

    Temperatura_summers += Temperatura_autumns
    Humidade_summers += Humidade_autumns
    Indice_summers += Indice_autumns

    Temperatura_summers += Temperatura_winters
    Humidade_summers += Humidade_winters
    Indice_summers += Indice_winters
    
    training_data.append(Temperatura_summers)
    training_data.append(Humidade_summers)
    #training_data.append(Pressao_summers)
    training_data.append(Indice_summers)
    
    validation_data = []
    
    validation_data.append(Temperatura_winters)
    validation_data.append(Humidade_winters)
    #validation_data.append(Pressao_winters)
    validation_data.append(Indice_winters)

    test_data = []
    
    test_data.append(Temperatura_autumns)
    test_data.append(Humidade_autumns)
    #test_data.append(Pressao_autumns)
    test_data.append(Indice_autumns)
  
    #print("\n\n")
    #print(training_data)
    #print("\n\n")
    #print(validation_data)
    #print("\n\n")
    #print(test_data)
    #print("\n\n")
    #print(max_Temperatura_)
    #print(min_Temperatura_)
    #print(max_Humidade_)
    #print(min_Humidade_)
    #print(max_Indice_)
    #print(min_Indice_)
    
    return training_data,validation_data,test_data,max_Indice_,min_Indice_ 