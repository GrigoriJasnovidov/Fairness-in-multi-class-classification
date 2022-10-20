#!/usr/bin/env python
# coding: utf-8

# In[128]:


## general imports
import pandas as pd
import numpy as np
import random
import sklearn
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

#this functions creates a synthetic dataset 
def synthetic_dataset(size = 1000, influence = True):
    '''
    size - number of observations 
    influence - if True, then the dependence between the sensitive attribute and label is imposed. If False is chosen, 
                than the sensitieve attribute is independent of the label.  
    '''

    def simple_splitter(arr):
        arr_unchanged = arr.copy()
        arr = np.sort(np.array(arr))
        l = len(arr)
        n1 = arr[int(l/3)]
        n2 = arr[int(2*l/3)]
        result = []
        for i in range(l):
            if arr_unchanged[i] <= n1:
                result.append(0)
            elif (arr_unchanged[i]>n1 and arr_unchanged[i] <= n2):
                result.append(1)
            else:
                result.append(2)
        result = np.array(result)
        return result

    attr = np.random.choice([0,1],size = size)
    error_x = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_y = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_z = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_target = np.random.normal(loc=0.0, scale=0.5, size=size)

    y1 = np.random.normal(loc= 1, scale = 1, size = size)
    y2 = np.random.normal(loc= 1, scale = 1, size = size)
    y3 = np.random.normal(loc= 1, scale = 1, size = size)

    x = y1+y2+error_x
    y = y1+y3+error_y
    z = y2+y3+error_z

    if influence == True:
        target = x*(1+attr)+y*(1-0.25*attr)+z*(1-0.75*attr)+error_target*attr
    if influence == False:
        target = x+y+z+error_target
    
    target = simple_splitter(target)

    synthetic_df = pd.DataFrame(np.array((x,y,z,attr,target))).T.rename(columns = {0:'x',1:'y',2:'z',3:'attr',4:'target'})  
    
    return synthetic_df


# this function creates conditional use accuracy equality metric
def cuae(y_true,y_pred, sensitive_features):    
    '''
    y_true - stands for the true label
    y_pred - a forecast
    sensitive_features - sensitive attribute
    '''
    true = np.array(y_true)
    pred = np.array(y_pred)
    protected = np.array(sensitive_features)
    df = pd.DataFrame({'true':true, 'pred':pred, 'protected':protected}).astype('category')
    classes = df['true'].drop_duplicates()
    protected_groups_values = df['protected'].drop_duplicates()  
    np_ans = np.zeros(shape = [len(protected_groups_values),len(classes)])
    for j in range(len(protected_groups_values)):
        for i in range(len(classes)):
            protected_value = protected_groups_values[protected_groups_values.index[j]]
            current_part = df[df['protected']==protected_value]
            ndf = current_part[(current_part['true'] == classes[classes.index[i]])]
            res = accuracy_score(ndf['true'],ndf['pred'])
            np_ans[j,i] = res
    df = pd.DataFrame(np_ans, columns = np.array(classes), index = np.array(protected_groups_values))
    
    max_diff = []
    max_ratio = []

    for i in df.columns:
        column = df[i]
        sort = np.array(column.sort_values())
        max_ratio.append(sort[-1]/sort[0])
        max_diff.append(sort[-1]-sort[0])
    max_diff = np.array(max_diff)
    max_ratio = np.array(max_ratio)
    total_diff = max_diff.max()
    total_ratio = max_ratio.max()
    global_max = df.max().max()
    global_min = df.min().min()
    variation = global_max - global_min
    
    ans = {'df':df,
           'diff': total_diff,
           'ratio': total_ratio,
            'variation': variation}
    return ans



# This is the main function. It solves the relabelling programm and gives the result in terms of equalized odds metric.
def fmclp(dataset, estimator, number_iterations = None, prefit = False, interior_classifier = 'rf',
        verbose = False, multiplier=1, random_state = None):
    
    #Below are a few axiulary functions. The first function converts arrays 
    #of [0,0,1,  0,1,0,   0,0,1, ..... ] types to usual classes [0,1,2,....].
    
    def zeros_ones_to_classes(x,length = 3):
        n = int(len(x)/length)
        l = []
        for i in range(n):
            z = x[i*length:i*length+length]
            l.append(z.argmax())
        return np.array(l, dtype=int)

    #Next function creates an answer. It gets array of predictions for the part where protected attribute is zero, 
    #an array of predictions where protected attribute is one and then a series that indicates the order of this 
    #observations. Than it merges these arrays to get an answer.
    
    def answer_creator(x,y,grouper):
        x = np.array(x)         #array of 1
        y = np.array(y)         #array of 0
        grouper = np.array(grouper)
        ans = []
        x_ind = 0
        y_ind = 0
        l = len(grouper)
        for i in range(l):
            if grouper[i] == 0:
                ans.append(y[y_ind])
                y_ind+=1
            else:
                ans.append(x[x_ind])
                x_ind +=1
        return np.array(ans)
    
    #The function below collects all neccessary information about the initial ml model
    def ml_model(df,random_state = None, estimator = LGBMClassifier(), prefit = False):

        y = df.drop('target',axis=1)
        x = df['target']
    
        y_train,y_test,x_train,x_test = train_test_split(y,x, random_state = random_state)
        if prefit ==False:
            estimator.fit(y_train,x_train)
        estimator_pred= estimator.predict(y_test)
        accuracy_estimator = accuracy_score(estimator_pred,x_test)
 
        zero_train_features = y_train[y_train['attr']==0]
        one_train_features = y_train[y_train['attr']==1]
        zero_train_labels = x_train[zero_train_features.index]
        one_train_labels = x_train[one_train_features.index]
    
        zero_test_features = y_test[y_test['attr']==0]
        one_test_features = y_test[y_test['attr']==1]
    
        zero_total = zero_train_features.shape[0]
        one_total = one_train_features.shape[0]
    
        one_ratio = one_total/(one_total+zero_total)
        zero_ratio = zero_total/(one_total+zero_total)
        group = int(np.sqrt(one_total+zero_total))
        one_group = int(one_ratio*group)
        zero_group = int(zero_ratio*group)
    
        one_train_probs = pd.DataFrame(estimator.predict_proba(one_train_features)).rename(
            columns = {0:'zero_class',1:'first_class', 2:'second_class'})
        one_train_probs['label'] = np.array(one_train_labels)
    
        zero_train_probs = pd.DataFrame(estimator.predict_proba(zero_train_features)).rename(
            columns = {0:'zero_class',1:'first_class', 2:'second_class'})
        zero_train_probs['label'] = np.array(zero_train_labels)
   
        one_test_probs = pd.DataFrame(estimator.predict_proba(one_test_features)).rename(
            columns = {0:'zero_class',1:'first_class', 2:'second_class'})
    
        zero_test_probs = pd.DataFrame(estimator.predict_proba(zero_test_features)).rename(
            columns = {0:'zero_class',1:'first_class', 2:'second_class'})

        d_ans = {'dataset': df,
                 'estimator': estimator,
                 'y_train':y_train,
                 'y_test': y_test,
                 'x_train': x_train,
                 'x_test': x_test,
                 'predictions':  estimator_pred,
                 'estimator_accuracy': accuracy_estimator,
                 'group': group,
                 'one_group': one_group,
                 'zero_group': zero_group,
                 'one_train_probs': one_train_probs,
                 'zero_train_probs': zero_train_probs,
                 'one_test_probs': one_test_probs,
                 'zero_test_probs': zero_test_probs
                 }
        return d_ans
    
    
    #The function below is the core of our approach. It solves the linear programm and force the classifier to be fair.
    
    def lp_solver(d, number_iterations = 10, classifier = RandomForestClassifier(), verbose = False, multiplier = 1,
             random_state = None):

        group = multiplier*d['group']
        one_group = multiplier*d['one_group']
        zero_group = multiplier*d['zero_group']

        one_train_probs = d['one_train_probs']
        zero_train_probs = d['zero_train_probs']   
    
        bounds = []
        for i in range(3*one_group+3*zero_group):
            bounds.append((0,1))

        equation_vector = [1]*(one_group+zero_group)
        for i in range(3):
            equation_vector.append(0)

        equation_matrix0 = np.zeros((one_group+zero_group,3*one_group+3*zero_group))
        for i in range(one_group+zero_group):
            equation_matrix0[i,3*i] = 1
            equation_matrix0[i,3*i+1] = 1
            equation_matrix0[i,3*i+2] = 1
        equation_matrix0 = np.array(equation_matrix0)

        equation_vector = [1]*(one_group+zero_group)
        for i in range(3):
            equation_vector.append(0)

        one_predictor_array =[]
        zero_predictor_array = []

        if verbose:
            print('Start fitting')
        for k in range(number_iterations):
            if random_state == None:
                one_sample = d['one_train_probs'].sample(one_group)
                zero_sample = d['zero_train_probs'].sample(zero_group)
            else:
                one_sample = d['one_train_probs'].sample(one_group, random_state = k)
                zero_sample = d['zero_train_probs'].sample(zero_group, random_state = k)
            #I0, I1, I2 labels:
            I0 = one_sample[one_sample['label']==0]
            I1 = one_sample[one_sample['label']==1]
            I2 = one_sample[one_sample['label']==2]

            #J0, J1, J2 labels:
            J0 = zero_sample[zero_sample['label']==0]
            J1 = zero_sample[zero_sample['label']==1]
            J2 = zero_sample[zero_sample['label']==2]

            lenI0 = len(I0)
            lenI1 = len(I1)
            lenI2 = len(I2)
            lenJ0 = len(J0)
            lenJ1 = len(J1)
            lenJ2 = len(J2)

            vectorI0 = [ ]
            vectorI1 = [ ]
            vectorI2 = [ ]
            for i in one_sample.index:
                if i in I0.index:
                    vectorI0.append(lenJ0)
                    vectorI0.append(0)
                    vectorI0.append(0)
                else:
                    vectorI0.append(0)
                    vectorI0.append(0)
                    vectorI0.append(0)
            for i in one_sample.index:
                if i in I1.index:
                    vectorI1.append(0)
                    vectorI1.append(lenJ1)
                    vectorI1.append(0)
                else:
                    vectorI1.append(0)
                    vectorI1.append(0)
                    vectorI1.append(0)
            for i in one_sample.index:
                if i in I2.index:
                    vectorI2.append(0)
                    vectorI2.append(0)
                    vectorI2.append(lenJ2)
                else:
                    vectorI2.append(0)
                    vectorI2.append(0)
                    vectorI2.append(0)
            vectorI0 = np.array(vectorI0)
            vectorI1 = np.array(vectorI1)
            vectorI2 = np.array(vectorI2)

            vectorJ0 = [ ]
            vectorJ1 = [ ]
            vectorJ2 = [ ]

            for i in zero_sample.index:
                if i in J0.index:
                    vectorJ0.append(-lenI0)
                    vectorJ0.append(0)
                    vectorJ0.append(0)
                else:
                    vectorJ0.append(0)
                    vectorJ0.append(0)
                    vectorJ0.append(0)
            for i in zero_sample.index:
                if i in J1.index:
                    vectorJ1.append(0)
                    vectorJ1.append(-lenI1)
                    vectorJ1.append(0)
                else:
                    vectorJ1.append(0)
                    vectorJ1.append(0)
                    vectorJ1.append(0)
            for i in zero_sample.index:
                if i in J2.index:
                    vectorJ2.append(0)
                    vectorJ2.append(0)
                    vectorJ2.append(-lenI2)
                else:
                    vectorJ2.append(0)
                    vectorJ2.append(0)
                    vectorJ2.append(0)
            vectorJ0 = np.array(vectorJ0)
            vectorJ1 = np.array(vectorJ1)
            vectorJ2 = np.array(vectorJ2)

            row0 =  np.concatenate((vectorI0, vectorJ0)).reshape(1,-1)
            row1 =  np.concatenate((vectorI1, vectorJ1)).reshape(1,-1)
            row2 =  np.concatenate((vectorI2, vectorJ2)).reshape(1,-1)
            rows = np.concatenate((row0,row1,row2),axis=0)

            equation_matrix = np.concatenate((equation_matrix0,rows),axis=0)

            C = np.array(one_sample[['zero_class', 'first_class', 'second_class']]).ravel()
            B = np.array(zero_sample[['zero_class', 'first_class', 'second_class']]).ravel()
            objective = (-1)*np.concatenate((C,B))
            array = linprog(
                    c = objective, A_ub=None, b_ub=None, 
                       A_eq=equation_matrix, 
                       b_eq=equation_vector, 
                    bounds=bounds, method='highs-ipm', callback=None, options=None, x0=None).x

            fair_pred = zeros_ones_to_classes(array)
            fair_pred_one = fair_pred[:one_group]
            fair_pred_zero = fair_pred[one_group:]

            # here we prepare classes to relabeling
            one_df = pd.DataFrame(one_sample, columns = ['zero_class', 'first_class','second_class'])     
            one_predictor = classifier
            one_predictor.fit(one_df, fair_pred_one)
            one_predictor_array.append(one_predictor)
        
            zero_df = pd.DataFrame(zero_sample, columns = ['zero_class', 'first_class','second_class'])    
            zero_predictor = classifier
            zero_predictor.fit(zero_df,fair_pred_zero)
            zero_predictor_array.append(zero_predictor)
            if verbose:
                print(k+1)
        ans = {'one_predictor_array':one_predictor_array,
               'zero_predictor_array': zero_predictor_array}
        if verbose:
            print('Fitting is finished')
        return ans
    
    #The function below is used for forecasting.
    
    def predictor(solved,d,verbose=False):
        if verbose:
            print('Predicting in process')
        one_predictor_array = solved['one_predictor_array']
        zero_predictor_array = solved['zero_predictor_array']

        one_probs = d['one_test_probs']
        zero_probs = d['zero_test_probs']
        one_rows = one_probs.shape[0]
        zero_rows = zero_probs.shape[0]
        one_cols = len(one_predictor_array)
        zero_cols = len(zero_predictor_array)

        one_final_array = np.empty(shape = (one_cols,one_rows))
        for i in range(one_cols):
            one_final_array[i] = one_predictor_array[i].predict(one_probs)
        one_final_array = pd.DataFrame(one_final_array)

        one_final_ans = []
        for i in range(one_rows):
            one_final_ans.append(one_final_array[i].value_counts().sort_values(ascending = False).index[0])
        
        zero_final_array = np.empty(shape = (zero_cols,zero_rows))
        for i in range(zero_cols):
            zero_final_array[i] = zero_predictor_array[i].predict(zero_probs)
        zero_final_array = pd.DataFrame(zero_final_array)

        zero_final_ans = []
        for i in range(zero_rows):
            zero_final_ans.append(zero_final_array[i].value_counts().sort_values(ascending = False).index[0])
    
        preds = answer_creator(one_final_ans, zero_final_ans, d['y_test']['attr'] )
    
        ans = {'one_preds': one_final_ans,
               'zero_preds': zero_final_ans,
               'preds': preds}
        if verbose:
            print('Predicting is finished')
       
        return ans
    
    
    #Now are ready to implement our algorithm.
    
    interior_classifier_dict = {'rf':RandomForestClassifier(random_state  = random_state),
                                'lr': LogisticRegression(),
                                'dt': DecisionTreeClassifier(random_state = random_state),
                                'svm':SVC(),
                                'lgb':LGBMClassifier(),
                                'knn':KNeighborsClassifier(n_neighbors=3)}
    
    model = ml_model(df = dataset, estimator = estimator, random_state = random_state, prefit = prefit)
    solved = lp_solver(model,
                       classifier =  interior_classifier_dict[interior_classifier],
                       number_iterations=number_iterations,
                       verbose = verbose, 
                       multiplier=multiplier,
                       random_state = random_state)
    pred = predictor(solved, model,verbose)
    fair_cuae = cuae(y_true = model['x_test'], y_pred = pred['preds'], 
                                           sensitive_features = model['y_test']['attr'])
    fair_accuracy = accuracy_score(pred['preds'],model['x_test'])
    
    ans = {'accuracy_of_initial_classifier': model['estimator_accuracy'],
           'fairness_of_initial_classifier': cuae(y_true = model['x_test'], y_pred = model['predictions'],
                                                                    sensitive_features = model['y_test']['attr']),
           'solved' : solved,
           'predictions': pred,
           'fairness_of_fair_classifier': fair_cuae,
           'accuracy_of_fair_classifier':fair_accuracy,
           'dataset':dataset,
           'model':model,
           'multiplier':multiplier,
           'interior_classifier':interior_classifier
           }
    
    return ans

# example 

if __name__ == '__main__':
    d = synthetic_dataset(400)
    dataset = synthetic_dataset(20000)    
    random_state = 78
    cl = LGBMClassifier(random_state= random_state)
    y = d.drop('target',axis=1)
    x = d['target']
    y_train,y_test,x_train,x_test = train_test_split(y,x, random_state = random_state)
    cl.fit(y_train,x_train)
    res = fmclp(dataset = dataset, 
           estimator = cl, 
           number_iterations = 10, 
           prefit = True, 
           interior_classifier = 'knn',
           verbose = True, 
           multiplier=30, 
           random_state = random_state)

