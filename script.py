import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import pow
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    maxVal = int(np.max(y))
    minVal = int(np.min(y))
    means = np.zeros((0,X.shape[1]))
    for i in range(minVal,maxVal+1):
        temp = np.zeros((0,X.shape[1]))
        for j in range(0,X.shape[0]):
            if int(y[j,0]) == i:
                t = np.matrix(X[j,:])
                temp = np.concatenate((temp,t),axis=0)
        temp = np.sum(temp,axis=0)/(temp.shape[0])
        means = np.concatenate((means,temp),axis=0)
        
    covmat = np.cov(np.transpose(X))
    means = np.transpose(means)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    maxVal = int(np.max(y))
    minVal = int(np.min(y))
    means = np.zeros((0,X.shape[1]))
    list = []
    for i in range(minVal,maxVal+1):
        temp = np.zeros((0,X.shape[1]))
        for j in range(0,X.shape[0]):
            if int(y[j,0]) == i:
                t = np.matrix(X[j,:])
                temp = np.concatenate((temp,t),axis=0)
        temp1 = np.sum(temp,axis=0)/(temp.shape[0])
        covmat = np.cov(np.transpose(temp))
        list.append(covmat)
        means = np.concatenate((means,temp1),axis=0)
        
    covmats = np.array(list)
    means = np.transpose(means)
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    prediction = np.zeros((0,1))
    inverse = np.linalg.inv(covmat)
    meansT = np.transpose(means)
    for i in range(0,Xtest.shape[0]):
        max = np.array([[-float('inf')]]);
        index = 0;
        for j in range(0,means.shape[1]):
            p = np.log(0.2)
            logPi = (Xtest.shape[0]*np.log(2*3.14))/2
            covD = 0.5*np.log(np.linalg.det(covmat))
            intCal = np.transpose(Xtest[i,:]-meansT[j,:])
            temp = p-logPi-covD-0.5*np.dot(np.dot(np.transpose(intCal),inverse),intCal)
            if temp > max:
                max = temp
                index = j+1
        prediction = np.concatenate((prediction,np.array([[index]])),axis=0)
    acc = 100*np.mean((prediction == ytest).astype(float))
    
    """
    plt.close("all")
    colors = iter(cm.rainbow(np.linspace(0,1,5)))
    plt.figure(1)
    for i in range(1,6):
        plt.scatter(Xtest[(prediction==i).reshape((prediction==i).size),0],Xtest[(prediction==i).reshape((prediction==i).size),1],color=next(colors))
    plt.title('QDA boundary plot')
    plt.show()
    """
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    prediction = np.zeros((0,1))
    #inverse = np.linalg.inv(covmat)
    meansT = np.transpose(means)
    for i in range(0,Xtest.shape[0]):
        max = np.array([[-float('inf')]]);
        index = 0;
        for j in range(0,means.shape[1]):
            p = np.log(0.2)
            logPi = (Xtest.shape[0]*np.log(2*3.14))/2
            covD = 0.5*np.log(np.linalg.det(covmats[j,:]))
            inverse = np.linalg.inv(covmats[j,:])
            intCal = np.transpose(Xtest[i,:]-meansT[j,:])
            temp = p-logPi-covD-0.5*np.dot(np.dot(np.transpose(intCal),inverse),intCal)
            if temp > max:
                max = temp
                index = j+1
        prediction = np.concatenate((prediction,np.array([[index]])),axis=0)
    acc = 100*np.mean((prediction == ytest).astype(float))
    
    """
    x = np.arange(-2,16,0.1)
    data_test = []
    for i in range(0,len(x)):
   	for j in range(0,len(x)):
  		data_test.append([x[i],x[j]])
   	pass
    pass		
    data_test = np.array(data_test)
    
    plt.close("all")
    colors = iter(cm.rainbow(np.linspace(0,1,5)))
    plt.figure(1)
    for i in range(1,6):
        plt.scatter(data_test[(prediction==i).reshape((prediction==i).size),0],data_test[(prediction==i).reshape((prediction==i).size),1],color=next(colors))
    plt.title('QDA boundary plot')
    plt.show()
    """
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD     
    a = np.dot(X.transpose(),X)
    b = np.linalg.inv(a)
    c = np.dot(b,X.transpose())
    w = np.dot(c,y)                                               
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    a = lambd*np.identity(X.shape[1])*X.shape[0]
    b= np.dot(X.transpose(),X)
    d=a+b
    e=np.linalg.inv(d)
    f=np.dot(e,X.transpose())
    w=np.dot(f,y)                                                 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    #w = w.flatten()
    a = (ytest-np.dot(Xtest,w))*(ytest-np.dot(Xtest,w))
    b = np.sum(a)
    c = sqrt(b)
    rmse = c/a.shape[0]
    #print rmse
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    
    w = w.reshape(65,1)
    #print w.shape  
    a= y-np.dot(X,w)
    error = ((0.5*np.dot(a.transpose(),a))/X.shape[0]) + 0.5*lambd* np.dot(w.transpose(),w)

    part1 = -np.dot(np.transpose(y),X)
    part2 = np.dot(np.transpose(w),np.dot(np.transpose(X),X))
    part3 = lambd*np.transpose(w)
    error_grad = ((part1+part2)/X.shape[0])+part3
    error_grad = error_grad.flatten()                                         
    #print error_grad
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (p+1))                                                         
    # IMPLEMENT THIS METHOD
    
    x = x.reshape(x.shape[0],1)
    Xd = np.zeros((0,p+1))
    
    for i in range(0,x.shape[0]):
        temp = np.zeros((1,p+1))
        for j in range(0,p+1):
            temp[0,j] = pow(x[i,0],j)
        Xd = np.concatenate((Xd , temp))
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('C:/Users/VidyadharReddy/Desktop/pa2/sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))


# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('C:/Users/VidyadharReddy/Desktop/pa2/diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_train = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_train = testOLERegression(w_i,X_i,y)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))
print('RMSE without intercept '+str(mle_train))
print('RMSE with intercept '+str(mle_i_train))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.clf()
plt.plot(lambdas,rmses3)
plt.savefig("C:/Users/VidyadharReddy/Desktop/pa2/ans3_test.png")
plt.clf()
plt.plot(lambdas,rmses_train)
plt.savefig("C:/Users/VidyadharReddy/Desktop/pa2/ans3_train.png")
# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    #print w_l_1
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.clf()
#print rmses4
plt.plot(lambdas,rmses4)
plt.savefig("C:/Users/VidyadharReddy/Desktop/pa2/ans4_1.png")


# Problem 5
pmax = 7
lamda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lamda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
print rmses5
plt.clf()
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.savefig("C:/Users/VidyadharReddy/Desktop/pa2/ans5_1.png")

