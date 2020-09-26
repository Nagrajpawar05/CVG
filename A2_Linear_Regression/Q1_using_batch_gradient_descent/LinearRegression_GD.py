import numpy as np
class LinearRegression:
    def __init__(self,lr,num_iter,adp_lr,ep):
     
        self.lr=lr
        self.num_iter=num_iter
        self.adp_lr=adp_lr
        self.ep=ep
    
    def initialize_with_zeros(self):
        theta=[1,0]
        return theta
    
    def h(self,X,theta):
        return ((theta[1]*X+theta[0]))
    
    def Normalize(self,X):
        X = (X - np.min(X))/(np.max(X)-np.min(X))
        return X
    def _indexing(self,x, indices):
        return [x[idx] for idx in indices]
    
    def train_test_split(self,X,y,test_size=0.2,shuffle=True,random_seed=1):
        length = X.shape[0]
        len_test = int(np.ceil(length*test_size))
        len_train = length - len_test
        
        
        
        if shuffle:
            perm = np.random.RandomState(random_seed).permutation(length)
            test_indices = perm[:len_test]
            train_indices = perm[len_test:]
            return (self._indexing(X,train_indices),self._indexing(y,train_indices),self._indexing(X,test_indices),self._indexing(y,test_indices))
        else:
            return (X[:len_train],y[:len_train],X[len_train:],y[len_train])
            
        
    
    def cost(self,theta,X,y):
        m = y.size
        
        h=self.h(X,theta)
        
        J = (1/(2*m))*np.sum(np.square(h-y))
        
        return J

    def grad(self,theta,X,y):
        m=X.size
        dJ_theta = [0]*2
        
        dJ_theta[0] = (1/m) * np.sum(self.h(X,theta)-y)
        dJ_theta[1] = (1/m) * np.sum((self.h(X,theta)-y)*X)
        return dJ_theta
    
    def gradientDescent(self,X,y,theta,theta_prev):
        print(self.cost(theta_prev,X,y))
        j=0
        J=[]
        t0=[]
        t1=[]
        m=X.size
        pre_loss=0
        for i in range(self.num_iter):
            theta_prev= theta
            dJ_theta=self.grad(theta_prev,X,y)
            t=i+1
            if not self.adp_lr:
                theta0 = theta_prev[0] - self.lr*dJ_theta[0]
                theta1 = theta_prev[1] - self.lr*dJ_theta[1]
            else:
                theta0 = theta_prev[0] - ((self.lr)/np.sqrt(t))*dJ_theta[0]
                theta1 = theta_prev[1] -  ((self.lr)/np.sqrt(t))*dJ_theta[1]
                
            t0.append(theta0)
            t1.append(theta1)
            theta = [theta0,theta1]
            
            loss=self.cost(theta,X,y)
            
            J.append(loss)
            if(i%100==0):
                print("after {} iterations, loss = {}".format(i,loss))
                
            if np.abs(pre_loss - loss) < 1e-12:
                print("convergence criteria at met {} iteration ...STOP training".format(i))
                break
            pre_loss=loss
            
        return (theta,J,t0,t1)
            
            
class Gradient_Descent:
    def __init__(self,lr_list=[]):
        self.lr_list=lr_list

    def prediction_function(self,x,theta,theta0):
        return theta*x + theta0

    def loss_function(self,x,y,theta,theta0,prediction_function):
        m = len(y)
        return (1/2*m)*np.sum((y-self.prediction_function(x,theta,theta0))**2)

    def compute_gradient_descent(self,theta,theta0,x,y,lr):
        
        m = len(y)
        
        theta_derivative = (1/m)*np.sum(x*((theta*x+theta0)-y))
        theta0_derivative = (1/m)*np.sum((theta*x+theta0)-y)
        
        theta = theta - theta_derivative*lr
        theta0 = theta0 - theta0_derivative*lr
        
        return theta,theta0   
            
            
        
        

        
    
