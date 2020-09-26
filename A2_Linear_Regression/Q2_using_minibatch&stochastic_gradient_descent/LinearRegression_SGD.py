import numpy as np
class LinearRegression:
    def __init__(self,lr,num_iter,batch_size):
     
        self.lr=lr
        self.num_iter=num_iter
        self.batch_size=batch_size
        #self.ep=ep
        
    def generate_data(self,mu,sd,N):
        
        X1 = np.random.normal(loc=mu,scale=sd, size=N)
        X1 = np.array(X1)
        return X1

  
    
    def h(self,X,theta,b):
        y_pred= np.dot(X.T,theta)+b
        
        return y_pred
    
    def cost(self,theta,b,X,y):
        m = X.shape[1]
        
        h=self.h(X,theta,b)
        
        J = (1/(2*m))*np.sum(np.square(h-y))
        
        return J
    
    def initialize_and_vectorize(self,x1,x2):
        X=np.zeros((2,1000000))
        X[0,:]=x1
        X[1,:]=x2
        
        theta=np.zeros((2,1))
        b=1
        
        return (X,theta,b)
        
    def Normalize(self,X):
        X = (X - np.min(X))/(np.max(X)-np.min(X))
        return X
    

    def grad(self,theta,b,X,y):
        m=X.shape[1]
        dJ_theta = [0]*3
        y_pred=self.h(X,theta,b)
        #print("grad",y.shape)
        #print("x",X.shape)
        db = (1/m) * np.sum(y_pred-y)
        
        dtheta = (1/m) *(np.dot(X,(y_pred-y)))

        return (dtheta,db)
    

    def check_(self):
        print("working")
    def train(self,X,y,theta,b,theta_prev,b_prev):
        loss=[]
        t0=[]
        t1=[]
        t2=[]
        
        jj=0
        print(theta)
        pre_loss=0
        tt1_p=0
        tt2_p=0
        tt0_p=0
        for i in range (self.num_iter):
            
            batch_per_epoch=int(X.shape[1] / self.batch_size)
            XX = np.zeros((2,self.batch_size))
            #print(batch_per_epoch)
            shuffle_index = np.random.permutation(X.shape[1])
            
            for step in range(batch_per_epoch):
                
                theta_prev=theta
                b_prev=b
                
                
                batch_x1 = X[0,shuffle_index[step*self.batch_size:(step+1)*self.batch_size]]
                batch_x2= X[1,shuffle_index[step*self.batch_size:(step+1)*self.batch_size]]
                XX[0,:]=batch_x1
                
                XX[1,:]=batch_x2
                batch_y = y[shuffle_index[step*self.batch_size:(step+1)*self.batch_size]]
                
                J=self.cost(theta,b,XX,batch_y)
                #print("loss=",J)
                loss.append(J)
                
              
                    
                dJ_theta,db=self.grad(theta,b,XX,batch_y)
                
                theta0 = b_prev - self.lr*db
                theta1 = theta_prev - self.lr*dJ_theta
                theta=theta1
                b=theta0
                
                t0.append(theta0)
                t1.append(theta1)
                
                
                
                
                
            
                
                
            loss_ = self.cost(theta,b,X,y)
            
            
            if(i and X.shape[1]!=1000000):
                print("Loss after {} epoch = {} and pamameters= {} {}" .format(i,J,theta,b))
            if(i%100==0 and X.shape[1]==1000000):
                print("Loss after {} epoch = {} and pamameters= {} {}" .format(i,J,theta,b))
            #COnvergence Criteria
            
            if abs(loss_-pre_loss) < 0.01 and X.shape[1]!=1000000:
                print("convergence criteria met at {} epoch ...STOP training".format(i))
                break
          
            pre_loss=loss_
            if abs(float(dJ_theta[0]))<0.001 and abs(float(dJ_theta[1]))<0.001 and abs(float(db))<0.001 and X.shape[1]!=1000000 :
                print("convergence criteria met at {} epoch ...STOP training1".format(i))
                break
            tt0_p=b
            tt1_p=theta[0]
            tt2_p=theta[1]
            
        
        
        return(loss,t0,t1)
                
                
                
    
