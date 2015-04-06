"""
Title: Single Hidden layered Gradient descent based Neural Network
Author: Bishnu Sarker, Universite Pierre et Marie Curie, Paris 6, France.
Email: bishnukuet@gmail.com, bishnu.sarker@etu.upmc.fr
web: www.sites.google.com/site/bsarkerbandhan
"""

import numpy as np
import math
import decimal
class DataIteration:
    def __init__(self,data_size=None,train_size=None,test_size=None,arity=None):   #train_size and test_size parameter is not useful in this version
        self.data_size=data_size
        self.train_size=train_size
        self.test_size=test_size
        self.test_end_size=0
        self.arity=arity
        self.current_index=0
        self.train_index=0
        self.test_index=self.train_index+1
        self.dataset=list()
        
    def reset_index(self):
        pass
        
    def next_instance(self):
        
        T= self.dataset[self.current_index]
        self.current_index=self.current_index+1
        return T
    def next_train_instance(self):
        
        T= self.dataset[self.train_index]
        self.train_index=self.train_index+1
        return T
    
    def next_test_instance(self):
        
        T= self.dataset[self.test_index]
        self.test_index=self.test_index+1
        return T 
    def has_next(self):
        
        return self.data_size-self.current_index
    def next_k_fold_test_instance(self):
        T= self.dataset[self.test_index]
        self.test_index=self.test_index+1
        #print(self.test_index)
        return T
    def next_k_fold_train_instance(self):
        
        T= self.dataset[self.train_index]
        self.train_index=self.train_index+1
        return T
    def has_train_next(self):
        
        return self.train_size-self.train_index

    def has_test_next(self):
        
        return self.data_size-self.test_index

    def has_k_fold_train_next(self):

        if self.train_index==self.test_index:
                 
            self.train_index=self.test_end_size+1
            
        #print(self.data_size,self.train_index)
        return (self.data_size-self.train_index)
    def has_k_fold_test_next(self):
        return self.test_end_size-self.test_index
    def read_dataset(self,filename):
        self.dataset=np.loadtxt(filename)
        shape=self.dataset.shape
        self.data_size=shape[0]
        
        self.arity=shape[1]
        self.normalize_data()
        self.train_size=math.floor(self.data_size*0.80)
        self.train_index=0
        self.test_size=self.data_size-self.train_size
        self.test_index=self.train_size
        
    def make_dataset(self,data):
        self.dataset=data
        
    def normalize_data(self):
        r=self.data_size
        c=self.arity-1
        mn=[[max(self.dataset[:,j]),min(self.dataset[:,j])] for j in range(c)]
        for i in range(r):
            for j in range(c):
                self.dataset[i][j]=(self.dataset[i][j]-mn[j][1])/(mn[j][0]-mn[j][1])
    
    
class ANN:

    def __init__(self,ni,nh,no,in_vector=None,out_vector=None,nu=0.5, alpha=0.1):
        np.random.seed(2)
        self.alpha=alpha #momentum
        self.nu=nu #learning rate
        self.in_vector=in_vector
        self.out_vector=out_vector
        self.no_of_input_nodes=ni
        self.no_of_output_nodes=no;
        self.no_of_hidden_nodes=nh
        self.input_values=[1.0]*ni
        self.output_values=[1.0]*no
        self.hidden_values=[1.0]*nh
        self.weights_input_2_hidden=[ list(j*2.0-1.0 for j in np.random.ranf(nh)) for i in range(ni)]#[[w11,w12,w13][w21,w22,w23]]
        self.weights_hidden_2_output=[ list( j*2.0-1.0 for j in np.random.ranf(no)) for i in range(nh)]  #[[w11,w12],[w21,w22]]
        #self.weights_input_2_hidden=[[0.1,0.4],[0.8,0.6]]
        #self.weights_hidden_2_output =[[0.3],[0.9]]
        self.delta_output=[1.0]*no
        self.delta_hidden=[1.0]*(nh)
        self.error=[0.0]*no
        
    def __print__(self):
        pass
    def reset_parameter(self):
        np.random.seed(2)
        self.input_values=[1.0]*self.no_of_input_nodes
        self.output_values=[1.0]*self.no_of_output_nodes
        self.hidden_values=[1.0]*self.no_of_hidden_nodes
        self.weights_input_2_hidden=[ list(j*2.0-1.0 for j in np.random.ranf(self.no_of_hidden_nodes)) for i in range(self.no_of_input_nodes)]#[[w11,w12,w13][w21,w22,w23]]
        self.weights_hidden_2_output=[ list( j*2.0-1.0 for j in np.random.ranf(self.no_of_output_nodes)) for i in range(self.no_of_hidden_nodes)]  #[[w11,w12],[w21,w22]]
        #self.weights_input_2_hidden=[[0.1,0.4],[0.8,0.6]]
        #self.weights_hidden_2_output =[[0.3],[0.9]]
        self.delta_output=[1.0]*self.no_of_output_nodes
        self.delta_hidden=[1.0]*self.no_of_hidden_nodes
        self.error=[0.0]*self.no_of_output_nodes
        
    def set_input(self,T):
        self.in_vector=T[:self.no_of_input_nodes]
        self.out_vector=T[self.no_of_input_nodes:len(T)]
        
            
    def compute_hidden_nodes(self):
        #print(self.input_values)
        for i in range(self.no_of_hidden_nodes):
            lc=0.0
            for j in range(self.no_of_input_nodes): #
                lc=lc+self.weights_input_2_hidden[j][i]*self.in_vector[j]
                #print(self.weights_input_2_hidden[j][i],self.in_vector[j])
            #print(lc)
            self.hidden_values[i]=self.sigma(lc)
            #print(self.hidden_values[i],'\n')
        return self.hidden_values[:]
    def compute_output_nodes(self):
        #print(self.hidden_values)
        for i in range(self.no_of_output_nodes):
            lc=0.0
            for j in range(self.no_of_hidden_nodes):
                lc=lc+self.weights_hidden_2_output[j][i]*self.hidden_values[j]
                #print(self.weights_hidden_2_output[j][i],self.hidden_values[j])
            #print(lc,'\n')
            self.output_values[i]=self.sigma(lc)
            #print(self.output_values[i])
        return self.output_values[:]

    def get_square_error(self):
        sqe=0.0
        for er in self.error:
            sqe=sqe+er*er
        return sqe
    def forward_pass(self):
        
        self.compute_hidden_nodes()
        #print(self.hidden_values)
        self.compute_output_nodes()
        #print(self.output_values)
        
        
    def update_hidden2output(self):
        for i in range(self.no_of_output_nodes):
            self.error[i]=self.out_vector[i]-self.output_values[i]
            
            self.delta_output[i]=self.error[i]*self.derivative_sigma(self.output_values[i])#self.output_values[i]*(1.0-self.output_values[i])
        #print(self.error)
        
    def update_input2hidden(self):
        sm=[0.0]*self.no_of_hidden_nodes
        for i in range(self.no_of_hidden_nodes):
            err=0.0
            err=self.derivative_sigma(self.hidden_values[i])#self.hidden_values[i]*(1-self.hidden_values[i])
            for j in range(self.no_of_output_nodes):
                sm[i]=sm[i]+self.delta_output[j]*self.weights_hidden_2_output[i][j]    

            self.delta_hidden[i]=err*sm[i]
            
        
    def update_hidden_weights(self):
        for i in range(self.no_of_hidden_nodes):
            for j in range(self.no_of_output_nodes):

                self.weights_hidden_2_output[i][j]=self.weights_hidden_2_output[i][j]+self.nu*self.delta_output[j]*self.hidden_values[i]
        
    def update_input_weights(self):
        for i in range(self.no_of_input_nodes):
            for j in range(self.no_of_hidden_nodes):
                self.weights_input_2_hidden[i][j]=self.weights_input_2_hidden[i][j]+self.nu*self.delta_hidden[j]*self.in_vector[i]
        
    def backward_pass(self):
        
        self.update_hidden2output()
        self.update_input2hidden()
        self.update_hidden_weights()
        self.update_input_weights()
            
    def print_param(self):
        #print("Input to hidden weights:\n")
        print(self.weights_input_2_hidden)
        #print("Hidden to Output Weights:\n")
        print(self.weights_hidden_2_output)
    def sigma(self,x):
        
        return 1.0/(1.0+math.exp(-x))
    def derivative_sigma(self,dv):
        #dv=sigma(x)
        return dv*(1.0-dv)
        
    def delta_error(self,fx,y):
        return y-fx
    def train_network (self,Dt,threshold_error,iteration=100):
        
        for i in range(iteration):
            
            error=0.0
            while Dt.has_k_fold_train_next():
                T=Dt.next_k_fold_train_instance()
                self.set_input(list(T))
                self.forward_pass()
                #print(T[2]-self.output_values[0])
                self.backward_pass()
                error=error+self.get_square_error()
                #print("Training")
                #print(error)
            #print("Iteration:",i,"Error:",0.5*error,"\n")
            
            #total_error=
            #if (0.5*error)<=threshold_error:
             #   print("No of iteration performed before the threshold achieved is:",i)
              #  break
            Dt.train_index=0
           # print("Training")
    def test_network(self,Dt):
        #Dt.test_index=0
        well_class=0
        miss_class=0
        while Dt.has_k_fold_test_next():
            T=Dt.next_k_fold_test_instance()
            self.set_input(list(T))
            self.forward_pass()
            if self.no_of_output_nodes>1:
                print("Output > 1")
                if abs(self.output_values.index(max(self.output_values))- self.out_vector.index(max(self.out_vector))):
                    print(T,"->\n",self.output_values,"<->",self.out_vector,"Miss Classified", "\n\n\n")
                    miss_class=miss_class+1
                else:
                    print(T,"->\n",self.output_values,"<->",self.out_vector,"Well Classified", "\n\n\n")
                    well_class=well_class+1
            else:
                #print("Output <1")
                print(Dt.test_index-1)
                print(T,"->\n",self.output_values,"<->",self.out_vector,"\n\n\n")
                if round(self.output_values[0],1)>=0.5:
                    out=1
                else:
                    out=0
                    
                if abs(self.out_vector[0]-out)==0:
                    well_class=well_class+1
                else:
                    miss_class=miss_class+1
                    
            
                
                
            
        #print("classified:",well_class,"miss classified:",miss_class,'\n\n')
        #print("classification rate:", (well_class/(well_class+miss_class)))
        rate=(well_class/(well_class+miss_class))
        return  rate       
    def cross_validate_NN(self,fold=10):
        Dt=DataIteration()
        Dt.read_dataset("Diabetic.txt")
        fold_range=math.floor(Dt.data_size/fold)
        Error=list()    
        for i in range(fold):
            print("Fold",i)
            Dt.test_index=i*fold_range
            print("Test Index: ",Dt.test_index)
            Dt.test_end_size=Dt.test_index+fold_range -1
            print("Test End Index: ",Dt.test_end_size)
            Dt.train_index=0
            print("train",i)
            Dt.train_size=Dt.data_size
            print("Test",i)
            self.train_network(Dt,0.001,100)
            err=self.test_network(Dt)
            Error.append(err)
            print("Fold:",i,"Finished")
            self.reset_parameter()
            
        print("The outcome of Cross Validation is as Follows:\n")
        print(Error)
        print(sum(Error)/len(Error))
            
    def neural_resonance(self,threshold_error=0.00001):
        Dt=DataIteration()
        Dt.read_dataset("Diabetic.txt")
        print("-----------Dataset Information--------------\n")
        print("Data size:",Dt.data_size,"Arity:",Dt.arity,'\n')
        print("---------------------------------------------\n\n")
        print("Training Network Information:\n")
        print("Train Size:",Dt.train_size,"Test Size:",Dt.test_size,'\n')
        self.train_network(Dt,threshold_error,100)

        print("--------------Training Ends.------------\n\n\ -----------------Testing Starts-------------------------\n\n")
        
        
        self.test_network(Dt)
        print("----------------Testing Ends----------------------\n\n\n")
        

#Starts the progam run
for i in range(5):        
    ann=ANN(62,132+i,1,0.8,0.2)

#ann.neural_resonance(0.0000001)
    ann.cross_validate_NN()
#ann.print_param()
#print(ann.weights_hidden_2_output,'\n\n')
#print(ann.weights_input_2_hidden,'\n\n')





