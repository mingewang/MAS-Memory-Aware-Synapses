from MAS import *
from MNIST_split import *
import os

#Mimic the standard setup used when after each task the omega is compute on the training samples and accumelated
all_digits=[[3,4],[5,6],[7,8],[9,0]]
reg_sets=[]
dataset_path='data/Pytorch_MNIST_dataset//split12_dataset.pth.tar'

exp_dir='exp_dir/SGD_MNIST_NET12'

reg_lambda=1#0.5
for digits in all_digits:
    reg_sets=[]
    reg_sets.append(dataset_path)
    model_path=os.path.join(exp_dir,'best_model.pth.tar')
    
    dlabel=''
    for i in digits:
        dlabel=dlabel+str(i)

    dataset_path='data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar'

    exp_dir='exp_dir/SGD_MNIST_NETACC'+dlabel
  
    num_epochs=20
    data_dir=None

    if all_digits.index(digits)>0:
        MAS_Omega_Acuumelation(dataset_path,previous_task_model_path=model_path,exp_dir=exp_dir,data_dir=data_dir,reg_sets=reg_sets,reg_lambda=reg_lambda,norm='L2', num_epochs=num_epochs,lr=0.1e-2,batch_size=200,b1=False)
    else:
        MAS(dataset_path,previous_task_model_path=model_path,exp_dir=exp_dir,data_dir=data_dir,reg_sets=reg_sets,reg_lambda=reg_lambda,norm='L2', num_epochs=num_epochs,lr=0.1e-2,batch_size=200,b1=False)
