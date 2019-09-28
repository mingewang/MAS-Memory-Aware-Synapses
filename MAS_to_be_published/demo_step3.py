from MAS import *
from MNIST_split import *
import os

#MIMIC the case when samples from the previous tasks are seen in each step
from MAS import *
all_digits=[[3,4],[5,6],[7,8],[9,0]]
reg_sets=[]
dataset_path='data/Pytorch_MNIST_dataset//split12_dataset.pth.tar'

exp_dir='exp_dir/SGD_MNIST_NET12'
pevious_pathes=[]
reg_lambda=1
for digits in all_digits:
    reg_sets.append(dataset_path)
    model_path=os.path.join(exp_dir,'best_model.pth.tar')
    pevious_pathes.append(model_path)
    dlabel=''
    for i in digits:
        dlabel=dlabel+str(i)

    dataset_path='data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar'

    exp_dir='exp_dir/SGD_MNIST_NET'+dlabel
   
    num_epochs=20
    data_dirs=None

    # now learning a new task with all previous tasks' model/dataset
    MAS_sequence(dataset_path=dataset_path,pevious_pathes=pevious_pathes,previous_task_model_path=model_path,exp_dir=exp_dir,data_dirs=data_dirs,reg_sets=reg_sets,reg_lambda=reg_lambda,batch_size=200,num_epochs=num_epochs,lr=1e-2,norm='L2',b1=False)
