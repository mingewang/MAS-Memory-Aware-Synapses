from MAS import *
from MNIST_split import *
import os

#estimate forgetting
all_digits=[[1,2],[3,4],[5,6],[7,8],[9,0]]
average_forgetting=0
exp_dir='exp_dir/SGD_MNIST_NET12'
from Test_sequential  import *
for digits in all_digits:
    dlabel=''
    for i in digits:
        dlabel=dlabel+str(i)
    if all_digits.index(digits)>0:
        exp_dir='exp_dir/SGD_MNIST_NETACC'+dlabel
    previous_model_path=os.path.join(exp_dir,'best_model.pth.tar')
    exp_dir='exp_dir/SGD_MNIST_NETACC90'
    dataset_path='data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar'
    current_model_path=os.path.join(exp_dir,'best_model.pth.tar')
    acc2=test_seq_task_performance(previous_model_path=previous_model_path,current_model_path=current_model_path,dataset_path=dataset_path)
    acc1=test_model(previous_model_path,dataset_path)
    forgetting=acc1-acc2
    average_forgetting=average_forgetting+forgetting
average_forgetting=average_forgetting/len(all_digits)    
print(average_forgetting)
