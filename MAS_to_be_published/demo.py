from MAS import *
from MNIST_split import *
import os

# prepare dir
os.system("mkdir -p data/Pytorch_MNIST_dataset")

# download MNIST to data/processed, data/raw
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
batch_size=100
kwargs = {'num_workers': 1, 'pin_memory': True} 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# we split the full MNIST training
# data set into 5 subsets of consecutive digits. The 5 tasks
# correspond to learning to distinguish between two consecutive digits from 0 to 10.
for digits in [[1,2],[3,4],[5,6],[7,8],[9,0]]:
    dsets={}
    dsets['train']=    MNIST_Split('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),digits=digits)
    dsets['val']=  MNIST_Split('data', train=False, download=True, 
                      transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),digits=digits)
    dlabel=''
    for i in digits:
        dlabel=dlabel+str(i)
    torch.save(dsets,'data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar')


#FIRST TASK TRAINING [1,2]
model_path='General_utils/mnist_net.pth.tar'
from Finetune_SGD import *
digits = [1,2]
dlabel=''
for i in digits:
    dlabel=dlabel+str(i)

dataset_path='data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar'

exp_dir='exp_dir/SGD_MNIST_NET'+dlabel

num_epochs=10

# will generate a best_model.pth.tar if acc improved
fine_tune_SGD(dataset_path=dataset_path, num_epochs=num_epochs, exp_dir=exp_dir,
              model_path=model_path, lr=0.01, batch_size=200)
model_path=os.path.join(exp_dir,'best_model.pth.tar')


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



from MAS import *
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
        exp_dir='exp_dir/SGD_MNIST_NET'+dlabel
    previous_model_path=os.path.join(exp_dir,'best_model.pth.tar')
    exp_dir='exp_dir/SGD_MNIST_NET90'
    dataset_path='data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar'
    current_model_path=os.path.join(exp_dir,'best_model.pth.tar')
    acc2=test_seq_task_performance(previous_model_path=previous_model_path,current_model_path=current_model_path,dataset_path=dataset_path)
    acc1=test_model(previous_model_path,dataset_path)
    forgetting=acc1-acc2
    average_forgetting=average_forgetting+forgetting
average_forgetting=average_forgetting/len(all_digits)    
print(average_forgetting)


#Mimic the standard setup used when after each task the omega is compute on the training samples and accumelated
from MAS import *
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

#estimate forgetting
from MAS import *
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
