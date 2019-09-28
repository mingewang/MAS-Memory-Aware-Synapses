from MAS import *
from MNIST_split import *
import os

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

