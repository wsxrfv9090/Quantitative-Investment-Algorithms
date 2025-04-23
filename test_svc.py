import SVM.SVC as svc
import torch
import global_resources as gr

def nl():
    print("\n")

device = gr.set_device()
X = torch.tensor([[1., 2.],
            [3., 4.]], device = device)      # two data‐points in 2D  
y = torch.tensor([+1., -1.], device = device)    # labels +1 and –1  
weights = torch.tensor([0.5, -0.5], device = device, requires_grad  =True)  
bias = torch.tensor(0.1, device = device, requires_grad = True)  
lr = 0.01  

weights_bias = svc.create_random_weights_bias(X.shape[-1] + 1, dtype = X.dtype)
svc.train(X, y, weights = weights_bias[:-1], bias = weights_bias[-1], num_epochs = 1000, start_learning_rate = 0.01)
