import torch
import os
import torch.nn as nn 
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np
import math 
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PeptideDataset import *
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import random

#assuming r=1,
def cartesian_to_spherical(x, y, z):
    theta = math.acos(z / 1)  
    phi = math.atan2(y, x)
    if phi < 0:
        phi += 2*np.pi
    return [theta, phi]


    


def fibonacci_sphere(n):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        sph_points = [cartesian_to_spherical(x,y,z) for x,y,z in points]
    
    return sph_points

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalQuantumClassifierInterface:
    def __init__(
            self,
            num_of_input,
            num_of_output,
            num_of_wires,
            num_of_layers,
            var_Q_circuit,
            var_Q_bias,
            qdevice):

        self.var_Q_circuit = var_Q_circuit
        self.var_Q_bias = var_Q_bias
        self.num_of_input = num_of_input
        self.num_of_output = num_of_output
        self.num_of_wires = num_of_wires
        self.num_of_layers = num_of_layers

        self.qdevice = qdevice

        self.dev = qml.device(self.qdevice, wires = num_of_wires)


    def set_params(self, var_Q_circuit, var_Q_bias):
        self.var_Q_circuit = var_Q_circuit
        self.var_Q_bias = var_Q_bias

    def init_params(self):
        self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
        return self.var_Q_circuit

    def _statepreparation(self, angles):

        """Encoding block of circuit given angles

        Args:
            a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
        """
        # num_of_input determines the number of rotation needed.

        for i in range(self.num_of_input):
            qml.RY(angles[i,0], wires=i)
            qml.RZ(angles[i,1], wires=i)

    def _layer(self, W):
        """ Single layer of the variational classifier.

        Args:
            W (array[float]): 2-d array of variables for one layer

        """

        # Entanglement Layer

        for i in range(self.num_of_wires):
            qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

        # Rotation Layer
        for j in range(self.num_of_wires):
            qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

    def circuit(self, angles):

        @qml.qnode(self.dev, interface='torch', diff_method = "parameter-shift")
        def _circuit(var_Q_circuit, angles):
            """The circuit of the variational classifier."""
            self._statepreparation(angles)
            weights = var_Q_circuit

            for W in weights:
                self._layer(W)


            k = self.num_of_input-1
            return [qml.expval(qml.PauliZ(k))]

        return _circuit(self.var_Q_circuit, angles)



    def forward(self, angles):
        result = ((self.circuit(angles)))
        return torch.tensor(result, requires_grad=True)

vqc = VariationalQuantumClassifierInterface(
            num_of_input =6,
            num_of_output =1,
            num_of_wires=6,
            num_of_layers=2,
            var_Q_circuit=None,
            var_Q_bias = None,
            qdevice = "default.qubit")
           
fib_angles = fibonacci_sphere(5)
         

class VQCTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_params = nn.Parameter(0.01 * torch.randn(2, 12, 3))
    def get_angles(self, in_x):
        in_x_int = [int(item) for item in in_x.tolist()]
        angles = []
        for item in in_x_int:
            theta = fib_angles[item][0]
            phi = fib_angles[item][1]
            angles.append([theta, phi])

        return torch.tensor(angles, requires_grad=True)


    def forward(self, batch_item):
        vqc.var_Q_circuit = self.q_params
        output_batch = []

        for single_item in batch_item:
            angles = self.get_angles(single_item)

            q_out_elem = vqc.forward(angles)
            
            output_batch.append(q_out_elem)

        outputs = torch.stack(output_batch).view(len(batch_item), 1) 
        return outputs

params = {'batch_size' : 4, 'lr': 0.01, 'epochs': 50}
print(device)

dataset = ToyPeptideDataset()

# train_size = int(0.8 * len(dataset))
# test_size = (len(dataset) - train_size) // 2
# val_size = len(dataset) - train_size - test_size

train_size = int(0.16 * len(dataset))
test_size = int(0.02 * len(dataset))
val_size = int(0.02 * len(dataset))
not_used = len(dataset) - train_size - test_size - val_size
train_dataset, test_dataset, val_dataset, not_used_dataset = random_split(dataset, [train_size, test_size, val_size, not_used])

print(f"Dataset: {len(dataset)}")
print(f"Train: {len(train_dataset)}")
print(f"Test: {len(test_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Not_Used: {len(not_used_dataset)}")

batch_size = params['batch_size']

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

train_iter = iter(train_loader)
train_data = next(train_iter)
x_train, y_train = train_data

test_iter = iter(test_loader)
test_data = next(test_iter)
x_test, y_test = test_data

val_iter = iter(val_loader)
val_data = next(val_iter)
x_val, y_val = val_data

def saving_plotting(params, tr_list, val_list):
    exp_name = datetime.now().strftime("%m_%d_%H_%M_%S")

    directory = f"Exp:_{exp_name}"
    parent_dir = '/global/u2/r/rr637/VQC_Peptides/Results'
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    path_plots = path + '/Plots'
    os.mkdir(path_plots)
    path_models = path + '/Models'
    os.mkdir(path_models)    
    title = 'Train and Validation Loss'
    plot_loss(tr_list, val_list, exp_name, title)
    
    with open(path + "/train_loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tr_list)

    with open(path + "/val_loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(val_list)
    with open(path + "/params.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in params.items():
            writer.writerow([key, value])
    return 




def plot_loss(tr_l, vl_l,exp_name,title):
    plt.plot(tr_l,label = "train loss")
    plt.plot(vl_l,label = "val loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(f"Results/Exp:_{exp_name}/Plots/loss_plot.png")
    plt.show()             
             
  
 #Train and Validation Loop
def train_model(model, optimizer, train_loader, val_loader, params):

    criterion = nn.MSELoss()
    
    batch_size = params['batch_size']
    num_epochs = params['epochs']
    train_samples = len(train_dataset)
    val_samples = len(val_dataset)
    n_tr_iterations = math.ceil(train_samples/batch_size)
    n_val_iterations = math.ceil(val_samples/batch_size)

    train_loss_epoch = []
    val_loss_epoch = []
    for epoch in range(num_epochs):
        model.train()
        print(f"Model Parameters {model.parameters}")
        print(f"EPOCH: {epoch}")
        train_loss = 0
        for i, (data, target) in enumerate(train_loader):
            
            
            if i == 0:
                print(f'Inputs {data.shape} | Labels {target.shape}')
           
            data, target = data.double().to(device), target.double().to(device)
            y_predicted = model(data).double().to(device)
            loss = criterion(y_predicted, target)

            loss.backward()
            optimizer.step()
            

            optimizer.zero_grad()
            train_loss += loss.item()

            
            if (i+1) % 1000 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_tr_iterations}|train_loss = {loss.item():.4f}')
        avg_tr_loss = train_loss/n_tr_iterations
        print(f"Avg_Train_Loss: {avg_tr_loss}")
        train_loss_epoch.append(avg_tr_loss)
        model.eval()
        val_loss = 0
        for i, (data, target) in enumerate(val_loader):
           
            data, target = data.to(device), target.to(device)
            y_predicted = model(data).to(device)
            vloss = criterion(y_predicted, target)
            val_loss += vloss.item()
            if (i+1) % 500 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_val_iterations}|val_loss = {vloss.item():.4f}')
        avg_val_loss = train_loss/n_val_iterations
        print(f"Avg_Val_Loss: {avg_val_loss}")
        val_loss_epoch.append(avg_val_loss)
    return train_loss_epoch,val_loss_epoch
        
    

        
        

        


