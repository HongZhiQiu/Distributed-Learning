## ADMMNet ##
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Prox(nn.Module):
    def __init__(self):
        super(Prox, self).__init__()

    def forward(self, g, b, rho1):
        rho1_abs_g = rho1 * torch.abs(g)
        updated_g = torch.where(g != 0, (b + rho1_abs_g) / (1 + rho1) * torch.sign(g), b / (1 + rho1))
        return updated_g

class ADMMUnfoldLayer(nn.Module):
    def __init__(self, lb, ub, A):
        super(ADMMUnfoldLayer, self).__init__()
        self.lb = lb
        self.ub = ub
        self.rho1 = nn.Parameter(torch.full((1,), 1e-6))
        self.rho2 = nn.Parameter(torch.full((1,), 1e-6))
        self.prox = Prox()
        self.A_star_A_plus_rho2_I_inv = self.compute_inv_matrix(A)

    def compute_inv_matrix(self, A):
        n = A.shape[1]
        return torch.inverse(self.rho1 * torch.matmul(A.T, A) + self.rho2 * torch.eye(n))

    def forward(self, A, b, x, z, d, w):
        print("Shapes at the start of ADMMUnfoldLayer.forward:")
        print("A.shape:", A.shape, "b.shape:", b.shape, "x.shape:", x.shape, "z.shape:", z.shape, "d.shape:", d.shape, "w.shape:", w.shape)

        x = torch.matmul(self.A_star_A_plus_rho2_I_inv, (self.rho1 * torch.matmul(A.T, z) + torch.matmul(A.T, d) + self.rho2 * x - w))
        y = torch.maximum(x + w / self.rho2, self.lb)
        y = torch.minimum(y, self.ub)

        # 更新 z，使用 prox_vectorized_real_updated
        Ax_minus_d = torch.matmul(A, x) - d / self.rho1
        z = self.prox(Ax_minus_d, b, self.rho1)

        d = d + self.rho1 * (z - torch.matmul(A, x))
        w = w + self.rho2 * (x - y)
        print("Shapes at the end of ADMMUnfoldLayer.forward:")
        print("x.shape:", x.shape, "z.shape:", z.shape, "d.shape:", d.shape, "w.shape:", w.shape)

        return x, z, d, w


class ADMMNet(nn.Module):
    def __init__(self, lb, ub, num_layers, A):
        super(ADMMNet, self).__init__()
        self.layers = nn.ModuleList([ADMMUnfoldLayer(lb, ub, A) for _ in range(num_layers)])

    def forward(self, A, b, z_init, d_init, y_init, w_init):
        x, z, d, w = z_init, d_init, y_init, w_init
        for layer in self.layers:
            x, z, d, w = layer(A, b, x, z, d, w)  
        return x
    
class CustomDataset(Dataset):
    def __init__(self, b, masked_matrix, signal):
        self.b = b
        self.masked_matrix = masked_matrix
        self.signal = signal
    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, idx):
        b_sample = torch.Tensor(self.b[idx, :]).float()
        masked_matrix_sample = torch.Tensor(self.masked_matrix[idx, :]).float()
        s_sample = torch.Tensor(self.signal[idx, :]).float()
        return b_sample, masked_matrix_sample, s_sample
    
data_sensing = scipy.io.loadmat('sensing_matrix.mat')
A_mat = data_sensing['sensing_matrix']
A_tensor = torch.tensor(A_mat, dtype=torch.float32)


data = scipy.io.loadmat('TrainingData.mat')
b = data['y_data'] 
masked_matrix = data['masked_matrix_data'] 
signal = data['signal']

s_tensor = torch.tensor(signal, dtype=torch.float32)
b_tensor = torch.tensor(b, dtype=torch.float32)
masked_matrix_tensor = torch.tensor(masked_matrix, dtype=torch.float32)

dataset = CustomDataset(b_tensor, masked_matrix_tensor, s_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

lb = 1  
ub = 0  
num_layers = 10  
model = ADMMNet(lb, ub, num_layers, A_tensor)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Input data shapes:")
print("A_tensor.shape:", A_tensor.shape, "b_tensor.shape:", b_tensor.shape)
num_epochs = 200  

for epoch in range(num_epochs):
    for b_batch, masked_matrix_batch, s_batch in dataloader:
        print('shape of s_batch')
        print(s_batch.shape)
        
        z_init = torch.zeros_like(b_batch) 
        d_init = torch.zeros(A_tensor.size(1), b_batch.size(0), dtype=torch.float32) 
        y_init = torch.zeros(A_tensor.size(0), b_batch.size(0), dtype=torch.float32) 
        w_init = torch.zeros_like(y_init)

        optimizer.zero_grad() 

        estimated_masked_matrix = model(A_tensor, b_batch, z_init, d_init, y_init, w_init)

        loss = criterion(estimated_masked_matrix, masked_matrix_batch)
    
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
