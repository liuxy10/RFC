

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import tqdm
import numpy as np

class GaitPhasePredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=20):
        super(GaitPhasePredictor,self).__init__()
        # self.cnn = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=8, stride=1, padding='same')  
        self.lstm = nn.LSTM(
                    # 32,
                    input_size, 
                    hidden_size=hidden_size,
                    num_layers=1, 
                    # batch_first=True,
                    dtype=torch.float32)  
        self.fc_layers = nn.Linear(hidden_size, 2, dtype=torch.float32)
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(hidden_size, 30, dtype=torch.float32),
        #     nn.Softmax(),
        #     nn.Linear(30, 10, dtype=torch.float32),
        #     nn.Softmax(),
        #     nn.Linear(10, 1, dtype=torch.float32)
        # )
    
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = x
        # x, _ = self.cnn(x.transpose(1, 2))  # Transpose to (batch, channels, seq_len)
        x, _ = self.lstm(x)  # Use sequence output (batch, seq, features)
        # print(x.requires_grad)
        x = self.fc_layers(x[:, -1, :]).squeeze()  # Last timestep only[1]
        return x


    def train_model(self, X, y, epochs=15, lr=0.001):
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        y_tensor = torch.tensor(y, dtype=torch.float32,  requires_grad=True)

        # print("X_tensor.grad, y_tensor.grad", X_tensor.requires_grad, y_tensor.requires_grad)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), 
                                    batch_size=32//4,
                                    shuffle=True)
            
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()  # nn module train: to set the model to training mode
            with tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
                for X_batch, y_batch in t:
                    X_batch.requires_grad_()
                    y_batch.requires_grad_()
                    # print(X_batch.shape, y_batch.shape)
                    outputs = self(X_batch)
                    loss = F.mse_loss(outputs, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t.set_postfix(loss=loss.item())  
            torch.cuda.empty_cache()    
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    def predict(self, X):
        with torch.no_grad():
            return self(torch.tensor(X, dtype=torch.float32)).numpy()
        
    def eval_loss(self, X, y):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            outputs = self(X_tensor)
            loss = F.mse_loss(outputs, y_tensor)
        return loss.item()

    def train_phase_predictor(self, expert_qpos, expert_phase,
                              n_seq=1, epochs=50, lr=0.001, 
                              save_path='phase_predictor_lstm.pth', 
                              vis=False):
        X = np.array([expert_qpos[i:i+n_seq, 7:14] for i in range(len(expert_qpos)-n_seq)])
        y = expert_phase[n_seq:]
        y = np.vstack([np.cos(y * 2 * np.pi), np.sin(y * 2 * np.pi)]).T
        assert X.shape[0] == y.shape[0], f"X {X.shape} and y {y.shape} should have the same number of samples"

        self.train_model(X, y, epochs=epochs, lr=lr)
        torch.save(self.state_dict(), save_path)
        self.load_state_dict(torch.load(save_path))

        y_pred_p = self.predict(X)
        rmse_p = self.eval_loss(X, y)
        rs_p = 1 - np.sum((y_pred_p - y) ** 2) / np.sum((y - np.mean(y)) ** 2)
        y_pred = np.arctan2(y_pred_p[:, 1], y_pred_p[:, 0]) / (2 * np.pi) + (-np.sign(np.arcsin(np.clip(y_pred_p[:,1], -1, 1))))  / 2 + 1 / 2
        rmse = np.sqrt(np.mean((y_pred - expert_phase[n_seq:]) ** 2))
        rs = 1 - np.sum((y_pred - expert_phase[n_seq:]) ** 2) / np.sum((expert_phase[n_seq:] - np.mean(expert_phase[n_seq:])) ** 2)
        print(f"RMSE (polar): {rmse_p}, cartesian: {rmse}")
        print(f"R square (polar): {rs_p}, cartesian: {rs}")
        self.eval()
        print([p.requires_grad for p in self.parameters()])
        if vis:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(y_pred_p.shape[1], 1, figsize=(10, 6))
            for i in range(y_pred_p.shape[1]):
                ax[i].plot(y_pred_p[:, i], label='Predicted')
                ax[i].plot(y[:, i], label='True')
                ax[i].set_title(f'Phase {i+1}')
                ax[i].legend()
            plt.title(f'Gait Phase Prediction in polar coordinate, rmse = {rmse_p:.4f}, r square = {rs_p:.4f}')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(expert_phase, label='True Phase')
            ax.plot(y_pred, label='Predicted Phase')
            ax.set_title(f'Gait Phase Prediction, rmse = {rmse:.4f}, r square = {rs:.4f}')
            ax.legend()
            plt.show()
            exit()
if __name__ == "__main__":
    # Example usage
    model = GaitPhasePredictor()
    X = torch.randn(100, 2, 7)  # 100 samples, 2 timesteps, 7 features
    y = torch.randn(100)  # 100 target values

    model.train_model(X, y)
    predictions = model.predict(X)
    print(predictions)