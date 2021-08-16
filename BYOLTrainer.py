import os
import torch
from pathlib import Path
from dataset import initialize_dataset
import torch.nn.functional as F

class BYOLTrainer:
    def __init__(self, online, target, predictor, optimizer, device, params) -> None:
        print(params)
        self.online = online
        self.target = target
        self.predictor = predictor
        self.optimizer = optimizer
        self.device = device
        self.epochs = params.epochs
        self.m = params.momentum_update
        self.batch_size = params.batch_size
        self.save_model = params.save_model
    
    def initialize_target_network(self):
        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_target_network_parameters(self):
        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def train(self, n_views):
        dataset = initialize_dataset(root_folder='./data')
        data_loader = dataset.load_dataset(n_views=n_views, batch_size=self.batch_size)
        
        n_iter=0
        self.initialize_target_network()
        for epoch in range(self.epochs):
            for (batch1, batch2), _ in data_loader:
                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)

                loss = self.update(batch1, batch2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_target_network_parameters()
                n_iter += 1
        
            print(f"Epoch: {epoch}\tLoss: {loss}")


        if self.save_model:
            Path("checkpoints/").mkdir(parents=True, exist_ok=True)
            self.store_model(os.path.join('checkpoints/', 'model.pth'))
    
    def update(self, batch1, batch2):
        prediction_view_1 = self.predictor(self.online(batch1))
        prediction_view_2 = self.predictor(self.online(batch2))

        with torch.no_grad():
            targets_view_1 = self.target(batch1)
            targets_view_2 = self.target(batch2)

        loss = self.regression_loss(prediction_view_1, targets_view_1)
        loss += self.regression_loss(prediction_view_2, targets_view_2)

        return loss.mean()

    def store_model(self, path):
        if self.save_model:
            torch.save({
            'online_network_state_dict': self.online.state_dict(),
            'target_network_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)