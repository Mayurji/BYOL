"""
Contrastive Learning:

It is one of many paradigms that fall under Deep Distance Metric Learning 
where the objective is to learn a distance in a low dimensional space which is consistent 
with the notion of semantic similarity. In simple terms(considering image domain), it means 
to learn similarity among images where distance is less for similar images and more for 
dissimilar images.

Bootstrap Your Own Latent(BYOL):
Approach:

    1. Take two networks with same architecture: a fixed ‘target’ network(which is randomly 
       initialized) and a trainable ‘online’ network.
    2. Take an input image t and create an augmented view t1.
    3. Pass image t through online network, image t1 through target network and extract 
       predicted and target embeddings respectively.
    4. Minimize the distance between both embeddings.
    5. Update the target network — which is the moving exponential average of the previous 
       online networks.
    6. Repeat steps 2–5.

Reference: https://medium.com/swlh/neural-networks-intuitions-10-byol-paper-explanation-f8b1d6e83b1c
"""
import torch
import torch.nn as nn
from torchvision import models

class ResNetBYOL(nn.Module):
   def __init__(self, projection_in_dimension, projection_out_dimension) -> None:
      super().__init__()
      backbone = models.resnet50(pretrained=False)
      self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
      self.projection = nn.Sequential(
         nn.Linear(backbone.fc.in_features, projection_in_dimension),
         nn.BatchNorm1d(projection_in_dimension), nn.ReLU(),
         nn.Linear(projection_in_dimension, projection_out_dimension)
      )

   def forward(self, x):
      features = self.encoder(x)
      features = features.view(features.shape[0], features.shape[1])
      return self.projection(features)
