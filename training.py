import torch
import argparse
import torch.nn as nn
from BYOLTrainer import BYOLTrainer
from ResNet_BYOL import ResNetBYOL

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Self-Supervised Learning - BYOL - PyTorch')
parser.add_argument('-data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='Momentum (default: 0.9)',
                    dest='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--in_dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--momentum-update', default=0.996, type=float,
                    metavar='MU', help='Momentum Update (default: 0.996)',
                    dest='momentum_update')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--save-model', default=False, type=bool,
                    help='To save model, True or False')

class MLPHead(nn.Module):
    def __init__(self, input_channel, projection_in_dimension, projection_out_dimension) -> None:
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_channel, projection_in_dimension),
            nn.BatchNorm1d(projection_in_dimension), nn.ReLU(),
            nn.Linear(projection_in_dimension, projection_out_dimension)
        )

    def forward(self, x):
        return self.mlp_head(x)

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    online = ResNetBYOL(projection_in_dimension=args.in_dim, projection_out_dimension=args.out_dim).to(device)
    predictor = MLPHead(input_channel=online.projection[-1].out_features, projection_in_dimension=args.in_dim,
                        projection_out_dimension=args.out_dim).to(device)

    target = ResNetBYOL(projection_in_dimension=args.in_dim, projection_out_dimension=args.out_dim).to(device)

    optimizer = torch.optim.SGD(list(online.parameters())+list(predictor.parameters()), lr=args.lr,
                                weight_decay=args.weight_decay, momentum=args.momentum)
    trainer = BYOLTrainer(online=online, target=target, predictor=predictor,
                        optimizer=optimizer, device=device, params=args)

    trainer.train(n_views=args.n_views)

if __name__ == '__main__':
    main()