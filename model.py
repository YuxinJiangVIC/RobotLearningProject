import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

#implement the resnet50 
#align with https://github.com/google-research/simclr/blob/master/resnet.py
class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

    
# class SimCLR(pl.LightningModule):

#     def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
#         super().__init__()
#         self.save_hyperparameters()
#         assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
#         # Base model f(.)
#         self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
#         # The MLP for g(.) consists of Linear->ReLU->Linear
#         self.convnet.fc = nn.Sequential(
#             self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
#             nn.ReLU(inplace=True),
#             nn.Linear(4*hidden_dim, hidden_dim)
#         )

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.hparams.lr,
#                                 weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                             T_max=self.hparams.max_epochs,
#                                                             eta_min=self.hparams.lr/50)
#         return [optimizer], [lr_scheduler]

#     def info_nce_loss(self, batch, mode='train'):
#         imgs, _ = batch
#         imgs = torch.cat(imgs, dim=0)

#         # Encode all images
#         feats = self.convnet(imgs)
#         # Calculate cosine similarity
#         cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
#         # Mask out cosine similarity to itself
#         self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
#         cos_sim.masked_fill_(self_mask, -9e15)
#         # Find positive example -> batch_size//2 away from the original example
#         pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
#         # InfoNCE loss
#         cos_sim = cos_sim / self.hparams.temperature
#         nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
#         nll = nll.mean()

#         # Logging loss
#         self.log(mode+'_loss', nll)
#         # Get ranking position of positive example
#         comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
#                               cos_sim.masked_fill(pos_mask, -9e15)],
#                              dim=-1)
#         sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
#         # Logging ranking metrics
#         self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
#         self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
#         self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

#         return nll

#     def training_step(self, batch, batch_idx):
#         return self.info_nce_loss(batch, mode='train')

#     def validation_step(self, batch, batch_idx):
#         self.info_nce_loss(batch, mode='val')