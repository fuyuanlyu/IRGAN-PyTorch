import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CFModel(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, device):
        super(CFModel, self).__init__()
        self.device = device
        self.u_embedding = nn.Parameter(torch.rand([user_size, latent_dim], dtype=torch.float32))
        nn.init.uniform_(self.u_embedding, a=-0.05, b=0.05)
        self.i_embedding = nn.Parameter(torch.rand([item_size, latent_dim], dtype=torch.float32))
        nn.init.uniform_(self.i_embedding, a=-0.05, b=0.05)
        self.i_bias = nn.Parameter(torch.zeros([item_size, 1]))

    def forward_all(self, users, items):
        user_w = F.embedding(users, self.u_embedding)
        item_w = F.embedding(items, self.i_embedding)
        item_b = F.embedding(items, self.i_bias)

        all_logits = torch.matmul(user_w, torch.transpose(item_w,0,1)) + torch.transpose(item_b,0,1)
        return all_logits

    def forward(self, batch_data):
        u = batch_data[:,0]
        pos_i = batch_data[:,1]
        neg_i = batch_data[:,2]

        input_user = torch.from_numpy(np.concatenate([u, u], axis=None)).to(self.device)
        input_item = torch.from_numpy(np.concatenate([pos_i, neg_i], axis=None)).to(self.device)
        logits = self._forward(input_user, input_item)
        labels = torch.from_numpy(np.concatenate([np.ones(u.shape[0]), np.zeros(u.shape[0])], axis=None)).to(self.device)
        return (logits, labels)

    def _forward(self, u, i):
        user_w = F.embedding(u, self.u_embedding)
        item_w = F.embedding(i, self.i_embedding)
        item_b = F.embedding(i, self.i_bias)

        logits = (user_w * item_w).sum(dim=-1, keepdim=True) + item_b
        return logits


