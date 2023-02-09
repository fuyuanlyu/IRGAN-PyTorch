import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataSet(object):
    def __init__(self, folderpath):
        self.user_pos_train, self.train_U, self.train_I = self.get_dict(os.path.join(folderpath, 'movielens-100k-train.txt'))
        self.user_pos_valid, self.valid_U, self.valid_I = self.get_dict(os.path.join(folderpath, 'movielens-100k-test.txt'))
        self.all_U = list(set(self.train_U + self.valid_U))
        self.all_I = list(set(self.train_I + self.valid_I))
        
    def get_dict(self, filename):
        this_pos_dict = dict()
        U, I = set(), set()
        with open(filename) as fin:
            for line in fin:
                line = line.split()
                uid = int(line[0])
                iid = int(line[1])
                if uid in this_pos_dict:
                    this_pos_dict[uid].append(iid)
                else:
                    this_pos_dict[uid] = [iid]
                U.add(uid)
                I.add(iid)
        return this_pos_dict, list(U), list(I)


class IRGAN(MyDataSet):
    def __init__(self, folderpath, device):
        super(IRGAN, self).__init__(folderpath)
        self.device = device

    # Generate training result for Discriminator
    # This is the single user version
    def generate_for_d(self, model, temperature=1.0):
        data = []
        for u in self.user_pos_train:
            pos_i = self.user_pos_train[u]

            input_u = torch.Tensor([u]).to(torch.int32).to(self.device)
            input_i = torch.Tensor(self.all_I).to(torch.int32).to(self.device)
            
            rating = model.forward_all(input_u, input_i).squeeze().cpu().detach().numpy() / temperature
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)

            neg_i = np.random.choice(np.array(self.all_I), size=len(pos_i), p=prob)
            for i in range(len(pos_i)):
                data.append([int(u), int(pos_i[i]), int(neg_i[i])])
        return np.array(data)

    # Generate training result for Discriminator
    # This is the batch user version
    def generate_for_d_batch(self, model, temperature=1.0):
        data = []
        user_in_train = np.fromiter(self.user_pos_train.keys(), dtype=int)
        num_user_in_train = len(user_in_train)
        bs = 1000
        start = 0
        while start < num_user_in_train:
            if start + bs <= num_user_in_train:
                end = start + bs
            else:
                end = num_user_in_train
            batch_u = user_in_train[start:end]
            
            input_u = torch.from_numpy(batch_u).to(torch.int32).to(self.device)
            input_i = torch.Tensor(self.all_I).to(torch.int32).to(self.device)

            rating = model.forward_all(input_u, input_i).squeeze().cpu().detach().numpy() / temperature
            exp_rating = np.exp(rating)
            
            for i in range(end-start):
                pos_i = self.user_pos_train[batch_u[i]]
                this_exp_rating = exp_rating[i]
                prob = this_exp_rating / np.sum(this_exp_rating)

                neg_i = np.random.choice(np.array(self.all_I), size=len(pos_i), p=prob)
                for j in range(len(pos_i)):
                    data.append([int(batch_u[i]), int(pos_i[j]), int(neg_i[j])])
            start = end
        return np.array(data)




