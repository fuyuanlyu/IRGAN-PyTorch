import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import multiprocessing as mp

from utils.dataset import IRGAN
from utils.evaluate import evaluate_one_user
from model.basic import CFModel

torch.random.manual_seed(2022)
np.random.seed(2022)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ml-100k/', help='input dataset path.')
    parser.add_argument('--model_dir', type=str, default='cache/', help="Model Path")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--bs', type=int, default=2048, help='number of batch size.')
    parser.add_argument('--dim', type=int, default=16, help='size of latent dim.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    return parser.parse_args()

args = parse_args()
if args.gpu == -1:
    device = torch.device('cpu')
elif args.gpu >= 0 and args.gpu <= 7:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    device = torch.device('cuda')
else:
    raise ValueError("Wrong GPU ID!")


class Trainer(object):
    def __init__(self):
        os.makedirs(args.model_dir, exist_ok=True)
        self.negsampler = IRGAN(args.data_dir, device)
        self.num_U = len(self.negsampler.all_U)
        self.num_I = len(self.negsampler.all_I)
        self.dis_model = CFModel(self.num_U, self.num_I, args.dim, device).to(device)
        self.gen_model = CFModel(self.num_U, self.num_I, args.dim, device).to(device)
        self.loss_func = F.binary_cross_entropy_with_logits
        self.dis_optim = torch.optim.SGD(self.dis_model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.gen_optim = torch.optim.SGD(self.gen_model.parameters(), lr=args.lr)

    def evaluate(self, name):
        result = np.zeros(6)
        pool = mp.Pool(4)

        bs = args.bs * 10
        valid_u_i_dict = self.negsampler.user_pos_valid
        train_u_i_dict = self.negsampler.user_pos_train
        valid_user = self.negsampler.valid_U
        all_I = self.negsampler.all_I

        with torch.no_grad():
            # Batch Single Process Loop
            total_batch = int(len(valid_user) * 1.0 / bs) + 1
            start = 0
            for idx in range(total_batch):
                if start + bs <= len(valid_user):
                    end = start + bs
                else:
                    end = len(valid_user)
                batch_u = valid_user[start:end] 
                start = end

                input_u = torch.Tensor(batch_u).to(device).to(torch.int32)
                input_i = torch.Tensor(all_I).to(device.to(torch.int32))
                batch_scores = self.gen_model.forward_all(input_u, input_i).cpu().detach().numpy()

                batch_input = []
                for i in range(len(batch_u)):
                    a = batch_scores[i]
                    b = batch_u[i]
                    if b not in train_u_i_dict.keys() or b not in valid_u_i_dict.keys():
                        continue
                    else:
                        c = train_u_i_dict[b]
                        d = valid_u_i_dict[b]
                        batch_input.append((a, b, c, d, all_I))

                batch_result = pool.map(evaluate_one_user, batch_input)
                for re in batch_result:
                    result += re

        pool.close()
        ret = result / len(valid_user)
        ret = list(ret)
        return ret

    def run(self):
        best_p = 0.

        for epoch in range(15):
            for d_epoch in range(100):
                # Generate Training Dataset
                if d_epoch % 5 == 0:
                    train_data = self.negsampler.generate_for_d_batch(self.gen_model, temperature=0.2)
                    train_size = len(train_data)
                
                # Pretrain Discriminator for one epoch
                start = 0
                while start < train_size:
                    if start + args.bs <= train_size:
                        end = start + args.bs
                    else:
                        end = train_size

                    self.dis_optim.zero_grad()
                    batch_data = train_data[start:end]
                    outputs = self.dis_model.forward(batch_data)
                    loss = self.loss_func(outputs[0].squeeze(), outputs[1], reduction="sum")
                    loss.backward()
                    self.dis_optim.step()

                    start = end

            # Train Generator for on epoch
            for g_epoch in range(50):
                sample_lambda = 0.2

                # Changed to batched input
                data = []
                user_in_train = np.fromiter(self.negsampler.user_pos_train.keys(), dtype=int)
                num_user_in_train = len(user_in_train)
    
                start = 0
                while start < num_user_in_train:
                    if start + args.bs <= num_user_in_train:
                        end = start + args.bs
                    else:
                        end = num_user_in_train
                    batch_u = user_in_train[start:end]

                    input_u = torch.from_numpy(batch_u).to(torch.int32).to(device)
                    input_i = torch.Tensor(self.negsampler.all_I).to(torch.int32).to(device)

                    rating = self.gen_model.forward_all(input_u, input_i)
                    np_rating = rating.squeeze().cpu().detach().numpy()
                    exp_rating = np.exp(np_rating)

                    for i in range(end-start):
                        self.gen_optim.zero_grad()

                        u = batch_u[i]
                        pos_i = self.negsampler.user_pos_train[u]
                        this_exp_rating = exp_rating[i]
                        prob = this_exp_rating / np.sum(this_exp_rating)

                        pn = (1 - sample_lambda) * prob
                        pn[pos_i] += sample_lambda * 1.0 / len(pos_i)

                        sample = np.random.choice(self.negsampler.all_I, 2*len(pos_i), p=pn)

                        # Get reward and adapt it with importance sampling
                        input_u = torch.Tensor([u]).to(torch.int32).to(device)
                        input_sample = torch.from_numpy(sample).to(torch.int64).to(device)

                        reward = self.dis_model.forward_all(input_u, input_sample).squeeze().cpu().detach().numpy()
                        reward = reward * prob[sample] / pn[sample]

                        # Update G
                        total_prob = F.softmax(rating[i], dim=0)
                        sample_prob = torch.gather(total_prob, 0, input_sample)
                        gan_loss = -(torch.log(sample_prob) * torch.from_numpy(reward).to(device)).mean()
                        gan_loss.backward(retain_graph=True)
                        self.gen_optim.step()

                    start = end

                valid_result = self.evaluate(name="valid")
                print("Epoch", epoch, "G_Epoch:", g_epoch, "result:", valid_result)
                print("Epoch: %d | Generator Epoch: %d | Valid P@10: %.5f | Valid NDCG@10: %.5f" % (epoch, g_epoch, valid_result[2], valid_result[5]))

                if best_p <= valid_result[2]:
                    best_p = valid_result[2]
                    best_ndcg = valid_result[5]

        return best_p, best_ndcg


if __name__ == '__main__':
    trainer = Trainer()
    best_p, best_ndcg = trainer.run()
    print("Best P@10: %.5f, NDCG@10: %.5f" %(best_p, best_ndcg))
    



