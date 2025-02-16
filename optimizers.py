import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset
import sys

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, freq_reg: float = 0.0, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.freq_reg = freq_reg


    def epoch(self, examples: torch.LongTensor, device):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].to(device)

                predictions, factors, time, l_freq = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)

                l = l_fit + l_reg + l_time + l_freq * self.freq_reg

                self.optimizer.zero_grad()
                l = l.to(device)
                l.backward()
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad)
                
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                print('loss={},reg={},cont={},freq={}'.format(l_fit.item(),l_reg.item(),l_time.item(),l_freq.item()))
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}',
                    freq=f'{l_freq.item():.0f}'
                )

class TKBCOptimizer_sparse(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            sparse_optimizer: optim.Optimizer, dense_optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.sparse_optimizer = sparse_optimizer
        self.dense_optimizer = dense_optimizer
        self.batch_size = batch_size
        self.verbose = verbose


    def epoch(self, examples: torch.LongTensor, device):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].to(device)

                predictions, factors, time = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)

                l = l_fit + l_reg + l_time

                self.sparse_optimizer.zero_grad()
                self.dense_optimizer.zero_grad()
                l = l.to(device)
                l.backward()
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad)
                
                self.sparse_optimizer.step()
                self.dense_optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                print('loss={},reg={},cont={}'.format(l_fit.item(),l_reg.item(),l_time.item()))
                bar.set_postfix(
                    loss=f'{l_fit.item():.0f}',
                    reg=f'{l_reg.item():.0f}',
                    cont=f'{l_time.item():.0f}'
                )


