from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
import numpy as np
import sys
import torch.nn.functional as F
import torch.fft as fft
import torch.nn.init as init


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, device, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestam)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)  # 将输入进来的训练集分为几个batch
                while b_begin < len(queries):
                    if queries.shape[1] > 4:              # time intervals exist   对五元组中时间戳的处理
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start = -int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):        # 时间戳转换成id
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).to(device)
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).to(device)

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size] # 500, 4
                        q = self.get_queries(these_queries) # 500, 400
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().to(device)
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class TCompoundE(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,no_time_emb=False, init_size: float = 1e-2):
        super(TCompoundE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]] # without no_time_emb
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True
	
    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = (rel[0] + time[0]) * time[1], rel[1]        
        return torch.sum(
            ( (lhs[0] + rt[1]) * rt[0] ) * rhs[0], 1, keepdim=True)
	
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)  # S_{r}, T_{r}
        time = time[:, :self.rank], time[:, self.rank:] # T_{τ}, S_{τ}

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = (rel[0] + time[0]) * time[1], rel[1]   # S_{rτ}, T_{rτ}

        return (
                    ((lhs[0] + rt[1]) * rt[0] ) @ right[0].t()  # predictions based on geometric operations
               ), (
                   torch.sqrt(lhs[0] ** 2),
                   torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
                   torch.sqrt(rhs[0] ** 2)
               ),  self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        rt = (rel[0] + time[0]) * time[1], rel[1]
        return (lhs[0] + rt[1]) * rt[0]


class TCompoundE_DS(TKBCModel):    # deepseek
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, alpha: float = 10, init_size: float = 1e-2):
        super(TCompoundE_DS, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=False)
            for s in [sizes[0], sizes[1], sizes[3]]  # entities, relations, time
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846
        self.weight_separation = 1.0
        self.weight_high_freq_intensity = 1.0
        self.alpha = alpha

    @staticmethod
    def has_time():
        return True

    def decompose_relation_fft(self, rel_embedding):
        freq_domain = fft.fft(rel_embedding, dim=1)
        freqs = fft.fftfreq(self.rank, d=1.0).to(rel_embedding.device)
        abs_freqs = torch.abs(freqs)  # 取频率绝对值
        low_freq_mask = torch.exp(-abs_freqs * self.alpha).view(1, -1)  # 平滑低通滤波
        high_freq_mask = 1.0 - low_freq_mask  # 高频部分是剩下的部分
        low_freq = freq_domain * low_freq_mask
        high_freq = freq_domain * high_freq_mask
        low_freq_component = fft.ifft(low_freq, dim=1).real
        high_freq_component = fft.ifft(high_freq, dim=1).real

        return low_freq_component, high_freq_component, low_freq, high_freq

    def loss_freq(self, low_freq, high_freq):
        # 1. Separation Loss
        separation_loss = - torch.norm(low_freq - high_freq, p=2)
        # 2. High-frequency Intensity Loss
        high_freq_intensity_loss = torch.norm(high_freq, p=2)
        total_loss = (self.weight_separation * separation_loss \
                    + self.weight_high_freq_intensity * high_freq_intensity_loss) * (1 / (high_freq.shape[0] * high_freq.shape[0]))
        return total_loss

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        # Decompose relation into low-frequency and high-frequency components
        rel_low, rel_high, _, _ = self.decompose_relation_fft(rel[0])

        # Combine low-frequency and high-frequency components with time
        time_smoothed = time[0].mean(dim=1, keepdim=True)  # Smoothed time embedding
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])  # Time gradient
        rt_low = (rel_low + time_smoothed) * time[1]  # Combine with smoothed time
        rt_high = (rel_high + time_gradient) * time[1]  # Combine with time gradient
        rt = 0.5 * rt_low + 0.5 * rt_high, rel[1]

        return torch.sum(((lhs[0] + rt[1]) * rt[0]) * rhs[0], 1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        # Decompose relation into low-frequency and high-frequency components
        rel_low, rel_high, low_freq, high_freq = self.decompose_relation_fft(rel[0])

        # Combine low-frequency and high-frequency components with time
        time_smoothed = time[0].mean(dim=1, keepdim=True)  # Smoothed time embedding
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])  # Time gradient
        rt_low = (rel_low + time_smoothed) * time[1]  # Combine with smoothed time
        rt_high = (rel_high + time_gradient) * time[1]  # Combine with time gradient

        # Dynamic fusion of low-frequency and high-frequency components
        rt = 0.5 * rt_low + 0.5 * rt_high, rel[1]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        loss_freq = self.loss_freq(low_freq, high_freq)

        return (
            ((lhs[0] + rt[1]) * rt[0]) @ right[0].t()  # predictions based on geometric operations
        ), (
            torch.sqrt(lhs[0] ** 2),
            torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
            torch.sqrt(rhs[0] ** 2)
        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight, loss_freq

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        # Decompose relation into low-frequency and high-frequency components
        rel_low, rel_high, _, _ = self.decompose_relation_fft(rel[0])

        # Combine low-frequency and high-frequency components with time
        time_smoothed = time[0].mean(dim=1, keepdim=True)  # Smoothed time embedding
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])  # Time gradient
        rt_low = (rel_low + time_smoothed) * time[1]  # Combine with smoothed time
        rt_high = (rel_high + time_gradient) * time[1]  # Combine with time gradient

        # Dynamic fusion of low-frequency and high-frequency components
        rt = 0.5 * rt_low + 0.5 * rt_high, rel[1]

        return (lhs[0] + rt[1]) * rt[0]