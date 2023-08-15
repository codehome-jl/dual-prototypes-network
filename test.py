# import torch
# from torch.functional import norm
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict
# import math
# from itertools import combinations
# from torch.nn.init import xavier_normal_
#
# from torch.nn.modules.activation import MultiheadAttention
# import os
# from torch.autograd import Variable
# import torchvision.models as models
# from utils import *
# from einops import rearrange
# from plot import *
# import copy
#
#
# class CNN_FSHead(nn.Module):
#     """
#     Base class which handles a few-shot method. Contains a resnet backbone which computes features.
#     """
#
#     def __init__(self, args):
#         super(CNN_FSHead, self).__init__()
#         self.train()
#         self.args = args
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#         self.num = 0
#
#         last_layer_idx = -1
#
#         if self.args.backbone == "resnet18":
#             backbone = models.resnet18(pretrained=True)
#         elif self.args.backbone == "resnet34":
#             backbone = models.resnet34(pretrained=True)
#         elif self.args.backbone == "resnet50":
#             backbone = models.resnet50(pretrained=True)
#
#         if self.args.pretrained_backbone is not None:
#             checkpoint = torch.load(self.args.pretrained_backbone)
#             backbone.load_state_dict(checkpoint)
#
#         self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])
#
#     def get_feats(self, support_images, target_images, context_labels):
#
#         support_features = self.backbone(support_images).squeeze()
#         target_features = self.backbone(target_images).squeeze()
#
#         dim = int(support_features.shape[1])
#
#         support_features = support_features.reshape(-1, self.args.seq_len, dim)
#         target_features = target_features.reshape(-1, self.args.seq_len, dim)
#         self.num = self.num + 1
#         if self.num % 200 == 0:
#             feat = support_features.reshape(25, -1).cpu().detach().numpy()
#
#             fig = plt.figure(figsize=(10, 10))
#
#             plotlabels(visual(feat), context_labels.cpu().detach().numpy(), '(a)')
#
#             plt.savefig("./pict/11111111_{}.jpg".format(self.num))
#
#         return support_features, target_features
#
#     def forward(self, support_images, support_labels, target_images):
#         """
#         Should return a dict containing logits which are required for computing accuracy. Dict can also contain
#         other info needed to compute the loss. E.g. inter class distances.
#         """
#         raise NotImplementedError
#
#     def distribute_model(self):
#         if self.args.num_gpus > 1:
#             self.backbone = self.backbone.to(self.device)
#             self.backbone = torch.nn.DataParallel(self.backbone)
#             self.backbone.cuda(0)
#
#     def loss(self, task_dict, model_dict):
#         """
#         Takes in a the task dict containing labels etc.
#         Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
#         Default is cross entropy loss.
#         """
#         return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
#
#
# class PositionalEncoding(nn.Module):
#     """
#     Positional encoding from the Transformer paper.
#     """
#
#     def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.pe_scale_factor = pe_scale_factor
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
#         pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return self.dropout(x)
#
#
# class TemporalCrossTransformer(nn.Module):
#     """
#     A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
#     """
#
#     def __init__(self, args, temporal_set_size=3):
#         super(TemporalCrossTransformer, self).__init__()
#
#         self.args = args
#         self.temporal_set_size = temporal_set_size
#
#         max_len = int(self.args.seq_len * 1.5)
#         self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)
#
#         self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
#                                   self.args.trans_linear_out_dim).cuda()
#         self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
#                                   self.args.trans_linear_out_dim)  # .cuda()
#
#         self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
#         self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
#
#         self.class_softmax = torch.nn.Softmax(dim=1)
#
#         # generate all tuples
#         frame_idxs = [i for i in range(self.args.seq_len)]
#         frame_combinations = combinations(frame_idxs, temporal_set_size)
#         self.tuples = nn.ParameterList(
#             [nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
#         self.tuples_len = len(self.tuples)
#
#     def forward(self, support_set, support_labels, queries):
#         n_queries = queries.shape[0]
#         n_support = support_set.shape[0]
#
#         # static pe
#         support_set = self.pe(support_set)
#         queries = self.pe(queries)
#
#         # construct new queries and support set made of tuples of images after pe
#         s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
#         q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
#         support_set = torch.stack(s, dim=-2)
#         queries = torch.stack(q, dim=-2)
#
#         # apply linear maps
#         support_set_ks = self.k_linear(support_set)
#         queries_ks = self.k_linear(queries)
#         support_set_vs = self.v_linear(support_set)
#         queries_vs = self.v_linear(queries)
#
#         # apply norms where necessary
#         mh_support_set_ks = self.norm_k(support_set_ks)
#         mh_queries_ks = self.norm_k(queries_ks)
#         mh_support_set_vs = support_set_vs
#         mh_queries_vs = queries_vs
#
#         unique_labels = torch.unique(support_labels)
#
#         # init tensor to hold distances between every support tuple and every target tuple
#         all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)
#
#         for label_idx, c in enumerate(unique_labels):
#             # select keys and values for just this class
#             class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
#             class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
#             k_bs = class_k.shape[0]
#
#             class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
#                 self.args.trans_linear_out_dim)
#
#             # reshape etc. to apply a softmax for each query tuple
#             class_scores = class_scores.permute(0, 2, 1, 3)
#             class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
#             class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
#             class_scores = torch.cat(class_scores)
#             class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
#             class_scores = class_scores.permute(0, 2, 1, 3)
#
#             # get query specific class prototype
#             query_prototype = torch.matmul(class_scores, class_v)
#             query_prototype = torch.sum(query_prototype, dim=1)
#
#             # calculate distances from queries to query-specific class prototypes
#             diff = mh_queries_vs - query_prototype
#             norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
#             distance = torch.div(norm_sq, self.tuples_len)
#
#             # multiply by -1 to get logits
#             distance = distance * -1
#             c_idx = c.long()
#             all_distances_tensor[:, c_idx] = distance
#
#         return_dict = {'logits': all_distances_tensor}
#
#         return return_dict
#
#
# class CNN_TRX(CNN_FSHead):
#     """
#     Backbone connected to Temporal Cross Transformers of multiple cardinalities.
#     """
#
#     def __init__(self, args):
#         super(CNN_TRX, self).__init__(args)
#         self.NUM_SAMPLES = 1
#
#         self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])
#
#     def forward(self, support_images, support_labels, target_images):
#         support_features, target_features = self.get_feats(support_images, target_images)
#         all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
#         all_logits = torch.stack(all_logits, dim=-1)
#         sample_logits = all_logits
#         sample_logits = torch.mean(sample_logits, dim=[-1])
#
#         return_dict = {'logits': split_first_dim_linear(sample_logits, [self.NUM_SAMPLES, target_features.shape[0]])}
#         return return_dict
#
#
# class Mutildimension_attention(nn.Module):
#     def __init__(self, args, temporal_set_size=3):
#         super(Mutildimension_attention, self).__init__()
#         self.args = args
#         self.cos = torch.nn.CosineSimilarity()
#
#         self.n_dim = self.args.shot
#         self.num = 0
#         # self.dim_h = (self.args.trans_linear_in_dim * temporal_set_size)
#
#         self.temporal_set_size = temporal_set_size
#
#         max_len = int(self.args.seq_len * 1.5)  # seq_len = 8 frame per videos
#         self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)
#
#         self.W_Q = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
#                              self.args.trans_linear_in_dim * temporal_set_size)
#         self.W_K = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
#                              self.args.trans_linear_in_dim * temporal_set_size)
#         self.W_V = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
#                              self.args.trans_linear_in_dim * temporal_set_size)
#
#         self.norm_k = nn.LayerNorm(self.args.trans_linear_in_dim * temporal_set_size)
#         self.norm_v = nn.LayerNorm(self.args.trans_linear_in_dim * temporal_set_size)
#
#         self.class_softmax = torch.nn.Softmax(dim=-1)
#
#         frame_idxs = [i for i in range(self.args.seq_len)]
#         frame_combinations = combinations(frame_idxs, temporal_set_size)  # tuples eight frame
#         self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
#         self.tuples_len = len(self.tuples)
#
#     def forward(self, support_set, support_labels, queries):
#         n_queries = queries.shape[0]
#         n_support = support_set.shape[0]
#
#         support_set = self.pe(support_set)  # [25,8,2048]
#         queries = self.pe(queries)
#
#         s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
#         q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
#
#         support_set = torch.stack(s, dim=-2)
#         queries = torch.stack(q, dim=-2)
#         # print("===============support_set==================" + str(support_set.size()))
#
#         support_ks = self.norm_k(self.W_K(support_set))
#         queries_qs = self.norm_k(self.W_Q(queries))  # [25,56,1535]
#
#         support_vs = self.norm_k(self.W_V(support_set))
#         queries_vs = self.W_V(queries)  # [25,56,1535]
#
#         unique_labels = torch.unique(support_labels)
#
#         all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()
#
#         sample = []
#
#         sup = []
#         sup2 = []
#         label = []
#
#         for label_idx, c in enumerate(unique_labels):
#
#             all_prototype1 = 0
#             all_prototype2 = 0
#
#             class_k = torch.index_select(support_set, 0, extract_class_indices(support_labels, c))  # [5,56,1535]
#             sample.append(class_k)
#             class_v = torch.index_select(support_set, 0, extract_class_indices(support_labels, c))  # [5,56,1535]
#
#             class_n_k = torch.chunk(class_k, dim=0, chunks=self.n_dim)
#             class_n_v = torch.chunk(class_v, dim=0, chunks=self.n_dim)
#
#             # cross-attention1
#             for ids, k_ch in enumerate(class_n_k):
#                 if self.args.shot == 1:
#                     q_set = class_k
#                 else:
#                     if ids == 0:
#                         q_set = torch.narrow(class_k, dim=0, start=ids + 1, length=self.args.shot - 1)
#                     elif ids > 0 and ids < self.args.shot - 1:
#                         q_set1 = torch.narrow(class_k, dim=0, start=0, length=ids)
#                         q_set2 = torch.narrow(class_k, dim=0, start=ids + 1, length=self.args.shot - 1 - ids)
#                         q_set = torch.cat((q_set1, q_set2), dim=0)
#                     else:
#                         q_set = torch.narrow(class_k, dim=0, start=0, length=self.args.shot - 1)
#
#                 # print("+++++++++++q_set++++++++++++++++"+str(q_set))
#                 k_ch = k_ch.squeeze(0)
#                 k_ch = self.norm_k(self.W_K(k_ch))
#                 sup.append(k_ch)
#                 v_ch = self.norm_k(self.W_V(class_n_v[ids]))
#                 # v_ch = class_n_v[ids]
#                 class_sco = torch.matmul(q_set, k_ch.transpose(-2, -1)) / math.sqrt(
#                     self.args.trans_linear_in_dim * self.temporal_set_size)
#                 class_sco = torch.mean(class_sco, dim=0)
#                 class_sco = self.class_softmax(class_sco)
#                 prototype = torch.matmul(class_sco, v_ch)
#                 all_prototype1 += prototype
#                 # print("++++++++++++++++all_prototype1+++++++++++++++++"+str(all_prototype1.size()))
#
#             # cross-atention2
#             for idx, k_chunk in enumerate(class_n_k):
#                 k_chunk = k_chunk.squeeze(0)
#                 k_chunk = self.norm_k(self.W_K(k_chunk))
#                 # sup.append(k_chunk)
#                 v_chunk = self.norm_k(self.W_V(class_n_v[idx]))
#                 class_scores = torch.matmul(queries_qs, k_chunk.transpose(-2, -1)) / math.sqrt(
#                     self.args.trans_linear_in_dim * self.temporal_set_size)  # [25, 56, 56]
#                 class_scores = self.class_softmax(class_scores)
#                 query_prototype = torch.matmul(class_scores, v_chunk)
#                 # print("====================query_prototype================="+str(query_prototype.size()))
#                 all_prototype2 += query_prototype  # [25，56, 1535]
#                 # print("++++++++++++++++all_prototype12+++++++++++++++++" + str(all_prototype2.size()))
#
#             sim_result1 = cosine_similarity(all_prototype1.squeeze(dim=0), queries_vs)  # [25,56]
#             sim_result1 = torch.mean(sim_result1, dim=1) * 1000
#
#             sim_result2 = self.cos(all_prototype2, queries_vs)
#             sim_result2 = torch.mean(sim_result2, dim=1) * 1000
#             # print("===========sim_result==================" + str(sim_result.size()))
#             # sim_result = torch.mean(sim_result, dim=1)*1000
#             sim_result = sim_result1 + sim_result2
#             # print("===========sim_result==================" + str(sim_result))
#             c_idx = c.long()
#             all_distances_tensor[:, c_idx] = sim_result
#
#             sup = torch.stack(sup)
#             sup2.append(sup)
#             sup = []
#
#         self.num = self.num + 1
#         if self.num % 200 == 0:
#             sup2 = torch.stack(sup2)
#             print(sup2.size())
#             feat = sup2.reshape(25, -1).cpu().detach().numpy()
#
#             fig = plt.figure(figsize=(10, 10))
#
#             label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
#             label = np.array(label)
#             plotlabels(visual(feat), label, '(b)')
#
#             plt.savefig("./pict/2222_{}.jpg".format(self.num))
#
#         return_dict = {'logits': all_distances_tensor}
#
#         return return_dict
#
#
# class CNN_Mutildimension(CNN_FSHead):
#
#     def __init__(self, args):
#         super(CNN_Mutildimension, self).__init__(args)
#
#         self.dis_list = 0
#         self.num = 0
#
#         self.NUM_SAMPLES = 1
#         self.model = nn.ModuleList([Mutildimension_attention(args, s) for s in args.temp_set])
#
#     def forward(self, context_images, context_labels, target_images):
#         context_features, target_features = self.get_feats(context_images, target_images, context_labels)
#         # np.save("context_features",context_features.cpu().numpy())
#         # np.save("context_labels",context_labels.cpu().numpy())
#
#         model_dict = [t(context_features, context_labels, target_features) for t in self.model]
#
#         all_logits = [model_dict[i]['logits'] for i in range(len(model_dict))]
#         all_logits = torch.stack(all_logits, dim=-1)
#         sample_logits = all_logits
#         sample_logits = torch.mean(sample_logits, dim=[-1])
#
#         return_dict = {'logits': split_first_dim_linear(sample_logits, [self.NUM_SAMPLES, target_features.shape[0]])}
#         return return_dict
#
#
# class Ave_Pro(nn.Module):
#     def __init__(self, args, temporal_set_size=3):
#         super(Ave_Pro, self).__init__()
#
#         self.args = args
#
#         max_len = int(self.args.seq_len * 1.5)
#         self.NUM_SAMPLES = 1
#         self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)
#
#         self.frame_idxs = [i for i in range(self.args.seq_len)]
#         self.frame_combinations = combinations(self.frame_idxs, temporal_set_size)
#         self.tuples = [torch.tensor(comb).cuda() for comb in self.frame_combinations]
#         self.tuples_len = len(self.tuples)
#
#     def forward(self, support_set, support_labels, queries):
#         unique_labels = torch.unique(support_labels)
#
#         n_support = support_set.shape[0]
#         n_queries = queries.shape[0]
#
#         support_features = self.pe(support_set)
#         query_features = self.pe(queries)
#
#         s = [torch.index_select(support_features, -2, p).reshape(n_support, -1) for p in self.tuples]
#         q = [torch.index_select(query_features, -2, p).reshape(n_queries, -1) for p in self.tuples]
#         #
#         support_features = torch.stack(s, dim=-2)
#         query_features = torch.stack(q, dim=-2)
#
#         all_simily_tensor = torch.zeros(n_queries, self.args.way).cuda()
#
#         for label_idx, c in enumerate(unique_labels):
#             class_p = torch.index_select(support_features, 0, extract_class_indices(support_labels, c))  # [5,56,1536]
#             class_prototypes = torch.mean(class_p, dim=0)  # [56,1536]
#             # class_prototypes = torch.mean(class_p,dim=1)#[5,512]
#
#             # target_features = torch.mean(target_features,dim=1)#[25,1536]
#
#             sim_result = cosine_similarity(class_prototypes, query_features)  # [25,56]
#             sim_result = torch.mean(sim_result, dim=1) * 1000
#
#             c_idx = c.long()
#             all_simily_tensor[:, c_idx] = sim_result
#             # all_simily = torch.stack(all_simily_tensor)
#             # print("++++++++++all_simily+++++++++++++" + str(all_simily.size()))
#
#         return_dict = {'logits': all_simily_tensor}
#         return return_dict
#
#
# class CNN_Pro(CNN_FSHead):
#
#     def __init__(self, args):
#         super(CNN_Pro, self).__init__(args)
#
#         self.NUM_SAMPLES = 1
#         self.model = nn.ModuleList([Ave_Pro(args, s) for s in args.temp_set])
#
#     def forward(self, context_images, context_labels, target_images):
#         context_features, target_features = self.get_feats(context_images, target_images)
#
#         all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.model]
#         all_logits = torch.stack(all_logits, dim=-1)
#         sample_logits = all_logits
#         sample_logits = torch.mean(sample_logits, dim=[-1])
#
#         return_dict = {'logits': split_first_dim_linear(sample_logits, [self.NUM_SAMPLES, target_features.shape[0]])}
#         return return_dict
#
#
s = "b"
# t = "c"
# t = list(t)
# s = list(s)
# length1 = len(t)
# length2 = len(s)
# res = []
# for i in range(0,length2):
#     for j in range(0,length1):
#         print(t[j])
#         if t[j] == s[i]:
#             res.append(t[j])
#             break
# print(res)
# for i in range(length2):
#     if s[i] != res[i]:
#         print(" ")

