# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MultiLevelAttentionCNN(nn.Module):
    def __init__(self, opt, padding_idx=0, embedding=None):
        super(MultiLevelAttentionCNN, self).__init__()

        self.opt = opt

        # word embedding
        if opt["pretrained_words"]:
            assert embedding is not None
            self.word_embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt["fix_embedding"]:
                self.word_embedding.weight.requires_grad = False
        else:
            self.word_embedding = nn.Embedding(opt["vocab_size"], opt["embedding_dim"], padding_idx=padding_idx)

        self.dw = opt["embedding_dim"]
        self.dp = opt["position_embedding_dim"]
        self.d = self.dw + self.dp * 2

        self.k = opt["sliding_window"]
        self.kd = self.k * self.d

        self.np = opt["num_positions"]
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)

        self.p = (self.k - 1) // 2
        self.pad = nn.ConstantPad2d((0, 0, self.p, self.p), 0)

        self.We1 = nn.Parameter(torch.randn(self.dw, self.dw))
        self.We2 = nn.Parameter(torch.randn(self.dw, self.dw))

        self.dc = opt["num_filters"]
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.dc, kernel_size=(1, self.kd), stride=(1, self.kd))

        self.tanh = nn.Tanh()

        self.nr = opt["num_relations"]
        self.y_embbeding = nn.Embedding(self.nr, self.dc)
        self.U = nn.Parameter(torch.randn(self.dc, self.nr))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, self.dc), stride=(1, self.dc))

        self.dropout = opt["dropout"]

    def forward(self, x, e1, e2, dist1, dist2):
        # 1.input representation
        x_embed = self.word_embedding(x)  # [b, sl, dw]
        dist1_embed = self.dist1_embedding(dist1)
        dist2_embed = self.dist2_embedding(dist2)
        x_concat = torch.cat([x_embed, dist1_embed, dist2_embed], dim=2)  # [b,sl,word_embed_size+dist_embed_size*2]

        # 2. window concat
        b, sl, wl = x_concat.size()
        px = self.pad(x_concat.unsqueeze(1))  # [b, 1, sl+p*2, wl]
        px = px.view(b, sl+self.p*2, wl)  # [b, sl+p*2, wl]

        if torch.cuda.is_available():
            t_px = torch.index_select(px, dim=1, index=torch.arange(sl).cuda())
            m_px = torch.index_select(px, dim=1, index=torch.arange(1, sl+1).cuda())
            b_px = torch.index_select(px, dim=1, index=torch.arange(2, sl+2).cuda())
        else:
            t_px = torch.index_select(px, dim=1, index=torch.arange(sl))
            m_px = torch.index_select(px, dim=1, index=torch.arange(1, sl+1))
            b_px = torch.index_select(px, dim=1, index=torch.arange(2, sl+2))

        window_cat = torch.cat([t_px, m_px, b_px], dim=2)  # [b,sl,k*wl]
        window_cat = F.dropout(window_cat, p=self.dropout, training=self.training)

        # 3. contextual relevance matrices
        e1_embed = self.word_embedding(e1).unsqueeze(1)  # [b, 1, dw]
        e2_embed = self.word_embedding(e2).unsqueeze(1)
        W1 = self.We1.view(1, self.dw, self.dw).repeat(b, 1, 1)  # [b,dw,dw]
        W2 = self.We2.view(1, self.dw, self.dw).repeat(b, 1, 1)
        W1x = torch.bmm(x_embed, W1)  # [b, sl, dw]
        W2x = torch.bmm(x_embed, W2)
        A1 = torch.bmm(W1x, e1_embed.transpose(1, 2)).squeeze(2)  # [b, sl, dw] * [b, dw, 1] = [b, sl, 1]
        A2 = torch.bmm(W2x, e2_embed.transpose(1, 2)).squeeze(2)

        # 3.1 input attention composition
        alpha1 = F.softmax(A1, dim=1)
        alpha2 = F.softmax(A2, dim=1)
        alpha = torch.div(torch.add(alpha1, alpha2), 2)  # [b, sl, 1]
        alpha = alpha.unsqueeze(2).repeat(1, 1, self.kd)  # [b, sl, k*wl] wl:word_embed_size+dist_embed_size*2

        R = torch.mul(window_cat, alpha)  # [b, sl, k*wl]

        # 4. convolution layer
        R_ = self.conv(R.unsqueeze(1))  # [b,1,sl,kd] -> [b,dc,sl,1]
        R_ = self.tanh(R_)
        R_star = R_.squeeze(-1)  # [b,dc,sl]

        # 5. attentive pooling
        rel_weight = self.y_embbeding.weight  # [nr, dc]
        b_rel_weight = rel_weight.unsqueeze(0).repeat(b, 1, 1)  # [b, nr, dc]

        b_U = self.U.unsqueeze(0).repeat(b, 1, 1)  # [b, dc, nr]

        G = torch.bmm(R_star.transpose(1, 2), b_U)  # [b, sl, nr]
        G = torch.bmm(G, b_rel_weight)  # [b, sl, dc]

        AP = F.softmax(G, dim=2)
        wo = torch.bmm(R_star, AP)  # [b, dc, dc]
        wo = self.max_pool(wo.unsqueeze(1))  # [b, 1, dc, 1]
        wo = wo.squeeze(-1).squeeze(1)  # [b, dc]

        wo_norm = F.normalize(wo)  # [b, dc] Wo/||Wo||
        b, dc = wo_norm.size()
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.opt["num_relations"], 1)  # [b, nr, dc]
        b_rel_weight_norm = F.normalize(rel_weight).unsqueeze(0).repeat(b, 1, 1)  # [b, nr, dc]
        all_distance = torch.norm(wo_norm_tile - b_rel_weight_norm, p=2, dim=2)  # [b, nr]

        predict_prob, predict = torch.min(all_distance, dim=1)

        if self.training:
            return all_distance, wo_norm, rel_weight, predict

        else:
            return predict
















