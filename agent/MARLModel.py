import math

import torch
import torch.nn as nn
from env.MultiAgentEnv import UAVActionRet
from util.obs_process import build_batch, buildStateTensor
from util.get_device import get_device

# Set device
device = get_device()


class UAVCritics(nn.Module):
    def __init__(self, feature_dim=3, embed_dim=128, dropout=0.1, nhead=4):
        super(UAVCritics, self).__init__()
        self.linear_proj = nn.Linear(feature_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                        nhead=nhead,
                                                                        dropout=dropout,
                                                                        batch_first=True), num_layers=3)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=embed_dim,
                                                                        nhead=nhead,
                                                                        dropout=dropout,
                                                                        batch_first=True), num_layers=3)

    def forward(self, obs):
        """
        Args:
            obs: Dictionary of observation tensors.
        Returns:
            value: Tensor of shape (batch, 1) with the state value estimate.
        """
        batch = build_batch([next(iter(value for key, value in observation.items() if key.startswith("uav"))) for observation in obs])
        state_tensor = buildStateTensor(batch, device=device)
        linear_proj = self.linear_proj(state_tensor)
        encoder_out = self.encoder(linear_proj)
        decoder_out = self.decoder(encoder_out, memory=encoder_out)
        return decoder_out.mean(dim=(1,2), keepdim=True).squeeze(2)


class LSTMCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        # input_dim = feature_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Map the last hidden state to a scalar value
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        x: shape [batch_size, seq_length, feature_dim]
        Returns a value estimate of shape [batch_size, 1]
        """
        batch = build_batch(obs)
        x = buildStateTensor(batch, device=device)
        # out: [batch_size, seq_length, hidden_dim]
        # (h_n, c_n): last hidden & cell states, shape of h_n is [num_layers, batch_size, hidden_dim]
        out, (h_n, c_n) = self.lstm(x)

        # Take the top layer’s final hidden state: h_n[-1] => [batch_size, hidden_dim]
        last_hidden = h_n[-1]
        value = self.fc(last_hidden)  # [batch_size, 1]
        return value


class DecisionAttention(nn.Module):
    def __init__(self, embed_dim=128):
        super(DecisionAttention, self).__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.comb_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dist_linear = nn.Linear(1, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, dist, hn, proj, embed):
        query = hn[-1].unsqueeze(0)
        combined = torch.cat([query, proj, embed], dim=0)
        combined = self.linear(combined).sum(dim=0, keepdim=True)
        combined_expanded = combined.unsqueeze(1).repeat(1, dist.size(1), 1)
        combined_expanded = self.comb_linear(combined_expanded)  # [1, 21, 128]
        dist_expanded = dist.unsqueeze(-1)  # [1, 21, 1]
        dist_emb = self.dist_linear(dist_expanded)  # [1, 21, 128]
        attn_input = torch.tanh(combined_expanded + dist_emb)  # [1, 21, 128]
        scores = self.v(attn_input).squeeze(-1)  # [1, 21]
        # scores = F.softmax(scores, dim=1)
        return scores  # softmax is added later


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                      nhead=nhead,
                                                      dropout=dropout,
                                                      batch_first=True)
        # self.feed_forward = nn.Linear(embed_dim, embed_dim)
        # self.bn1 = nn.BatchNorm1d(embed_dim)
        # self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        after_transformer = self.transformer(x)
        return after_transformer
        # add1 = x + after_transformer
        # bn1 = self.bn1(add1.transpose(1, 2))
        # bn1 = bn1.transpose(1, 2)
        # after_feat = self.feed_forward(bn1)
        # add2 = bn1 + after_feat
        # bn2 = self.bn2(add2.transpose(1, 2))
        # return bn2.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 nhead=4,
                 num_layers=2,
                 feature_dim=3,
                 dropout=0.1,
                 ):
        super(Encoder, self).__init__()
        self.linear_proj = nn.Linear(feature_dim, embed_dim)
        self.attention = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, batch):
        candidates = buildStateTensor(batch, device=device)
        linear_proj = self.linear_proj(candidates)  # (batch, num_candidates, candidate_embed_dim)
        embed = self.attention(linear_proj)
        return embed


class Decoder(nn.Module):
    def __init__(self, embed_dim=128, num_layers=3, hidden_dim=128):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.status_linear_proj = nn.Linear(3, embed_dim)
        self.attention = DecisionAttention(embed_dim)

    def forward(self, embed_last, embed_mean, batch, latest):
        nodes = torch.cat([batch["nodes"], batch["truck"].unsqueeze(1)], dim=1).to(device)
        dist_vec = torch.cdist(nodes, nodes, p=2)
        dist_vec = dist_vec[torch.arange(dist_vec.size(0)), latest, :]
        status = torch.cat([batch["power"].unsqueeze(-1), batch["capacity"].unsqueeze(-1), batch["travel_distance"].unsqueeze(-1)], dim=-1).to(device)
        proj = self.status_linear_proj(status)
        lstm_out, (h_n, c_n) = self.lstm(embed_last)
        # print(f"dist_vec: {dist_vec.shape}, proj: {proj.shape}, hn: {h_n.shape}, embed_mean: {embed_mean.shape}")
        score = self.attention(dist_vec, h_n, proj, embed_mean)
        return score


class ImprovedDecoder(nn.Module):
    """
    改进版的 Decoder 模块：基于编码器输出、代理状态信息和附加约束条件生成动作打分。

    主要思路：
     - 通过一个全连接层将代理状态（例如 power、capacity、travel_distance）映射到与 embedding 相同的空间。
     - 利用多头注意力机制，将代理状态作为查询（query），候选点的 embedding 作为键（key）和值（value），计算注意力输出。
     - 将注意力输出与候选点的原始 embedding 融合后，通过前馈神经网络计算每个候选动作（例如候选点）的打分（logits）。

    Args:
        embed_dim (int): embedding 的维度（默认 128）。
        num_heads (int): 多头注意力的头数（默认 4）。
        ff_hidden_dim (int): 前馈网络的隐藏层维度（默认 256）。
        dropout (float): dropout 概率（默认 0.1）。
    """

    def __init__(self, num_agent=1, embed_dim=128, num_heads=4, ff_hidden_dim=256, dropout=0.1):
        super(ImprovedDecoder, self).__init__()
        # 将代理状态（3 维：power, capacity, travel_distance）投影到 embed_dim 空间
        self.state_proj = nn.Linear(3, embed_dim)
        self.agent_proj = nn.Linear(num_agent, embed_dim)
        # 进一步转换状态向量，为后续作为注意力查询作准备
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        # 多头注意力层：查询来自代理状态，键和值来自 encoder 输出
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 前馈网络，用于将融合后的 embedding 生成候选动作对应的打分（logit）
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, 1)
        )

    def forward(self, encoder_outputs, batch, agent, agent_state_extra=None):
        """
        Args:
            encoder_outputs (Tensor): 编码器的输出，形状 (batch, num_candidates, embed_dim)。
            batch (dict): 包含代理信息的字典，要求至少包含下列键：
                - "power": Tensor, 形状 (batch,)（代理剩余电量）
                - "capacity": Tensor, 形状 (batch,)（代理载重或任务容量）
                - "travel_distance": Tensor, 形状 (batch,)（代理已行驶距离或耗能指标）
            agent_state_extra (Tensor, optional): 额外的代理状态信息，形状 (batch, embed_dim)。
                用于进一步融合更多约束条件，默认不使用（None）。
        Returns:
            logits (Tensor): 每个候选动作的打分，形状 (batch, num_candidates)。
            attn_weights (Tensor): 多头注意力层输出的注意力权重。
        """
        # 将代理状态信息（power, capacity, travel_distance）拼接后投影到 embedding 空间
        state_features = torch.stack(
            [batch["power"], batch["capacity"], batch["travel_distance"]], dim=1).to(
             encoder_outputs.device)
        state_embed = self.state_proj(state_features)  # shape: (batch, embed_dim)
        agent_embed = self.agent_proj(torch.Tensor(agent).unsqueeze(1).to(encoder_outputs.device)).unsqueeze(1)  # shape: (batch, 1, embed_dim)
        query = self.query_proj(state_embed).unsqueeze(1)  # shape: (batch, 1, embed_dim)
        query = query + agent_embed

        # 如果有额外的代理状态信息，可以将其简单融合进查询向量中
        if agent_state_extra is not None:
            query = query + agent_state_extra.unsqueeze(1)

        # 利用多头注意力计算：查询为代理状态，键和值为候选 embedding
        attn_output, attn_weights = self.multihead_attn(query, encoder_outputs, encoder_outputs)
        # 扩展 attention 输出，使其与候选 embedding 维度匹配
        batch_size, num_candidates, _ = encoder_outputs.shape
        attn_expanded = attn_output.expand(-1, num_candidates, -1)  # (batch, num_candidates, embed_dim)
        # 融合候选 embedding 和注意力输出（例如简单相加，可试验其他融合策略）
        combined = encoder_outputs + attn_expanded
        # 通过前馈网络产生每个候选动作的打分
        logits = self.feed_forward(combined).squeeze(-1)  # (batch, num_candidates)
        return logits, attn_weights


class UAVActor(nn.Module):
    """
    UAVActor 模型：
    - 首先利用 Encoder 对候选点进行编码，
    - 然后通过 ImprovedDecoder 利用代理状态信息计算各候选动作的打分，
    - 最后将得分除以一个缩放系数并对不可行动作进行 mask。
    """
    def __init__(self, feature_dim=3, embed_dim=128, num_heads=4, ff_hidden_dim=256, mask_all=True):
        super(UAVActor, self).__init__()
        self.encoder = Encoder(feature_dim=feature_dim, embed_dim=embed_dim, nhead=num_heads, num_layers=2)
        self.decoder = ImprovedDecoder(embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim)
        self.scale = math.sqrt(embed_dim)
        self.mask_all = mask_all

    def forward(self, obs, agents):
        """
        Args:
            obs: 环境观察数据，类型与 build_batch 函数定义相同。
        Returns:
            logits (Tensor): 每个候选动作的打分，形状 (batch, num_candidates)。
        """
        obs = [observation[f"uav_0_{agent}"] for observation, agent in zip(obs, agents)]
        batch = build_batch(obs)
        # 对候选状态进行编码
        x = self.encoder(batch)  # (batch, num_candidates, embed_dim)
        # 利用改进版 decoder 得到候选动作得分
        logits, attn = self.decoder(x, batch, agents)
        # 缩放打分
        logits = logits / self.scale
        # 根据环境中 choice_mask 对不可行动作进行 mask
        if self.mask_all:
            infeasible = torch.Tensor(batch["choice_mask"] != UAVActionRet.FEASIBLE.value).to(x.device)
        else:
            # 如有其他 mask 逻辑，可在此进行扩展
            infeasible = torch.logical_and(torch.Tensor(batch["choice_mask"] == UAVActionRet.CLOSED_NODE.value),
                                           torch.Tensor(batch["choice_mask"] == UAVActionRet.SAME_TARGET.value)).to(x.device)
        logits = logits.masked_fill(infeasible, -1e9)
        return logits



if __name__ == "__main__":
    from env.VecEnv import SubprocVectorizedMultiAgentEnv
    from util.make_env import make_env

    config = {
        "uav_num": 1,
        "group_num": 1,
        "uav_velocity": 3,
        "max_step": 1000,
        "num_customer": 10,
        "space_width": 10,
        "space_height": 10,
        "cluster_number": 2,
        "render_mode": "rgb_array"
    }
    num_envs = 2
    envs = [make_env(config) for _ in range(num_envs)]
    parallel_env = SubprocVectorizedMultiAgentEnv(envs)
    obs, rewards, dones, _, _ = parallel_env.init()
    batch_ = build_batch(obs)
    tensor = buildStateTensor(batch_, device=device)

    lstm_critic = LSTMCritic(input_dim=tensor.shape[2], hidden_dim=128).to(device)
    critic = UAVCritics(feature_dim=tensor.shape[2], embed_dim=128).to(device)

    # actor = UAVActor(feature_dim=tensor.shape[2], embed_dim=128, attention_nhead=4, attention_num_layers=2,
    #                  lstm_num_layers=2, lstm_hidden_dim=128).to(device)

    print(lstm_critic(obs))
    print(critic(obs))
    exit()

    score = actor(obs)
    # score = F.softmax(score)
    dist = torch.distributions.Categorical(logits=score)
    action = dist.sample()
    logits = dist.logits
    neg_log_prob = - dist.log_prob(action)

    # print(score)
    # print(action)
    # print(logits)
    # print(neg_log_prob)

    # encoder = Encoder(feature_dim=tensor.shape[2], embed_dim=128, nhead=4, num_layers=2).to(device)
    latest = torch.argmax(
        (batch_["choice_mask"].clone().to(device) == UAVActionRet.SAME_TARGET.value).long(),
        dim=1
    )
    print(batch_["choice_mask"])
    print(latest)
    # latest = torch.argmax(batch_["choice_mask"].clone().to(device) == UAVActionRet.SAME_TARGET.value, dim=1)
    # x = encoder(batch_)
    # x_mean = x.mean(dim=1)
    # latest_embed = x[torch.arange(x.size(0)), latest, :]
    # #
    # decoder = Decoder(embed_dim=128, num_layers=3, hidden_dim=128).to(device)
    # score = decoder(latest_embed, x_mean, batch_, latest)

    # status = torch.cat([batch["power"].unsqueeze(-1), batch["capacity"].unsqueeze(-1), batch["travel_distance"].unsqueeze(-1)], dim=-1).to(device)
    # nodes = torch.cat([batch["nodes"], batch["truck"].unsqueeze(1)], dim=1).to(device)
    # dist_vec = torch.cdist(nodes, nodes, p=2)
    # dist_vec = dist_vec[torch.arange(dist_vec.size(0)), latest, :]

