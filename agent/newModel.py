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
        batch = build_batch(obs)
        state_tensor = buildStateTensor(batch, device=device)
        linear_proj = self.linear_proj(state_tensor)
        encoder_out = self.encoder(linear_proj)
        decoder_out = self.decoder(encoder_out, memory=encoder_out)
        return decoder_out.mean()


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


class UAVActor(nn.Module):
    def __init__(self,
                 feature_dim=3,
                 embed_dim=128,
                 attention_nhead=4,
                 attention_num_layers=2,
                 lstm_num_layers=2,
                 lstm_hidden_dim=128,
                 ):
        super(UAVActor, self).__init__()
        self.encoder = Encoder(feature_dim=feature_dim, embed_dim=embed_dim, nhead=attention_nhead,
                               num_layers=attention_num_layers)
        self.decoder = Decoder(embed_dim, lstm_num_layers, lstm_hidden_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, obs):
        batch = build_batch(obs)
        latest = torch.argmax(batch["choice_mask"].clone().to(device) == UAVActionRet.SAME_TARGET.value, dim=1)
        x = self.encoder(batch)
        x_mean = x.mean(dim=1)
        # print(x_mean.shape)  # (batch, candidate_embed_dim)
        last_target = x[torch.arange(x.size(0)), latest, :]
        score = self.decoder(last_target, x_mean, batch, latest) / self.scale

        # masked out infeasible actions
        # infeasible = torch.logical_and(batch["choice_mask"] == UAVActionRet.CLOSED_NODE.value,
        #                                batch["choice_mask"] == UAVActionRet.SAME_TARGET.value)
        infeasible = torch.Tensor(batch['choice_mask'] != UAVActionRet.FEASIBLE.value).to(device)
        score = score.masked_fill(infeasible, -1e9)
        # score = F.softmax(score, dim=-1)  # softmax is added in the Categorical distribution
        return score


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

    actor = UAVActor(feature_dim=tensor.shape[2], embed_dim=128, attention_nhead=4, attention_num_layers=2,
                     lstm_num_layers=2, lstm_hidden_dim=128).to(device)

    score = actor(obs)
    # score = F.softmax(score)
    dist = torch.distributions.Categorical(logits=score)
    action = dist.sample()
    logits = dist.logits
    neg_log_prob = - dist.log_prob(action)

    print(score)
    print(action)
    print(logits)
    print(neg_log_prob)

    # encoder = Encoder(feature_dim=tensor.shape[2], embed_dim=128, nhead=4, num_layers=2).to(device)
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

