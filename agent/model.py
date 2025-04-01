import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################
# Environment: A simplified TSP-D simulator #
#############################################

class TSPDEnv:
    def __init__(self, num_nodes=10, alpha=2.0):
        """
        A simplified TSP-D environment.
        Args:
            num_nodes (int): Total number of nodes (including depot, index 0).
            alpha (float): Drone speed factor (drone time = distance/alpha).
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.coords = None  # (num_nodes, 2)
        self.reset()

    def reset(self):
        # Generate random coordinates.
        # Depot (node 0) is fixed in the lower left corner.
        depot = np.array([[0.0, 0.0]])
        # Other nodes uniformly in [0, 1] x [0, 1]
        customers = np.random.rand(self.num_nodes - 1, 2)
        self.coords = np.concatenate([depot, customers], axis=0)
        # State: both truck and drone start at depot (index 0)
        self.truck_pos = 0
        self.drone_pos = 0
        # Keep track of visited nodes (boolean mask); depot is visited.
        self.visited = np.zeros(self.num_nodes, dtype=bool)
        self.visited[0] = True
        # Total time for each vehicle (they rendezvous at each step)
        self.time = 0.0
        return self.get_state()

    def get_state(self):
        # Return the coordinates, current positions, and visited mask.
        state = {
            "coords": self.coords.copy(),  # numpy array (num_nodes, 2)
            "visited": self.visited.copy(),  # bool array (num_nodes,)
            "truck_pos": self.truck_pos,  # int
            "drone_pos": self.drone_pos,  # int
            "time": self.time  # float, current makespan so far
        }
        return state

    def euclidean(self, i, j):
        # Euclidean distance between nodes i and j.
        diff = self.coords[i] - self.coords[j]
        return np.linalg.norm(diff)

    def step(self, truck_action, drone_action):
        """
        Executes one joint decision step.
        Both truck and drone choose one new (unvisited) node each.
        If only one unvisited node remains, both vehicles go to that node.
        The travel time for the step is the maximum of (truck_distance, drone_distance/alpha),
        because vehicles must rendezvous.
        After the step, update positions and mark visited nodes.
        Returns:
            next_state, step_cost, done
        """
        unvisited = np.where(~self.visited)[0]
        if len(unvisited) == 0:
            # All nodes visited; now both return to depot.
            truck_return = self.euclidean(self.truck_pos, 0)
            drone_return = self.euclidean(self.drone_pos, 0) / self.alpha
            step_cost = max(truck_return, drone_return)
            self.time += step_cost
            self.truck_pos = 0
            self.drone_pos = 0
            done = True
            return self.get_state(), step_cost, done

        # If only one unvisited remains, force both actions to that node.
        if len(unvisited) == 1:
            truck_action = drone_action = unvisited[0]

        # Get travel distances:
        d_truck = self.euclidean(self.truck_pos, truck_action)
        d_drone = self.euclidean(self.drone_pos, drone_action)
        t_truck = d_truck  # truck speed = 1
        t_drone = d_drone / self.alpha
        step_cost = max(t_truck, t_drone)

        # Update global time to the rendezvous time.
        self.time += step_cost

        # Update positions: both vehicles rendezvous at their new positions.
        self.truck_pos = truck_action
        self.drone_pos = drone_action

        # Mark visited nodes (if both choose the same node, mark it only once)
        self.visited[truck_action] = True
        self.visited[drone_action] = True

        # Check if done: when all nodes visited and vehicles have returned in a later step.
        done = np.all(self.visited)
        return self.get_state(), step_cost, done

    def get_total_cost(self):
        return self.time


#############################################
# Model: Hybrid Encoder-Decoder (HM)        #
#############################################

class HybridModel(nn.Module):
    def __init__(self, embed_dim=128, num_layers=3, num_heads=8, lstm_hidden_dim=128):
        """
        Hybrid Model with a multi-head attention encoder and two LSTM decoders
        (one for truck and one for drone).
        """
        super(HybridModel, self).__init__()
        self.embed_dim = embed_dim

        # Input projection: from (x,y) to embed_dim.
        self.input_proj = nn.Linear(2, embed_dim)

        # Encoder: TransformerEncoder (we use batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 2,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder LSTMCells for truck and drone.
        self.truck_lstm = nn.LSTMCell(embed_dim, lstm_hidden_dim)
        self.drone_lstm = nn.LSTMCell(embed_dim, lstm_hidden_dim)

        # Query layers to compute decoder queries.
        self.truck_query = nn.Linear(lstm_hidden_dim, embed_dim)
        self.drone_query = nn.Linear(lstm_hidden_dim, embed_dim)

    def forward(self, coords, visited_mask, truck_last_idx, drone_last_idx,
                truck_hx, truck_cx, drone_hx, drone_cx):
        """
        Args:
            coords: Tensor of shape (batch, num_nodes, 2)
            visited_mask: Tensor of shape (batch, num_nodes) with 1 for visited, 0 for not.
            truck_last_idx: Tensor of shape (batch,) indices of truck's last visited node.
            drone_last_idx: Tensor of shape (batch,) indices of drone's last visited node.
            truck_hx, truck_cx: Tensors of shape (batch, lstm_hidden_dim) for truck LSTM state.
            drone_hx, drone_cx: Tensors of shape (batch, lstm_hidden_dim) for drone LSTM state.
        Returns:
            truck_logprobs, drone_logprobs: Tensors (batch, num_nodes) with logit scores (masked).
            Updated truck and drone LSTM states.
            Encoded node embeddings (batch, num_nodes, embed_dim) used for computing logits.
        """
        batch_size, num_nodes, _ = coords.size()

        # Project coordinates
        proj = self.input_proj(coords)  # (batch, num_nodes, embed_dim)
        # Encode the graph with multi-head attention encoder.
        h_enc = self.encoder(proj)  # (batch, num_nodes, embed_dim)

        # For decoder input, get the embedding of the last visited node for each vehicle.
        # Use gather along dim=1.
        truck_last_embed = h_enc.gather(1, truck_last_idx.view(batch_size, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(
            1)
        drone_last_embed = h_enc.gather(1, drone_last_idx.view(batch_size, 1, 1).expand(-1, 1, self.embed_dim)).squeeze(
            1)

        # Update truck LSTM state.
        truck_hx, truck_cx = self.truck_lstm(truck_last_embed, (truck_hx, truck_cx))
        # Update drone LSTM state.
        drone_hx, drone_cx = self.drone_lstm(drone_last_embed, (drone_hx, drone_cx))

        # Compute queries.
        truck_q = self.truck_query(truck_hx)  # (batch, embed_dim)
        drone_q = self.drone_query(drone_hx)  # (batch, embed_dim)

        # Compute logits as dot product between query and each node embedding.
        # (batch, num_nodes)
        truck_logits = torch.bmm(h_enc, truck_q.unsqueeze(2)).squeeze(2)
        drone_logits = torch.bmm(h_enc, drone_q.unsqueeze(2)).squeeze(2)

        # Mask out visited nodes by setting logits to -inf.
        neg_inf = -1e9
        truck_logits = truck_logits.masked_fill(visited_mask.bool(), neg_inf)
        drone_logits = drone_logits.masked_fill(visited_mask.bool(), neg_inf)

        # Compute probabilities.
        truck_probs = F.softmax(truck_logits, dim=-1)
        drone_probs = F.softmax(drone_logits, dim=-1)

        return truck_probs, drone_probs, truck_hx, truck_cx, drone_hx, drone_cx, h_enc


#############################################
# Critic Network                            #
#############################################

class Critic(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, graph_embed):
        """
        Args:
            graph_embed: Tensor of shape (batch, embed_dim) representing the graph encoding
                         (e.g. mean of encoder outputs).
        Returns:
            value estimate: Tensor of shape (batch, 1)
        """
        x = F.relu(self.fc1(graph_embed))
        value = self.fc2(x)
        return value


#############################################
# Training Code using A2C                 #
#############################################

def sample_episode(model, env, device):
    """
    Runs one episode (one TSP-D instance) and returns:
      - log probability sum (actor log prob)
      - total cost (makespan)
      - lists of (graph, visited mask, truck and drone last indices) if needed for critic input.
    For simplicity we treat one instance as one episode (batch size = 1).
    """
    state = env.reset()
    coords = torch.tensor(state["coords"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, num_nodes, 2)
    visited = torch.tensor(state["visited"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, num_nodes)
    # Initially, both vehicles are at depot (index 0)
    truck_last = torch.zeros(1, dtype=torch.long, device=device)
    drone_last = torch.zeros(1, dtype=torch.long, device=device)
    # Initialize LSTM states (for truck and drone)
    lstm_hidden_dim = model.truck_lstm.hidden_size
    truck_hx = torch.zeros(1, lstm_hidden_dim, device=device)
    truck_cx = torch.zeros(1, lstm_hidden_dim, device=device)
    drone_hx = torch.zeros(1, lstm_hidden_dim, device=device)
    drone_cx = torch.zeros(1, lstm_hidden_dim, device=device)

    log_prob_sum = 0.0
    done = False

    while not done:
        # Get action probabilities from model.
        truck_probs, drone_probs, truck_hx, truck_cx, drone_hx, drone_cx, h_enc = model(
            coords, visited, truck_last, drone_last, truck_hx, truck_cx, drone_hx, drone_cx)
        # Sample actions (indices)
        truck_dist = torch.distributions.Categorical(truck_probs)
        drone_dist = torch.distributions.Categorical(drone_probs)
        truck_action = truck_dist.sample()  # (1,)
        drone_action = drone_dist.sample()  # (1,)

        log_prob_sum = log_prob_sum + truck_dist.log_prob(truck_action) + drone_dist.log_prob(drone_action)

        # Update state in environment (convert tensor actions to int)
        truck_action_idx = truck_action.item()
        drone_action_idx = drone_action.item()
        next_state, step_cost, done = env.step(truck_action_idx, drone_action_idx)
        # Update visited mask and last indices.
        visited_np = next_state["visited"].astype(np.float32)
        visited = torch.tensor(visited_np, dtype=torch.float32, device=device).unsqueeze(0)
        truck_last = torch.tensor([env.truck_pos], dtype=torch.long, device=device)
        drone_last = torch.tensor([env.drone_pos], dtype=torch.long, device=device)
    total_cost = env.get_total_cost()
    return log_prob_sum, total_cost, h_enc.mean(dim=1)  # return mean encoder embedding as graph rep.


def train(num_epochs=1000, num_nodes=10, alpha=2.0, lr=1e-4):
    model = HybridModel(embed_dim=128, num_layers=3, num_heads=8, lstm_hidden_dim=128).to(device)
    critic = Critic(embed_dim=128, hidden_dim=128).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(critic.parameters()), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # Create a new instance for each episode.
        env = TSPDEnv(num_nodes=num_nodes, alpha=alpha)
        log_prob_sum, cost, graph_embed = sample_episode(model, env, device)
        cost_tensor = torch.tensor([cost], dtype=torch.float32, device=device)

        # Critic: predict baseline value from graph embedding.
        baseline = critic(graph_embed)  # shape (1, 1)
        advantage = cost_tensor - baseline.squeeze()

        # Actor loss (we want to minimize cost, so use cost - baseline as advantage)
        actor_loss = advantage.detach() * (-log_prob_sum)
        # Critic loss: MSE between predicted baseline and actual cost.
        critic_loss = F.mse_loss(baseline.squeeze(), cost_tensor)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.3f}, Loss = {loss.item():.3f}, Advantage = {advantage.item():.3f}")


if __name__ == '__main__':
    # For reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(num_epochs=1000, num_nodes=10, alpha=2.0, lr=1e-4)