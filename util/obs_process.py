import torch


def build_batch(obs):
    """
    Build a batch of observations for the UAV agents.
    :param obs: expect a dict: index -> obs. if env of a index is terminated, index should not be in the keys
    :return: observation batch, next step is to build state tensor
    """
    keys = ["nodes", "parcel", "truck", "coordinate", "power", "capacity", "travel_distance", "choice_mask"]
    batch = {}
    for key in keys:
        # Convert each field to a tensor and stack over UAV agents.
        batch[key] = torch.stack([torch.tensor(x[key]) for x in obs], dim=0).float()
    return batch


def buildStateTensor(batch, device="cpu"):
    # Process customer nodes.
    customers = batch["nodes"]  # (batch, num_customer, 2)
    if batch["parcel"].dim() == 2:
        parcel = batch["parcel"].unsqueeze(-1)  # (batch, num_customer, 1)
    else:
        parcel = batch["parcel"]
    if batch["choice_mask"].dim() == 2:
        choice_mask = batch["choice_mask"].unsqueeze(-1)
    else:
        choice_mask = batch["choice_mask"]
    customers = torch.cat([customers, parcel], dim=-1)  # (batch, num_customer, 3)

    # Process truck nodes.
    trucks = batch["truck"]  # (batch, truck_num, 2)
    dummy_weight = torch.zeros(trucks.size(0), 1, device=trucks.device)
    trucks = torch.cat([trucks, dummy_weight], dim=-1).unsqueeze(1)  # (batch, truck_num, 3)

    # Concatenate candidates: customers first, then trucks.
    candidates = torch.cat([customers, trucks], dim=1).to(device)  # (batch, num_candidates, 3)
    # candidates = torch.cat([candidates, choice_mask], dim=-1)  # (batch, num_candidates, 4)

    return candidates
