# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        # If each group is its own pack, no complex packing needed.
        # Create results directly on the original device.
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    original_device = weight.device
    # Sort indices on CPU for the Python loop, as Python lists are used for pack_weights and pack_items.
    # The weight tensor itself can remain on its original device.
    indices = weight.float().sort(-1, descending=True).indices.cpu()

    # Create tensors for results on CPU, then move them to the original_device at the end.
    pack_index_cpu = torch.full_like(weight,
                                     fill_value=-1,
                                     dtype=torch.int64,
                                     device="cpu")
    rank_in_pack_cpu = torch.full_like(pack_index_cpu, fill_value=-1)

    for i in range(num_layers):
        # Use float for pack_weights to avoid potential issues with large integer sums
        # and for consistency with the input `weight` being float.
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            # Find the pack with the minimum current weight that still has capacity.
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index_cpu[i, group] = pack
            rank_in_pack_cpu[i, group] = pack_items[pack]
            # Convert scalar tensor to Python float for addition to Python list.
            pack_weights[pack] += weight[i, group].item()
            pack_items[pack] += 1
    return pack_index_cpu.to(original_device), rank_in_pack_cpu.to(original_device)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device

    # Initialize `phy2log` and `rank` for the initial `num_log` experts.
    # These are the non-replicated experts, mapping physical 0..num_log-1 to logical 0..num_log-1
    # with replica rank 0.
    initial_phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(n, 1)
    initial_rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Pre-allocate `phy2log` and `rank` to the final `num_phy` size.
    final_phy2log = torch.empty(n, num_phy, dtype=torch.int64, device=device)
    final_rank = torch.empty(n, num_phy, dtype=torch.int64, device=device)

    # Copy initial expert mappings
    final_phy2log[:, :num_log] = initial_phy2log
    final_rank[:, :num_log] = initial_rank

    # Replicate experts by adding `num_redundant` physical experts.
    # These new experts are appended starting from index `num_log`.
    for i in range(num_log, num_phy):
        # Find the logical expert that currently has the highest average load per replica.
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        # Assign this logical expert to the current physical expert slot `i`.
        final_phy2log[:, i] = redundant_indices
        # The rank of this new replica is its current count for that logical expert.
        final_rank[:, i] = logcnt[arangen, redundant_indices]
        # Increment the count of replicas for the chosen logical expert.
        logcnt[arangen, redundant_indices] += 1
    return final_phy2log, final_rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    # Ensure weight is float, but keep it on its original device.
    # The .cpu() call is removed to allow GPU operations where possible,
    # reducing device transfer overhead if `weight` originates from GPU.
    weight = weight.float()
    original_device = weight.device

    if num_groups % num_nodes == 0:
        # Use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Use global load-balance policy when groups are not perfectly divisible by nodes.
        # This simplifies the problem to a single "node" and "group" for global balancing.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    # maxlogcnt determines the maximum number of replicas any single logical expert can have.
    # This is used to size the `log2phy` tensor, which maps logical experts to their physical replicas.
    maxlogcnt = num_redundant_experts + 1

    # Initialize `log2phy` tensor with -1, indicating empty slots.
    # This tensor maps [layer, logical_expert_id, replica_rank] to physical_expert_id.
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=original_device,
    )

    # Populate `log2phy` using scatter_ based on `phy2log` and `phyrank`.
    # `phy2log` gives the logical ID for each physical expert.
    # `phyrank` gives the replica rank for that logical ID.
    # We flatten the last two dimensions of `log2phy` for scatter_ operation.
    # The index for scatter_ is calculated as `logical_id * maxlogcnt + replica_rank`.
    # The values to scatter are the physical expert IDs (0 to num_replicas-1).
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=original_device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

