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

    # Store device from input weight
    device = weight.device

    if groups_per_pack == 1:
        # Each group becomes its own pack. No actual packing or sorting needed.
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Sort weights and get indices
    sorted_weights, sorted_indices = weight.float().sort(-1, descending=True)

    # Initialize pack_index and rank_in_pack on the correct device
    pack_index = torch.full((num_layers, num_groups), -1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full((num_layers, num_groups), -1, dtype=torch.int64, device=device)

    # Initialize pack_loads and pack_counts on the correct device
    pack_loads = torch.zeros(num_layers, num_packs, dtype=torch.float32, device=device)
    pack_counts = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)

    # Iterate through sorted items and assign them to the least loaded pack
    # Vectorize operations across num_layers for better performance.
    layer_indices = torch.arange(num_layers, device=device)
    for g in range(num_groups):
        # Get the original index of the g-th sorted item for all layers
        original_group_indices_g = sorted_indices[:, g]  # [num_layers]

        # Create a mask for available packs for each layer
        available_packs_mask = pack_counts < groups_per_pack  # [num_layers, num_packs]

        # Set loads of unavailable packs to infinity to exclude them from argmin
        masked_pack_loads = torch.where(available_packs_mask, pack_loads, torch.full_like(pack_loads, float('inf')))

        # Find the chosen pack index for each layer (index of min load)
        chosen_pack_indices = torch.argmin(masked_pack_loads, dim=-1)  # [num_layers]

        # Update pack_index and rank_in_pack using advanced indexing
        pack_index[layer_indices, original_group_indices_g] = chosen_pack_indices
        
        # Get current pack counts for the chosen packs before incrementing
        current_rank = pack_counts[layer_indices, chosen_pack_indices]
        rank_in_pack[layer_indices, original_group_indices_g] = current_rank

        # Get the weight of the current item for each layer
        weights_to_add = weight[layer_indices, original_group_indices_g]

        # Update pack_loads and pack_counts using scatter_add_
        pack_loads.scatter_add_(1, chosen_pack_indices.unsqueeze(1), weights_to_add.unsqueeze(1))
        pack_counts.scatter_add_(1, chosen_pack_indices.unsqueeze(1), torch.ones_like(weights_to_add, dtype=torch.int64).unsqueeze(1))

    return pack_index, rank_in_pack


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
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


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

    # Unified rebalancing logic:
    # The hierarchical approach is used when num_groups is divisible by num_nodes.
    # Otherwise, it falls back to a global balancing strategy by setting num_nodes=1 and num_groups=1.
    # We can simplify this by directly implementing the global strategy when needed,
    # or by ensuring the hierarchical logic correctly handles num_nodes=1.

    # Determine the effective number of groups and nodes for balancing.
    # If num_groups is not divisible by num_nodes, we treat it as a single group
    # across all nodes for the first level of balancing.
    effective_num_groups = num_groups if num_groups % num_nodes == 0 else 1
    effective_num_nodes = num_nodes if num_groups % num_nodes == 0 else 1

    # If num_groups % num_nodes != 0, we effectively want to balance across all groups as if they were one large group.
    # And then balance those groups onto nodes.
    # This is equivalent to setting num_nodes = 1 and num_groups = 1 in the hierarchical logic.
    # Let's adjust parameters for a unified call to a simplified hierarchical logic.

    # Simplified approach:
    # First, balance logical experts across groups within layers.
    # Then, balance these groups across nodes.
    # Finally, replicate experts within nodes to balance across GPUs.

    # Step 1: Balance logical experts into groups.
    # If num_groups == 1, this step is trivial and doesn't change anything.
    group_size = num_logical_experts // num_groups
    tokens_per_group_layer = weight.unflatten(-1, (num_groups, group_size)).sum(-1) # [num_layers, num_groups]

    # Pack groups onto nodes.
    # If num_groups % num_nodes != 0, we effectively want to balance all groups onto a single 'node' conceptually,
    # and then distribute those across the actual num_nodes.
    # Let's consider the case where num_groups is not divisible by num_nodes.
    # The original code uses num_nodes for packing groups. If num_groups < num_nodes, this packing might be problematic.
    # A more robust approach is to balance groups onto num_nodes, and if num_groups < num_nodes, some nodes get 0 groups.
    # The current `balanced_packing` assumes num_groups is divisible by num_packs (num_nodes here).

    # Let's redefine the packing strategy:
    # We want to pack 'num_groups' items (groups) into 'num_nodes' packs.
    # If num_groups is not divisible by num_nodes, balanced_packing needs to handle it.
    # The current `balanced_packing` asserts `num_groups % num_packs == 0`.
    # This means if num_groups % num_nodes != 0, the original code calls `rebalance_experts_hierarchical` with num_nodes=1, num_groups=1.
    # This is a global balancing strategy.

    # Let's ensure the function call correctly reflects the intended strategy.
    # If num_groups % num_nodes == 0, use hierarchical.
    # Otherwise, use global (which is hierarchical with num_nodes=1, num_groups=1).

    if num_groups % num_nodes == 0:
        # Hierarchical balancing: balance groups to nodes, then nodes to GPUs.
        # Step 1: Pack groups to nodes.
        group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group_layer, num_nodes)
        # Map logical experts to "middle" logical experts (logical experts per node)
        # log2mlog: [num_layers, num_logical_experts] -> maps logical expert to its middle logical expert
        mlog2log_map = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                        torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
        # mlog2log_map: [num_layers, num_logical_experts] maps each logical expert to its "middle" logical expert ID.
        # We need the inverse mapping for `weight.gather` later.
        # This inverse mapping is tricky because multiple logical experts can map to the same middle expert.
        # The `inverse` function is designed for permutations, which isn't directly applicable here.

        # Let's rethink the mapping and replication.
        # We have `num_layers` of `weight` ([num_layers, num_logical_experts]).
        # First, we group logical experts: `tokens_per_group_layer` ([num_layers, num_groups]).
        # Then, we pack these groups onto `num_nodes`: `group_pack_index` ([num_layers, num_groups]) and `group_rank_in_pack` ([num_layers, num_groups]).
        # This means `num_groups` are assigned to `num_nodes`. `groups_per_node` groups per node.
        # So, we have `num_layers * num_nodes` "node-level" expert weight tensors.
        # Each node-level tensor has `num_logical_experts // num_nodes` "middle" experts.

        # Let's create the "middle" logical experts' weights.
        # `mlog_weights`: [num_layers * num_nodes, num_logical_experts // num_nodes]
        # This requires carefully gathering weights.
        # The `mlog2log` mapping needs to be constructed correctly.

        # `mlog2log_map` assigns each logical expert to a conceptual "middle" logical expert.
        # e.g., if num_logical_experts=8, num_groups=2, group_size=4.
        # Group 0 contains logical experts 0,1,2,3. Group 1 contains 4,5,6,7.
        # If num_nodes=2, groups_per_node=1.
        # Group 0 (log experts 0-3) goes to node 0. Group 1 (log experts 4-7) goes to node 1.
        # Node 0 has 1 middle expert. Node 1 has 1 middle expert.
        # Middle expert for node 0 is logical experts 0-3. Middle expert for node 1 is 4-7.
        # The mapping `mlog2log_map` should map:
        # 0,1,2,3 -> 0 (on node 0)
        # 4,5,6,7 -> 0 (on node 1)
        # The current `mlog2log` calculation seems to be constructing the indices of the middle experts.
        # `log2mlog` is `[num_layers, num_logical_experts]`.
        # It maps logical expert ID to its index within the middle expert group.
        # The `mlog2log` is the inverse of this index mapping.

        # Let's reconstruct `mlog2log` and `tokens_per_mlog` more directly.
        # For each layer, we have `num_groups` of `group_size`.
        # `group_pack_index`: [num_layers, num_groups] - node assignment for each group.
        # `group_rank_in_pack`: [num_layers, num_groups] - index of group within the node's assigned groups.

        # We want to create `num_layers * num_nodes` "node-level" expert groups.
        # Each node-level group corresponds to a set of original logical experts.
        # The `tokens_per_mlog` should be `[num_layers * num_nodes, num_logical_experts // num_nodes]`.

        # Calculate the logical expert index for each "middle" expert.
        # `mlog_expert_indices`: [num_layers, num_nodes, groups_per_node, group_size]
        # This is becoming complex. Let's simplify the conceptualization.

        # Simpler approach:
        # 1. Pack groups to nodes. This determines which logical experts are grouped together on each node.
        # `group_pack_index`: [num_layers, num_groups] -> node ID for each group.
        # `group_rank_in_pack`: [num_layers, num_groups] -> rank within node for each group.

        # 2. For each node, create a combined weight tensor for the logical experts assigned to it.
        #    The number of logical experts per node will be `groups_per_node * group_size`.
        #    We need to gather the weights of logical experts based on `group_pack_index`.

        # Let's create a mapping from (layer, node) to a contiguous block of logical experts.
        # `node_logical_expert_map`: [num_layers, num_nodes, groups_per_node * group_size]

        # This requires constructing a permutation for each layer and node.
        # The original `mlog2log` calculation is:
        # `((group_pack_index * groups_per_node + group_rank_in_pack) * group_size)`
        # This forms a base index for each logical expert.
        # `log2mlog` is this base index + `arange(group_size)`.
        # `mlog2log` is the inverse of `log2mlog`.

        # Let's directly construct the `tokens_per_mlog` and `mlog2log`.
        # `tokens_per_mlog` is the weight tensor for experts *within* each node, considering the grouped logical experts.
        # The number of such "middle" experts per node is `groups_per_node * group_size`.

        # For each layer:
        # `group_pack_index` tells us which node each group belongs to.
        # `group_rank_in_pack` tells us the order within the node.

        # Let's create `num_layers * num_nodes` tensors, each of size `groups_per_node * group_size`.
        # `mlog_weights`: [num_layers * num_nodes, groups_per_node * group_size]
        # `mlog_to_orig_log_map`: [num_layers * num_nodes, groups_per_node * group_size]

        # Example: num_layers=1, num_logical_experts=8, num_groups=2, group_size=4, num_nodes=2, groups_per_node=1, num_gpus=2, phy_experts_per_gpu=1.
        # weight: [1, 8]
        # tokens_per_group_layer: [1, 2] (sum of weights for group 0 and group 1)
        # balanced_packing(tokens_per_group_layer, num_nodes=2) ->
        #   group_pack_index: [1, 2] -> [[0, 1]] (group 0 to node 0, group 1 to node 1)
        #   group_rank_in_pack: [1, 2] -> [[0, 0]] (rank 0 within node for both groups)
        # groups_per_node = 1, group_size = 4.

        # Middle expert for node 0 is logical experts 0,1,2,3.
        # Middle expert for node 1 is logical experts 4,5,6,7.
        # `mlog_to_orig_log_map` for node 0 should be [0,1,2,3].
        # `mlog_to_orig_log_map` for node 1 should be [4,5,6,7].

        # The original `mlog2log` calculation:
        # `base_idx = (group_pack_index * groups_per_node + group_rank_in_pack) * group_size`
        # `base_idx`: [1, 2] -> [[0*1+0, 1*1+0]] * 4 = [[0, 4]]
        # `log2mlog`: `base_idx.unsqueeze(-1) + torch.arange(group_size)`
        # `log2mlog`: [[0, 4]].unsqueeze(-1) + [0,1,2,3] -> [[0,1,2,3], [4,5,6,7]]
        # This `log2mlog` is `[num_layers, num_logical_experts]`. It maps original logical expert ID to its "middle" expert index.
        # The original logic uses `weight.gather(-1, mlog2log)` which is incorrect here.
        # `weight.gather(-1, mlog2log)` expects `mlog2log` to be `[num_layers, num_middle_experts]` where `num_middle_experts` is the number of columns in `weight`.
        # The shape should be `[num_layers, num_logical_experts]` if we are gathering from `weight` which is `[num_layers, num_logical_experts]`.
        # The `mlog2log` calculated here has shape `[num_layers, num_logical_experts]`.
        # It maps `original_logical_expert_id` to `middle_expert_id`.
        # We need to gather `weight[layer, original_logical_expert_id]` and group them by `middle_expert_id`.

        # Let's create `mlog_weights` directly.
        # `mlog_weights`: [num_layers * num_nodes, groups_per_node * group_size]
        # `mlog_to_orig_log_map`: [num_layers * num_nodes, groups_per_node * group_size]

        # Vectorized construction of mlog_weights and mlog_to_orig_log_map
        # This replaces the nested Python loops for better performance.

        # Calculate the new "middle expert group" index for each original group.
        # This new index combines node assignment and rank within the node.
        # Shape: [num_layers, num_groups]
        new_group_idx = group_pack_index * groups_per_node + group_rank_in_pack

        # Create a tensor that maps new group index back to original group index.
        # This is a permutation tensor for groups within each layer.
        # Shape: [num_layers, num_nodes * groups_per_node] (which is num_groups)
        group_permutation = torch.empty((num_layers, num_groups),
                                        dtype=torch.int64,
                                        device=weight.device)
        group_permutation.scatter_(1, new_group_idx,
                                   torch.arange(num_groups, device=weight.device).expand(num_layers, -1))

        # Reshape original weights to [num_layers, num_groups, group_size]
        weight_reshaped = weight.view(num_layers, num_groups, group_size)

        # Gather weights based on the group_permutation
        # The .gather() operation reorders the groups according to the permutation.
        # Shape: [num_layers, num_groups, group_size]
        mlog_weights_ordered = weight_reshaped.gather(1, group_permutation.unsqueeze(-1).expand(-1, -1, group_size))
        
        # Reshape to [num_layers, num_nodes, groups_per_node * group_size]
        # This is the final mlog_weights.
        mlog_weights = mlog_weights_ordered.view(num_layers, num_nodes, groups_per_node * group_size)

        # Construct mlog_to_orig_log_map similarly
        # Original logical expert IDs, reshaped to [num_groups, group_size]
        orig_log_ids_reshaped = torch.arange(num_logical_experts,
                                             dtype=torch.int64,
                                             device=weight.device).view(num_groups, group_size)
        
        # Expand to [num_layers, num_groups, group_size] and gather
        mlog_to_orig_log_map_ordered = orig_log_ids_reshaped.unsqueeze(0).expand(num_layers, -1, -1).gather(
            1, group_permutation.unsqueeze(-1).expand(-1, -1, group_size))
        
        # Reshape to [num_layers, num_nodes, groups_per_node * group_size]
        mlog_to_orig_log_map = mlog_to_orig_log_map_ordered.view(
            num_layers, num_nodes, groups_per_node * group_size)

        # Now, `mlog_weights`: [num_layers, num_nodes, num_logical_experts // num_nodes]
        # And `mlog_to_orig_log_map`: [num_layers, num_nodes, num_logical_experts // num_nodes]

        # Step 2: Replicate experts within each node.
        # `mlog_weights` are the weights for experts within each node.
        # `num_physical_experts // num_nodes` is the target number of physical experts per node.
        # `replicate_experts` takes `[X, num_log]` and returns `phy2log`, `rank`, `logcnt`.
        # Here, X = num_layers * num_nodes, num_log = num_logical_experts // num_nodes.
        # So, we need to reshape `mlog_weights` and `mlog_to_orig_log_map`.

        num_middle_experts = num_logical_experts // num_nodes
        reshaped_mlog_weights = mlog_weights.view(-1, num_middle_experts)
        reshaped_mlog_to_orig_log_map = mlog_to_orig_log_map.view(-1, num_middle_experts)

        # `phy2mlog`: [num_layers * num_nodes, num_physical_experts // num_nodes]
        # `phyrank`: [num_layers * num_nodes, num_physical_experts // num_nodes]
        # `mlogcnt`: [num_layers * num_nodes, num_middle_experts]
        phy2mlog, phyrank, mlogcnt = replicate_experts(
            reshaped_mlog_weights, num_physical_experts // num_nodes)

        # Step 3: Pack physical experts to GPUs.
        # `phy2mlog` maps physical experts (within each node) to middle logical experts.
        # `phy2mlog` is `[num_layers * num_nodes, num_phy_per_node]`
        # `phyrank` is `[num_layers * num_nodes, num_phy_per_node]`
        # `mlogcnt` is `[num_layers * num_nodes, num_middle_experts]`

        # We need to map these back to global physical expert indices.
        # The `phy2mlog` refers to the *middle* logical experts.
        # The `replicate_experts` function returns `phy2log` which is the *original* logical expert ID.
        # So, `phy2mlog` from `replicate_experts` is actually `[num_layers*num_nodes, num_phy_per_node]`,
        # where each entry is an index into the *middle* logical experts.

        # We need to map these `phy2mlog` indices back to original logical expert IDs.
        # `reshaped_mlog_to_orig_log_map`: [num_layers * num_nodes, num_middle_experts]
        # `final_phy2log_per_node`: [num_layers * num_nodes, num_phy_per_node]
        final_phy2log_per_node = reshaped_mlog_to_orig_log_map.gather(-1, phy2mlog)

        # `phyrank` is the replica rank for the middle experts.
        # We need to combine this with `mlogcnt` to get the final `logcnt`.
        # `mlogcnt` is `[num_layers * num_nodes, num_middle_experts]`. It tells us how many replicas each middle expert has.
        # The `phyrank` is the rank of the replica for a *specific* middle expert.

        # The `logcnt` returned by `replicate_experts` is `[X, num_log]`, which is `[num_layers*num_nodes, num_middle_experts]`.
        # We need to map this back to `[num_layers, num_logical_experts]`.

        # Let's use the `mlogcnt` directly to get the total count of replicas for each logical expert.
        # `mlogcnt` is `[num_layers * num_nodes, num_middle_experts]`.
        # We need to sum/average counts based on the `mlog_to_orig_log_map`.

        # Let's refine the output mapping:
        # `physical_to_logical_map`: [num_layers, num_replicas]
        # `logical_to_physical_map`: [num_layers, num_logical_experts, X]
        # `expert_count`: [num_layers, num_logical_experts]

        # `final_phy2log_per_node`: [num_layers * num_nodes, num_phy_per_node]
        # This is the logical expert ID for each physical expert *within its node*.
        # We need to convert this to global physical expert IDs.
        # Physical experts are grouped by GPU. `num_gpus` total, `num_gpus // num_nodes` per node.
        # `phy_experts_per_gpu` = `num_physical_experts // num_gpus`.
        # So, each GPU has `phy_experts_per_gpu` physical experts.
        # Each Node has `num_gpus // num_nodes` GPUs.
        # Total physical experts per node = `(num_gpus // num_nodes) * phy_experts_per_gpu`
        # This should equal `num_physical_experts // num_nodes`.

        # `final_phy2log_per_node`: [num_layers, num_nodes, num_phy_per_node]
        # `final_rank_per_node`: [num_layers, num_nodes, num_phy_per_node] (this is `phyrank`)

        # We need to flatten this to `[num_layers, num_physical_experts]`.
        # `num_physical_experts = num_nodes * num_phy_per_node`.

        # `final_phy2log_global`: [num_layers, num_physical_experts]
        # `final_rank_global`: [num_layers, num_physical_experts]

        # The `mlog_to_orig_log_map` correctly maps middle experts to original logical experts.
        # `phy2mlog` maps physical experts (within a node) to middle experts.
        # So, `phy2mlog.gather(-1, final_phy2log_per_node)` gives the logical expert ID.
        # This is what `final_phy2log_per_node` already represents.

        # Let's rename `final_phy2log_per_node` to `phy2log_per_node` for clarity.
        phy2log_per_node = final_phy2log_per_node
        rank_per_node = phyrank # [num_layers * num_nodes, num_phy_per_node]

        # Flatten these to get global mappings for physical experts.
        phy2log_global = phy2log_per_node.view(num_layers, -1)
        rank_global = rank_per_node.view(num_layers, -1)

        # Now, let's construct the `logcnt` (expert_count).
        # `mlogcnt`: [num_layers * num_nodes, num_middle_experts]
        # `mlog_to_orig_log_map`: [num_layers * num_nodes, num_middle_experts]

        # We need to aggregate `mlogcnt` based on `mlog_to_orig_log_map`.
        # For each logical expert, its total count is the sum of counts of its constituent middle experts.
        # This aggregation needs to be done carefully.

        # Let's create a tensor for `logcnt`: [num_layers, num_logical_experts] initialized to zeros.
        expert_count = torch.zeros(num_layers, num_logical_experts, dtype=torch.int64, device=weight.device)

        # The expert_count aggregation:
        # mlog_to_orig_log_map: [num_layers, num_nodes, num_middle_experts]
        # mlogcnt: [num_layers * num_nodes, num_middle_experts]
        # Flatten mlog_to_orig_log_map and mlogcnt to [num_layers, N] for scatter_add_
        mlog_to_orig_log_map_flat = mlog_to_orig_log_map.view(num_layers, -1)
        mlogcnt_reshaped = mlogcnt.view(num_layers, -1)

        expert_count = torch.zeros(num_layers, num_logical_experts, dtype=torch.int64, device=weight.device)
        # Use scatter_add_ to efficiently aggregate counts for each logical expert across layers
        expert_count.scatter_add_(1, mlog_to_orig_log_map_flat, mlogcnt_reshaped)

        # The logical_to_physical_map construction is handled by the outer rebalance_experts function.
        # So, this function only needs to return the global mappings and counts.
        return phy2log_global, rank_global, expert_count


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
    device = weight.device  # Store device from input weight
    weight = weight.float() # Ensure float type, but keep on original device

    if num_groups % num_nodes == 0:
        # Use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Use global load-balance policy by effectively setting num_groups=1 and num_nodes=1.
        # The rebalance_experts_hierarchical function's 'if' branch will handle this case.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    # All necessary outputs (phy2log, phyrank, logcnt) are now available.
    # Construct logical_to_physical_map based on these outputs.
    num_physical_experts_total = num_replicas

    # Calculate max_replicas_per_expert: In the worst case, one expert could receive all redundant replicas.
    # This assumes num_physical_experts_total >= num_logical_experts, which is typically true for replication.
    max_replicas_per_expert = num_physical_experts_total - num_logical_experts + 1

    # Initialize the map with -1 (indicating no physical expert assigned)
    logical_to_physical_map = torch.full((num_layers, num_logical_experts, max_replicas_per_expert),
                                         -1, dtype=torch.int64, device=device)

    # Create physical expert IDs for scattering
    physical_expert_ids = torch.arange(num_physical_experts_total, dtype=torch.int64, device=device)

    # Expand physical_expert_ids to match the shape of phy2log and phyrank for broadcasting
    # Shape: [num_layers, num_physical_experts_total]
    physical_expert_ids_expanded = physical_expert_ids.unsqueeze(0).expand(num_layers, num_physical_experts_total)

    # Create layer indices for advanced indexing
    layer_indices = torch.arange(num_layers, dtype=torch.int64, device=device).unsqueeze(1)
    
    # Assign physical expert IDs to the correct positions in the map using advanced indexing.
    # This is a vectorized operation, highly efficient for populating the map.
    logical_to_physical_map[layer_indices, phy2log, phyrank] = physical_expert_ids_expanded

    # Return the computed mappings and counts.
    return phy2log, logical_to_physical_map, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

