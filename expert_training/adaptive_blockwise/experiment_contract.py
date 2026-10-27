"""Shared validation for scalar legacy K and heterogeneous block groups."""


def block_groups(value, block_id):
    if isinstance(value, dict):
        value = value[str(block_id)] if str(block_id) in value else value[block_id]
    result = int(value)
    if result < 1:
        raise ValueError("Each block must have at least one expert group")
    return result


def validate_partition_groups(config, partition):
    blocks = partition["blocks"]
    ids = [int(block["block_id"]) for block in blocks]
    if ids != list(range(len(blocks))):
        raise ValueError("Block IDs must be consecutive and ordered")
    groups = config["num_groups"]
    if isinstance(groups, dict) and {int(key) for key in groups} != set(ids):
        raise ValueError(
            f"Partition has {len(blocks)} blocks, incompatible with K={groups}. "
            "Confirm partition constraints before proceeding; never truncate the K vector."
        )
    for block_id in ids:
        block_groups(groups, block_id)
    layers = [int(layer) for block in blocks for layer in block["layers"]]
    if layers != list(range(int(config["num_layers"]))):
        raise ValueError("Blocks must cover all decoder layers once, in order")


def local_accumulation(global_batch, local_batch, world_size):
    denominator = int(local_batch) * int(world_size)
    if denominator <= 0 or int(global_batch) <= 0 or int(global_batch) % denominator:
        raise ValueError("Global effective batch must be divisible by local batch times world size")
    return int(global_batch) // denominator
