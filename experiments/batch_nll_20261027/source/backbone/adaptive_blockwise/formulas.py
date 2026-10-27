import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def metric_kernel(metric, metric_inverse):
    """Return M^-1 M M^-T without assuming a numerically exact inverse."""
    return metric_inverse @ metric @ metric_inverse.transpose(-1, -2)


def directional_score(source_gradient, query_gradient, query_metric, query_metric_inverse):
    """Equation 5: S(source -> query), summed outside this function over modules."""
    kernel = metric_kernel(query_metric, query_metric_inverse)
    weighted_query = query_gradient @ kernel.transpose(-1, -2)
    source_gram = source_gradient.transpose(-1, -2) @ source_gradient
    # Preserve any leading query dimensions (for example, the sample axis in
    # per-sample assignment) and contract only the LoRA output/rank axes.
    return 2.0 * torch.sum(source_gradient * weighted_query, dim=(-2, -1)) - torch.sum(
        source_gram * kernel, dim=(-2, -1)
    )


def directional_score_matrix(source_gradients, query_gradients, query_metrics, query_metric_inverses):
    """Vectorized Equation 5 for [source, query] pairs of one LoRA module."""
    kernels = metric_kernel(query_metrics, query_metric_inverses)
    weighted_query = torch.einsum("qor,qrs->qos", query_gradients, kernels)
    cross = torch.einsum("sor,qor->sq", source_gradients, weighted_query)
    source_grams = torch.einsum("sor,sot->srt", source_gradients, source_gradients)
    energy = torch.einsum("srt,qrt->sq", source_grams, kernels)
    return 2.0 * cross - energy


def symmetric_compatibility(directional):
    return 0.5 * (directional + directional.transpose(-1, -2))


def normalize_compatibility(matrix, epsilon=EPS):
    """Equation 6. The undefined diagonal is set to zero after normalization."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square compatibility matrix, got {matrix.shape}")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    if values.size == 0:
        raise ValueError("At least two calibration items are required")
    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64))
    normalized = ((matrix - mean) / (std + epsilon)).astype(np.float32)
    np.fill_diagonal(normalized, 0.0)
    return normalized, mean, std


def upper_fingerprint(matrix):
    matrix = np.asarray(matrix)
    return matrix[np.triu_indices(matrix.shape[0], k=1)].astype(np.float64, copy=False)


def correlation_distance(left, right, epsilon=EPS):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= epsilon:
        return 0.0 if np.allclose(left, right, atol=epsilon) else 1.0
    corr = float(np.dot(left, right) / denom)
    return float(np.clip(1.0 - corr, 0.0, 2.0))


def bootstrap_layer_distances(normalized_matrices, repeats=3, seed=20260718):
    """Equation 7 with shared bootstrap indices across all layers."""
    matrices = np.asarray(normalized_matrices, dtype=np.float32)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("normalized_matrices must have shape [layers, n, n]")
    layers, count, _ = matrices.shape
    rng = np.random.default_rng(seed)
    runs = np.zeros((repeats, layers, layers), dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, count, size=count)
        fingerprints = [upper_fingerprint(matrix[np.ix_(indices, indices)]) for matrix in matrices]
        for left in range(layers):
            for right in range(left + 1, layers):
                value = correlation_distance(fingerprints[left], fingerprints[right])
                runs[repeat, left, right] = value
                runs[repeat, right, left] = value
    return runs.mean(axis=0), runs.var(axis=0), runs


def block_heterogeneity(distance_matrix, start, end):
    """Equation 9 for inclusive layer interval [start, end]."""
    if end < start:
        raise ValueError("end must be >= start")
    length = end - start + 1
    if length == 1:
        return 0.0
    block = np.asarray(distance_matrix)[start : end + 1, start : end + 1]
    return float(block[np.triu_indices(length, k=1)].mean())


def solve_contiguous_partition(candidate_costs, num_layers, min_length=3, max_length=10, min_blocks=3, max_blocks=8, router_penalty=0.0):
    """Exact dynamic program for Equation 11 under all stated constraints."""
    infinity = float("inf")
    dp = np.full((max_blocks + 1, num_layers + 1), infinity, dtype=np.float64)
    previous = np.full((max_blocks + 1, num_layers + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for groups in range(1, max_blocks + 1):
        for end_exclusive in range(1, num_layers + 1):
            for length in range(min_length, max_length + 1):
                start = end_exclusive - length
                if start < 0 or not math.isfinite(dp[groups - 1, start]):
                    continue
                key = (start, end_exclusive - 1)
                if key not in candidate_costs:
                    continue
                value = dp[groups - 1, start] + float(candidate_costs[key]) + float(router_penalty)
                if value < dp[groups, end_exclusive] - 1e-12:
                    dp[groups, end_exclusive] = value
                    previous[groups, end_exclusive] = start
    choices = [(float(dp[g, num_layers]), g) for g in range(min_blocks, max_blocks + 1)]
    objective, groups = min(choices, key=lambda item: (item[0], item[1]))
    if not math.isfinite(objective):
        raise ValueError("No feasible contiguous partition for the requested constraints")
    blocks = []
    end_exclusive = num_layers
    for group in range(groups, 0, -1):
        start = int(previous[group, end_exclusive])
        if start < 0:
            raise RuntimeError("Dynamic-program backtracking failed")
        blocks.append((start, end_exclusive - 1))
        end_exclusive = start
    blocks.reverse()
    return blocks, objective


def symmetric_group_mean_scores(
    sample_gradients,
    sample_metrics,
    sample_metric_inverses,
    group_mean_gradients,
    group_mean_gradient_grams,
    group_mean_weighted_gradients,
    group_mean_kernels,
):
    """Exact mean symmetric compatibility from samples to each batch group.

    Group statistics are sufficient to average both directional terms over
    every batch in a group without retaining the original batch signatures.
    The returned tensor has shape ``[samples, groups]`` for one LoRA module.
    """
    sample_kernels = metric_kernel(sample_metrics, sample_metric_inverses)
    sample_weighted_gradients = torch.einsum(
        "bor,brs->bos", sample_gradients, sample_kernels
    )
    group_to_sample = 2.0 * torch.einsum(
        "kor,bor->bk", group_mean_gradients, sample_weighted_gradients
    ) - torch.einsum("krt,brt->bk", group_mean_gradient_grams, sample_kernels)

    sample_gradient_grams = torch.einsum(
        "bor,bot->brt", sample_gradients, sample_gradients
    )
    sample_to_group = 2.0 * torch.einsum(
        "bor,kor->bk", sample_gradients, group_mean_weighted_gradients
    ) - torch.einsum("brt,krt->bk", sample_gradient_grams, group_mean_kernels)
    return 0.5 * (group_to_sample + sample_to_group)


def soft_assignment(scores, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = F.softmax(torch.as_tensor(scores, dtype=torch.float32) / float(temperature), dim=-1)
    labels = probabilities.argmax(dim=-1)
    if probabilities.shape[-1] == 1:
        # With one group there is no runner-up; the assignment is certain.
        confidence = torch.ones_like(labels, dtype=probabilities.dtype)
    else:
        top2 = torch.topk(probabilities, k=2, dim=-1).values
        confidence = top2[..., 0] - top2[..., 1]
    return probabilities, labels, confidence


def class_weights(labels, num_groups):
    labels = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_groups).float()
    weights = torch.zeros(int(num_groups), dtype=torch.float32)
    present = counts > 0
    weights[present] = labels.numel() / (float(num_groups) * counts[present])
    return weights


def confidence_class_balanced_loss(logits, labels, confidence, weights, normalizer=None):
    """Equation 20 weighted numerator with an optional fixed global Z.

    When ``normalizer`` is omitted this retains the historical batch-local
    behavior. Router training passes a dataset-level denominator instead.
    """
    labels = labels.to(logits.device, dtype=torch.long)
    confidence = confidence.to(logits.device, dtype=logits.dtype)
    weights = weights.to(logits.device, dtype=logits.dtype)
    sample_weights = confidence * weights[labels]
    losses = F.cross_entropy(logits, labels, reduction="none")
    if normalizer is None:
        denominator = sample_weights.sum()
    else:
        denominator = torch.as_tensor(normalizer, device=logits.device, dtype=logits.dtype)
    denominator = denominator.clamp_min(torch.finfo(logits.dtype).eps)
    return torch.sum(sample_weights * losses) / denominator
