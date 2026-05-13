import torch


def unique(x, dim=-1):
    """
    This auxiliary function is used to get the unique elements of a tensor and the corresponding inverse indices.
    Natively, torch.unique() only returns the unique elements and the inverse indices. It is useful to perform
    other indexing operations based on the unique elements.
    :param x:
    :param dim:
    :return:
    """
    uniq, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return uniq, inverse.new_empty(uniq.size(dim)).scatter_(dim, inverse, perm)

def compute_class_weights(data_y, total_classes, power: float = 1.0, max_ratio: float | None = None,
                          min_weight: float | None = None):
    """
    Get inverse-frequency class weights, normalised so the mean weight is 1.0.

    ``power < 1`` tempers rare-class upweighting.
    ``min_weight`` sets a floor so no class gets a near-zero weight after normalisation.
    :param total_classes:
    :param data_y:
    :param power: exponent applied to inverse frequency (< 1 compresses the spread)
    :param max_ratio: cap the ratio between the largest and smallest weight
    :param min_weight: floor applied to all weights after normalisation
    :return:
    """
    flat = data_y.reshape(-1).long()
    flat = flat[(flat >= 0) & (flat < total_classes)]
    counts = torch.bincount(flat, minlength=total_classes).float()
    class_weights = torch.zeros(total_classes)

    nonzero = counts > 0
    if torch.any(nonzero):
        class_weights[nonzero] = counts[nonzero].reciprocal().pow(power)
        class_weights[nonzero] /= class_weights[nonzero].mean()

        if max_ratio is not None:
            w_min = class_weights[nonzero].min()
            class_weights[nonzero] = torch.clamp(class_weights[nonzero], max=w_min * max_ratio)
            class_weights[nonzero] /= class_weights[nonzero].mean()

        if min_weight is not None:
            class_weights[nonzero] = torch.clamp(class_weights[nonzero], min=min_weight)
            class_weights[nonzero] /= class_weights[nonzero].mean()

    return class_weights