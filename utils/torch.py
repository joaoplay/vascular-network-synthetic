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
    return unique, inverse.new_empty(uniq.size(dim)).scatter_(dim, inverse, perm)


def compute_class_weights(data_y, total_classes):
    """
    Get the weight of each class excluding padding (class 201)
    :param total_classes:
    :param data_y:
    :return:
    """
    # Create weight vector based on the frequency of each class
    class_weights = torch.ones(total_classes)
    for i in range(total_classes):
        n_samples = torch.sum(data_y == i).item()
        if n_samples > 0:
            class_weights[i] = 1.0 / n_samples

    return class_weights
