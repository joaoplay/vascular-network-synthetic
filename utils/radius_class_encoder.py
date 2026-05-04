import torch
import numpy as np
from sgg.radius_classes import R_EDGES


class RadiusClassEncoder:
    """
    Encodes continuous radius values into discrete classes using R_EDGES = [1, 2, 3, 4, 5, 6].
    - Class 0: r < 1
    - Class 1: 1 <= r < 2
    - Class 2: 2 <= r < 3
    - Class 3: 3 <= r < 4
    - Class 4: 4 <= r < 5
    - Class 5: 5 <= r < 6
    - Class 6: r >= 6
    """
    
    def __init__(self, top_class_max: float = 8.8):
        self.r_edges = np.array(R_EDGES, dtype=np.float32)
        self.n_classes = len(self.r_edges) + 1
        self.top_class_max = float(top_class_max)

    def transform(self, radius_data: torch.Tensor):
        """
        Convert continuous radius values to discrete class indices.
        :param radius_data: tensor of radius values
        :return: tensor of class indices
        """

        is_tensor = isinstance(radius_data, torch.Tensor)
        if not is_tensor:
            radius_data = torch.tensor(radius_data)

        device = radius_data.device
        #ensure a floating dtype for comparisons
        if not radius_data.dtype.is_floating_point:
            radius_data = radius_data.float()

        edges = torch.as_tensor(self.r_edges, dtype=radius_data.dtype, device=device)
        class_indices = torch.searchsorted(edges, radius_data.contiguous())
        class_indices = class_indices.clamp(0, self.n_classes - 1).long()

        return class_indices
    
    def inverse_transform(self, class_indices):
        labels = ['tiny', 'small', 'medium', 'normal', 'large', 'big', 'huge']

        if isinstance(class_indices, torch.Tensor):
            class_indices = class_indices.detach().cpu().numpy()

        arr = np.clip(np.asarray(class_indices).astype(int), 0, self.n_classes - 1)
        result = np.vectorize(labels.__getitem__)(arr)

        #if its a scalar, return a string, otherwise, return a list
        return result.item() if result.shape == () else result.tolist()

    def class_to_value(self, class_indices):
        """
        Convert class indices to a radius value
        Returns a torch tensor if input was a torch tensor, otherwise a numpy array.
        """
        is_tensor = isinstance(class_indices, torch.Tensor)
        if is_tensor:
            device = class_indices.device
            idx = class_indices.detach().cpu().numpy().astype(int)
        else:
            idx = np.asarray(class_indices).astype(int)

        idx = np.clip(idx, 0, self.n_classes - 1)
        edges = self.r_edges
        top = self.top_class_max

        # Build bin bounds: [0, edges[0]), [edges[0], edges[1]), ..., [edges[-1], top]
        lo = np.concatenate([[0.0], edges])
        hi = np.concatenate([edges, [top]])

        flat_idx = np.ravel(idx)
        flat = np.array([
            np.random.uniform(lo[k], hi[k]) for k in flat_idx
        ], dtype=np.float32).reshape(idx.shape)

        if is_tensor:
            return torch.tensor(flat, device=device, dtype=torch.float32)
        return flat