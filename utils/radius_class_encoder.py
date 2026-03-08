import torch
import numpy as np
from sgg.radius_classes import R_EDGES


class RadiusClassEncoder:
    """
    Encodes continuous radius values into discrete classes using different thresholds.
    - Class 0: radius < 1.0
    - Class 1: 1.0 <= r < 2.0
    - Class 2: 2.0 <= r < 3.0
    - Class 3: 3.0 <= r < 5.0
    - Class 4: 5.0 <= r < 9.0
    - Class 5: r >= 9.0
    """
    
    def __init__(self, top_class_max: float = 38.0):
        self.r_edges = np.array(R_EDGES, dtype=np.float32)
        self.n_classes = len(self.r_edges) + 1  # 6 classes for 5 edges    
        self.top_class_max = float(top_class_max)
    def transform(self, radius_data: torch.Tensor):
        """
        Convert continuous radius values to discrete class indices.
        :param radius_data: tensor of radius values
        :return: tensor of class indices
        """
        #kept breaking when i tried this with cuda
        is_tensor = isinstance(radius_data, torch.Tensor)
        if not is_tensor:
            radius_data = torch.tensor(radius_data)

        device = radius_data.device
        #ensure a floating dtype for comparisons
        if not radius_data.dtype.is_floating_point:
            radius_data = radius_data.float()

        edges = torch.as_tensor(self.r_edges, dtype=radius_data.dtype, device=device)
        class_indices = torch.searchsorted(edges, radius_data)
        class_indices = class_indices.clamp(0, self.n_classes - 1).long()

        return class_indices
    
    def inverse_transform(self, class_indices):
        labels = ['tiny', 'small', 'medium', 'normal', 'large', 'huge']

        if isinstance(class_indices, torch.Tensor):
            class_indices = class_indices.detach().cpu().numpy()

        arr = np.clip(np.asarray(class_indices).astype(int), 0, self.n_classes - 1)
        result = np.vectorize(labels.__getitem__)(arr)

        #if its a scalar, return a string, otherwise, return a list
        return result.item() if result.shape == () else result.tolist()

    #i wanted the class to be a label in the inverse
    #but i kept having errors in lists expecting squeeze and didnt know how to fix so i created this function below
    #and it fixed
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

        edges = self.r_edges
        top = self.top_class_max

        def idx_to_val(k):
            if k <= 0:
                return float(edges[0] / 2.0)
            if k < len(edges):
                return float((edges[k - 1] + edges[k]) / 2.0)
            return float((edges[-1] + top) / 2.0)
        

        # vectorize
        flat = np.asarray([idx_to_val(int(k)) for k in np.ravel(idx)], dtype=np.float32)
        flat = flat.reshape(idx.shape)

        if is_tensor:
            return torch.tensor(flat, device=device, dtype=torch.float32)
        return flat