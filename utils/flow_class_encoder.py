import torch
import numpy as np
from sgg.flow_classes import flow_edges


class FlowClassEncoder:
    """
    Encodes continuous flow values into discrete classes using flow_edges = [0.53, 1.42, 3.93, 10.68, 33.97].
    - Class 0: f < 0.53
    - Class 1: 0.53 <= f < 1.42
    - Class 2: 1.42 <= f < 3.93
    - Class 3: 3.93 <= f < 10.68
    - Class 4: 10.68 <= f < 33.97
    - Class 5: f >= 33.97
    """
    def __init__(self):
        self.flow_edges = np.array(flow_edges, dtype=np.float32)
        self.n_classes = len(self.flow_edges) + 1


    def transform(self, flow_data: torch.Tensor):
        """
        Convert continuous flow values to discrete class indices.
        :param flow_data: tensor of flow values
        :return: tensor of class indices
        """

        is_tensor = isinstance(flow_data, torch.Tensor)
        if not is_tensor:
            flow_data = torch.tensor(flow_data)

        device = flow_data.device
        #ensure a floating dtype for comparisons
        if not flow_data.dtype.is_floating_point:
            flow_data = flow_data.float()
        flow_data = flow_data.abs()

        edges = torch.as_tensor(self.flow_edges, dtype=flow_data.dtype, device=device)
        class_indices = torch.searchsorted(edges, flow_data.contiguous())
        class_indices = class_indices.clamp(0, self.n_classes - 1).long()

        return class_indices
    
    def class_to_value(self, class_indices):
        """
        Convert class indices to a representative float flow value (midpoint of each bin).
        """
        is_tensor = isinstance(class_indices, torch.Tensor)
        if is_tensor:
            device = class_indices.device
            idx = class_indices.detach().cpu().numpy().astype(int)
        else:
            idx = np.asarray(class_indices).astype(int)

        idx = np.clip(idx, 0, self.n_classes - 1)
        edges = self.flow_edges
        lo = np.concatenate([[0.0], edges])
        hi = np.concatenate([edges, [edges[-1] * 3]])

        flat_idx = np.ravel(idx)
        flat = np.array([
            np.random.uniform(lo[k], hi[k]) for k in flat_idx
        ], dtype=np.float32).reshape(idx.shape)

        if is_tensor:
            return torch.tensor(flat, device=device, dtype=torch.float32)
        return flat

    def inverse_transform(self, class_indices):
        labels = ['Q<0.53', 'Q<1.42', 'Q<3.93', 'Q<10.68', 'Q<33.97', 'Q>33.97']

        if isinstance(class_indices, torch.Tensor):
            class_indices = class_indices.detach().cpu().numpy()

        arr = np.clip(np.asarray(class_indices).astype(int), 0, self.n_classes - 1)
        result = np.vectorize(labels.__getitem__)(arr)

        #if its a scalar, return a string, otherwise, return a list
        return result.item() if result.shape == () else result.tolist()
