import torch
import numpy as np
from sgg.radius_classes import R_EDGES


class RadiusClassEncoder:
    """
    Encodes continuous radius values into discrete classes using predefined R_EDGES thresholds.
    R_EDGES = [1.0, 2.0, 3.0, 5.0, 9.0] defines 6 classes:
    - Class 0: radius < 1.0
    - Class 1: 1.0 <= radius < 2.0
    - Class 2: 2.0 <= radius < 3.0
    - Class 3: 3.0 <= radius < 5.0
    - Class 4: 5.0 <= radius < 9.0
    - Class 5: radius >= 9.0
    """
    
    def __init__(self, top_class_max: float = 38.0):
        self.r_edges = np.array(R_EDGES, dtype=np.float32)
        self.n_classes = len(self.r_edges) + 1  # 6 classes for 5 edges
        self.top_class_max = float(top_class_max)
    
    def transform(self, radius_data: torch.Tensor or np.ndarray):
        """
        Convert continuous radius values to discrete class indices.
        :param radius_data: tensor/array of radius values
        :return: tensor/array of class indices
        """
        is_tensor = isinstance(radius_data, torch.Tensor)
        device = radius_data.device if is_tensor else None
        #kept breaking when i tried this with cuda, but this 
        #fixes it i guess ???
        if is_tensor:
            radius_data = radius_data.detach().cpu().numpy()
        
        #use searchsorted to find which class each radius belongs to
        class_indices = np.searchsorted(self.r_edges, radius_data, side='left')
        
        #clamp to valid range [0, n_classes-1]
        #if radius is less than 0, clip it to the first class
        #if the radius is greater than the last class, clip it to that one
        #prevents out of bounds errors in case of unexpected radius values 
        class_indices = np.clip(class_indices, 0, self.n_classes - 1)
        
        if is_tensor:
            class_indices = torch.from_numpy(class_indices).long().to(device)
        
        return class_indices
    
    def inverse_transform(self, class_indices: torch.Tensor or np.ndarray):
        """
        Convert class indices back to approximate radius values (using class midpoints).
        :param class_indices: tensor/array of class indices
        :return: approximate radius values
        """
        is_tensor = isinstance(class_indices, torch.Tensor)
        device = class_indices.device if is_tensor else None
        if is_tensor:
            class_indices = class_indices.detach().cpu().numpy()

        # Clamp model outputs to valid radius class range to prevent out-of-bounds indexing.
        class_indices = np.clip(class_indices, 0, self.n_classes - 1).astype(np.int64)
        
        #create edges with boundaries for inverse transform
        edges_with_bounds = np.concatenate([[0], self.r_edges, [self.top_class_max]])
        
        #get midpoint of each class
        #to be honest, im kinda unsure if this is the best way to do it
        #joao can you look at this please
        #maybe i should use randint ? 
        midpoints = (edges_with_bounds[:-1] + edges_with_bounds[1:]) / 2
        radius_values = midpoints[class_indices]
        
        if is_tensor:
            radius_values = torch.from_numpy(radius_values).float().to(device)
        
        return radius_values