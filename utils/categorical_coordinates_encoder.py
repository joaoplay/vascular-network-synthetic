from os import path

import torch


class CategoricalCoordinatesEncoder:
    """
    This class is used to encode categorical coordinates into a single integer. Given a reference
    dataset and a maximum number of categories, it will scale the coordinates to the maximum number of categories
    and then encode them into a single integer. We also consider that the coordinates are centered around the
    zero value, with the range of negative values equal to the range of positive values. For example, if the maximum
    number of categories is 10, the range of negative values must be [0, 4] and the range of positive values must be
    [6, 10]. The category 5 is reserved for the zero value.
    """

    def __init__(self, n_categories, encoder_path='categorical_coordinates_encoder.pt',
                 threshold: float = 23.0, inner_fraction: float = 0.6) -> None:
        super().__init__()
        self.n_categories = n_categories
        self.threshold = threshold
        self.inner_fraction = inner_fraction
        self.min_value = None
        self.max_value = None
        self.bin_edges = None
        self.bin_centers = None
        self.encoder_path = encoder_path

    def fit(self, data: torch.Tensor | None = None, *, min_value: torch.Tensor | float | None = None,
            max_value: torch.Tensor | float | None = None):
        if data is not None:
            data = data[torch.isfinite(data)]
            min_value = torch.min(data)
            max_value = torch.max(data)


        self.min_value = torch.as_tensor(min_value, dtype=torch.float32)
        self.max_value = torch.as_tensor(max_value, dtype=torch.float32)

        n_edges = self.n_categories - 1
        zero_class = self.n_categories // 2
        n_neg = zero_class
        n_pos = n_edges - n_neg

        min_val = float(self.min_value)
        max_val = float(self.max_value)

        thr = abs(float(self.threshold))#distance for where the resolution is higher 
        inner_fraction = min(max(float(self.inner_fraction), 0.0), 1.0)

        neg_min = min(min_val, -thr)
        pos_max = max(max_val, thr)

        lower_res_neg = min(round(n_neg * (1.0 - inner_fraction)), n_neg - 1)
        higher_res_neg = n_neg - lower_res_neg
        lower_res_pos = min(round(n_pos * (1.0 - inner_fraction)), n_pos - 1)
        higher_res_pos = n_pos - lower_res_pos

        if neg_min < -thr and lower_res_neg > 0:
            neg_outer = torch.linspace(neg_min, -thr, lower_res_neg + 1, dtype=torch.float32)[1:]
            neg_inner = torch.linspace(-thr, 0.0, higher_res_neg + 1, dtype=torch.float32)[1:]
            negative_edges = torch.cat([neg_outer, neg_inner])
        else:
            negative_edges = torch.linspace(neg_min, 0.0, n_neg + 1, dtype=torch.float32)[1:]

        if pos_max > thr and lower_res_pos > 0:
            pos_inner = torch.linspace(0.0, thr, higher_res_pos + 1, dtype=torch.float32)[1:]
            pos_outer = torch.linspace(thr, pos_max, lower_res_pos + 1, dtype=torch.float32)[1:]
            positive_edges = torch.cat([pos_inner, pos_outer])
        else:
            positive_edges = torch.linspace(0.0, pos_max, n_pos + 1, dtype=torch.float32)[1:]

        self.bin_edges = torch.cat([negative_edges, positive_edges])

        # Bin centres for inverse_transform
        edges = self.bin_edges
        first_width = edges[1] - edges[0]
        last_width = edges[-1] - edges[-2]
        lower = torch.cat([edges[[0]] - first_width, edges])
        upper = torch.cat([edges, edges[[-1]] + last_width])
        self.bin_centers = (lower + upper) / 2
        self.bin_centers[zero_class] = 0.0

    def transform(self, data: torch.Tensor):
        if self.bin_edges is None:
            raise ValueError('The parameters of the categorical coordinates encoder have not been computed')


        self.fit(min_value=self.min_value, max_value=self.max_value)

        mask = torch.isfinite(data)
        edges = self.bin_edges.to(device=data.device, dtype=data.dtype)
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

        categorical_data = torch.bucketize(data, edges, right=True)
        categorical_data = torch.clamp(categorical_data, 0, self.n_categories - 1).long()

        padding_data = torch.full_like(categorical_data, self.n_categories)
        return torch.where(mask, categorical_data, padding_data)

    def inverse_transform(self, categorical_data: torch.Tensor):
        """
        Decode class indices back to continuous values using the centre of each bin.
        :param categorical_data:
        :return:
        """
        if self.bin_centers is None:
            raise ValueError('The parameters of the categorical coordinates encoder have not been computed')

        categorical_data = categorical_data.long().clamp(0, self.n_categories - 1)
        return self.bin_centers.to(categorical_data.device)[categorical_data]

    def get_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the parameters of the categorical coordinates encoder
        :return:
        """
        return self.bin_edges, self.bin_centers, self.min_value, self.max_value

    def save_parameters(self):
        """
        Save the parameters of the categorical coordinates encoder
        :return:
        """
        torch.save(self.get_parameters(), self.encoder_path)

    def load_parameters(self):
        """
        Load the parameters of the categorical coordinates encoder
        :return:
        """
        # Check if the file exists
        if not path.exists(path.join(self.encoder_path)):
            raise FileNotFoundError('Cannot find the file with the parameters of the categorical coordinates encoder')

        parameters = torch.load(self.encoder_path)
        if len(parameters) == 3:
            class_width, self.min_value, self.max_value = parameters
            centers = self.min_value + torch.arange(self.n_categories, dtype=torch.float32) * class_width
            self.bin_centers = centers
            self.bin_edges = (centers[:-1] + centers[1:]) / 2
            if (self.bin_edges.numel() != self.n_categories - 1
                    or self.bin_centers.numel() != self.n_categories
                    or not torch.isfinite(self.bin_edges).all()
                    or not torch.isfinite(self.bin_centers).all()
                    or not torch.all(self.bin_edges[1:] > self.bin_edges[:-1])):
                self.fit(min_value=self.min_value, max_value=self.max_value)
            return

        self.bin_edges, self.bin_centers, self.min_value, self.max_value = parameters
        if (self.bin_edges.numel() != self.n_categories - 1
                or self.bin_centers.numel() != self.n_categories
                or not torch.isfinite(self.bin_edges).all()
                or not torch.isfinite(self.bin_centers).all()
                or not torch.all(self.bin_edges[1:] > self.bin_edges[:-1])):
            self.fit(min_value=self.min_value, max_value=self.max_value)
