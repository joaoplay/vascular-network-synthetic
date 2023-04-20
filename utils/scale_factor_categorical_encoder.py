import torch


class ScaleFactorCategoricalCoordinatesEncoder:

    def __init__(self, n_categories, scale_factor=4) -> None:
        super().__init__()
        self.n_categories = n_categories
        self.scale_factor = scale_factor
        self.n_categories_mid = (self.n_categories - 1) // 2

    def transform(self, data: torch.Tensor):
        """
        Given a data tensor, it will encode the coordinates into a single integer.
        :param data:
        :return:
        """
        # Encode the coordinates into a single integer
        categorical_data = torch.round((data[~torch.isnan(data)] * self.scale_factor + self.n_categories_mid)).long()
        # Clamp the values to the range [0, n_categories]
        categorical_data = torch.clamp(categorical_data, 0, self.n_categories - 1).long()

        # Create empty tensor with the same shape as data
        encoded_data = torch.empty_like(data).long()

        encoded_data[~torch.isnan(data)] = categorical_data
        # Change NaN values to 0
        encoded_data[torch.isnan(data)] = self.n_categories

        return encoded_data

    def inverse_transform(self, categorical_data: torch.Tensor):
        """
        Given a tensor with relative coordinates encoded into a single integer, it will perform the inverse
        transformation
        :param categorical_data:
        :return:
        """
        data = (categorical_data - self.n_categories_mid) / self.scale_factor

        return data
