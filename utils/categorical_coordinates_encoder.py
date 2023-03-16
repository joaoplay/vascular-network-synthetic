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

    def __init__(self, n_categories, encoder_path='categorical_coordinates_encoder.pt') -> None:
        super().__init__()
        self.n_categories = n_categories
        self.scale = None
        self.offset = None
        self.encoder_path = encoder_path

    def fit(self, data: torch.Tensor):
        """
        Given a dataset, it will compute the scale and offset parameters to encode the coordinates into a single integer.
        The zero value of data will be the center of the range of values. For instance, if the maximum number of categories
        is 11, the range of negative values will be [0, 4], the zero value will be 5 and the range of positive values will
        be [6, 10].
        :param data:
        :return:
        """
        # Compute the scale and offset parameters. Exclude nan values
        self.scale = (self.n_categories - 1) / (2 * torch.max(torch.abs(data[~torch.isnan(data)])))
        self.offset = (self.n_categories - 1) / 2

    def transform(self, data: torch.Tensor):
        """
        Given a data tensor, it will encode the coordinates into a single integer.
        :param data:
        :return:
        """
        # Check if the parameters have been computed
        if self.scale is None or self.offset is None:
            raise ValueError('The parameters of the categorical coordinates encoder have not been computed')

        # Encode the coordinates into a single integer
        categorical_data = torch.round(data * self.scale + self.offset)
        # Clamp the values to the range [0, n_categories]
        categorical_data = torch.clamp(categorical_data, 0, self.n_categories - 1).long()

        return categorical_data

    def inverse_transform(self, categorical_data: torch.Tensor):
        """
        Given a tensor with relative coordinates encoded into a single integer, it will perform the inverse
        transformation
        :param categorical_data:
        :return:
        """
        # Check if the parameters have been computed
        if self.scale is None or self.offset is None:
            raise ValueError('The parameters of the categorical coordinates encoder have not been computed')

        # Inverse transform the coordinates
        data = (categorical_data - self.offset) / self.scale

        return data

    def get_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the parameters of the categorical coordinates encoder
        :return:
        """
        return self.scale, self.offset

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

        self.scale, self.offset = torch.load(self.encoder_path)
