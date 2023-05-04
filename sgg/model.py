import torch
from torch import nn
from torch.autograd import Variable


class GraphEncoderRNN(torch.nn.Module):
    """
    The graph encoder is responsible for encoding a path of relative coordinates into a single context vector (the
    final hidden state of the GRU). An embedding layer is used map from categorical coordinates to continuous
    embeddings.
    """

    def __init__(self, n_dimensions: int, n_classes: int, hidden_size: int, num_layers: int, embedding_size: int,
                 is_bidirectional: bool) -> None:
        """
        :param n_dimensions: Can be 3D or 2D. Currently, most of the code is written for 3D, although it should be
                             this module is already prepared for 2D.
        :param n_classes: Number of classes to be used to discretize the spatial relative coordinates
        :param hidden_size: Size of the hidden state of the GRU
        :param num_layers: Number of layers of the GRU
        :param embedding_size: Size of the embedding vector
        :param is_bidirectional: Whether the GRU is bidirectional or not
        """
        super().__init__()
        self.n_dimensions = n_dimensions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional

        # Create an embedding layer
        self.embedding = nn.Embedding(n_classes, embedding_size)

        # Create a GRU as encoder. The input size is an embedding representation for each dimension (3 when in 3D)
        self.encoder = nn.GRU(input_size=embedding_size * n_dimensions, hidden_size=hidden_size,
                              num_layers=self.num_layers, bias=True, batch_first=True,
                              bidirectional=self.is_bidirectional)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Performs a forward pass through the encoder. The hidden_next is the most important output, as it is
        the context vector that will be used by the decoder (only the last one).
        :param x: Input tensor of shape (batch_size, seq_len, n_dimensions)
        :param h: Initial hidden state of shape (num_layers * num_directions, batch, hidden_size)
        :return:
        """
        embedded = self.embedding(x).view(x.size(0), x.size(1), -1)

        output, hidden_next = self.encoder(embedded, h)

        return output, hidden_next

    def init_hidden(self, batch_size):
        """
        Get a zero-initialized hidden state
        :param batch_size: The batch size
        :return:
        """
        return Variable(torch.zeros(self.num_layers * (2 if self.is_bidirectional else 1), batch_size,
                                    self.hidden_size))


class GraphDecoderRNN(nn.Module):
    """
    The graph decoder is responsible for predicting the follow-up nodes, given a context vector (an aggregated
    representation of the final hidden states of the GRU). Similar to the encoder, an embedding layer is used to
    map from categorical coordinates to continuous embeddings.
    """

    def __init__(self, n_dimensions: int, n_classes: int, hidden_size: int, num_layers: int, embedding_size: int,
                 is_bidirectional: bool) -> None:
        """
        :param n_dimensions: Can be 3D or 2D. Currently, most of the code is written for 3D, although it should be
                             this module is already prepared for 2D.
        :param n_classes: Number of classes to be used to discretize the spatial relative coordinates
        :param hidden_size: Size of the hidden state of the GRU
        :param num_layers: Number of layers of the GRU
        :param embedding_size: Size of the embedding vector
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.n_dimensions = n_dimensions

        # Init an embedding layer for the coordinates
        self.embedding = nn.Embedding(n_classes, embedding_size)
        # Init a GRU as decoder
        self.decoder = nn.GRU(input_size=embedding_size * n_dimensions, hidden_size=hidden_size,
                              num_layers=self.num_layers, bias=True, batch_first=True, dropout=0,
                              bidirectional=self.is_bidirectional)

        self.relu = nn.ReLU()
        # Use a fully connected layer to map from the GRU output to classes
        self.out = nn.Linear(hidden_size * (2 if self.is_bidirectional else 1), n_classes * n_dimensions)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Performs a forward pass through the decoder. A fully connected layer is used to map from the GRU output
        to the classes.
        :param x:
        :param h:
        :return:
        """
        # Compute embeddings
        output = self.embedding(x).view(x.size(0), x.size(1), -1)
        output = self.relu(output)

        output, hidden_next = self.decoder(output, h)

        # Pass the GRU output through a fully connected layer
        output = self.out(output).view(x.size(0), x.size(1), self.n_dimensions, -1)

        return output, hidden_next

    def init_hidden(self, batch_size):
        """
        Get a zero-initialized hidden state
        :param batch_size: The batch size
        :return:
        """
        return Variable(torch.zeros(self.num_layers * (2 if self.is_bidirectional else 1), batch_size,
                                    self.hidden_size))


class GraphSeq2Seq(nn.Module):
    """
    This module represent a sequence to sequence model for generating spatial graphs. The model is composed of
    two RNNs: an encoder and a decoder. The encoder is GRU that encodes the input random paths
    into a single hidden state. The decoder hidden state is initialized with the aggregated hidden states and
    predicts the position of next nodes in relation to the current active node.
    """

    def __init__(self, n_classes, max_output_nodes, n_dimensions=3, hidden_size=512, num_layers=4, embedding_size=200,
                 is_bidirectional=True, device='cpu') -> None:
        """
        :param n_classes: Number of classes to be used to discretize the spatial relative coordinates
        :param max_output_nodes: Maximum number of nodes. It corresponds to the maximum number of iterations through
                                 the decoder
        :param n_dimensions: Can be 3D or 2D. Currently, most of the code is written for 3D, although it should be
                             this module is already prepared for 2D.
        :param hidden_size: Hidden size of the RNNs
        :param num_layers: Number of layers of the RNNs
        :param embedding_size: The size of the embedding representation of the relative coordinates
        :param is_bidirectional: If True, the encoder and decoder will be bidirectional
        :param device: Device to be used for training (e.g. 'cpu', 'cuda')
        """
        super().__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.max_output_nodes = max_output_nodes
        self.is_bidirectional = is_bidirectional
        self.device = device

        # Initialize encoder and decoder
        self.encoder = GraphEncoderRNN(n_dimensions, n_classes, hidden_size, num_layers, embedding_size,
                                       is_bidirectional).to(device)
        self.decoder = GraphDecoderRNN(n_dimensions, n_classes, hidden_size, num_layers, embedding_size,
                                       is_bidirectional).to(device)

    def forward(self, x, y=None):
        """
        The forward method can be used for training and inference. The x is a batch with the relative coordinates
        between consecutive nodes in the random walks. The y represents the relative coordinates of the next nodes (
        those not included in the sampled paths). When y is None, the model is used for inference. Otherwise, it is
        used for training.

        The model is trained using teacher forcing.
        :param x:
        :param y:
        :return:
        """
        # Get the batch size
        batch_size = x.size(0)

        # Determine the number of layers of the encoder (depends on whether it is bidirectional or not)
        num_layers = self.num_layers * 2 if self.is_bidirectional else self.num_layers
        # Initialize the hidden state of the encoder
        batch_encoder_hidden = torch.zeros(num_layers, batch_size, self.hidden_size)

        # Determine the class that corresponds to zero relative coordinates. The same number of classes are attributed
        # to positive and negative relative coordinates. The middle point is the zero class.
        zero_class = (self.n_classes // 2) - 1

        # Iterate over a batch of input paths and encode them
        for batch_sample_idx in range(batch_size):
            sample = x[batch_sample_idx]

            # Get mask of elements in dim 0 of sample that are not all zeros
            non_padding_paths = torch.amax(torch.abs(sample - zero_class), dim=(1, 2)) > 0

            # Filter out all padding paths
            sample = sample[non_padding_paths]

            print(sample.shape)

            encoder_hidden = self.encoder.init_hidden(sample.size(0)).to(device=self.device)

            # Encode paths and aggregate hidden states. The hidden states are summed!
            out, encoder_hidden = self.encoder(sample, encoder_hidden)
            batch_encoder_hidden[:, batch_sample_idx, :] = torch.sum(encoder_hidden, dim=1)

        # Aggregate context
        decoder_hidden = batch_encoder_hidden.to(device=self.device)

        # Initialize the decoder input with the zero class. Note that no special token is used for the start of the
        # sequence. It is simply indicating that no movement existed in the previous node.
        decoder_start_input = torch.tensor([[[zero_class] * self.n_dimensions]]).repeat(batch_size, 1, 1).long().to(
            device=self.device)

        if y is not None:
            # TRAINING: Perform teacher forcing
            decoder_input = torch.cat([decoder_start_input, y[:, 1, 0:-1]], dim=1)
            # Note that the aggregated context is being passed to the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.view(-1, self.n_classes)
        else:
            # INFERENCE: For debugging purposes, the most probable class is selected at each step. In the future, a
            # multinomial distribution will be used to sample from the output probabilities.
            decoder_input = decoder_start_input
            all_decoder_outputs = []
            for i in range(self.max_output_nodes):
                # The aggregated context is being passed to the decoder. The decoder input is ALWAYS the previous
                # output
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # Select the most probable class
                decoder_output = decoder_output.squeeze(0).softmax(dim=2)
                indices = torch.topk(decoder_output, 1).indices
                decoder_input = indices.view(1, 1, 3)
                all_decoder_outputs.append(indices.view(-1))

            decoder_output = torch.stack(all_decoder_outputs, dim=0)

        return decoder_output

    @torch.no_grad()
    def generate(self, x):
        """
        This is an auxiliary method that can be used to infer the follow-up nodes from a set of random paths (x).
        :param x:
        :return:
        """
        return self.forward(x, None)

    def load(self, path):
        """
        Load encoder and decoder from a file
        """
