import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class GraphEncoderRNN(torch.nn.Module):
    """
    The graph encoder is responsible for encoding a path of relative coordinates into a single context vector (the
    final hidden state of the GRU). An embedding layer is used map from categorical coordinates to continuous
    embeddings.
    """

    def __init__(self, n_dimensions: int, n_classes: int, hidden_size: int, num_layers: int, embedding_size: int,
                 is_bidirectional: bool, n_extra_classes: int = 8) -> None:
        """
        :param n_dimensions: Can be 3D or 2D. Currently, most of the code is written for 3D, although it should be
                             this module is already prepared for 2D.
        :param n_classes: Number of classes to be used to discretize the spatial relative coordinates
        :param hidden_size: Size of the hidden state of the GRU
        :param num_layers: Number of layers of the GRU
        :param embedding_size: Size of the embedding vector
        :param is_bidirectional: Whether the GRU is bidirectional or not
        :param n_extra_classes: Number of classes for radius
        """
        super().__init__()
        self.n_dimensions = n_dimensions
        self.spatial_dims = min(3, n_dimensions)
        self.extra_dims = max(0, n_dimensions - self.spatial_dims)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional
        self.spatial_embedding = nn.Embedding(n_classes, embedding_size)
        self.extra_embedding = nn.Embedding(n_extra_classes, embedding_size) 

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
        spatial_embedded = self.spatial_embedding(x[:, :, :self.spatial_dims]).view(x.size(0), x.size(1), -1)
        if self.extra_dims > 0:
            extra_embedded = self.extra_embedding(x[:, :, self.spatial_dims:]).view(x.size(0), x.size(1), -1)
            embedded = torch.cat([spatial_embedded, extra_embedded], dim=2)
        else:
            embedded = spatial_embedded

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
                 is_bidirectional: bool, n_extra_classes: int = 8) -> None:
        """
        :param n_dimensions: Can be 3D or 2D. Currently, most of the code is written for 3D, although it should be
                             this module is already prepared for 2D.
        :param n_classes: Number of classes to be used to discretize the spatial relative coordinates
        :param hidden_size: Size of the hidden state of the GRU
        :param num_layers: Number of layers of the GRU
        :param embedding_size: Size of the embedding vector
        :param n_extra_classes: Number of classes for radius
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.n_dimensions = n_dimensions
        self.spatial_dims = min(3, n_dimensions)
        self.extra_dims = max(0, n_dimensions - self.spatial_dims) #incase we want to add more features other than radius
        self.n_extra_classes = n_extra_classes

        #use separate embeddings for spatial coordinates (xyz) and radius and any other future feature
        self.spatial_embedding = nn.Embedding(n_classes, embedding_size)
        self.extra_embedding = nn.Embedding(n_extra_classes, embedding_size)
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
        #compute embeddings for xyz and other features
        spatial_embedded = self.spatial_embedding(x[:, :, :self.spatial_dims]).view(x.size(0), x.size(1), -1)
        if self.extra_dims > 0:
            extra_embedded = self.extra_embedding(x[:, :, self.spatial_dims:]).view(x.size(0), x.size(1), -1)
            output = torch.cat([spatial_embedded, extra_embedded], dim=2)
        else:
            output = spatial_embedded
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

    def __init__(self, n_classes, max_output_nodes, n_dimensions=3, n_extra_classes=8, hidden_size=512, num_layers=4,
                 embedding_size=200, is_bidirectional=True, device='cpu') -> None:
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
        self.n_extra_classes = n_extra_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.max_output_nodes = max_output_nodes
        self.is_bidirectional = is_bidirectional
        self.device = device

        # Initialize encoder and decoder
        self.encoder = GraphEncoderRNN(n_dimensions, n_classes, hidden_size, num_layers, embedding_size,
                                       is_bidirectional, n_extra_classes=n_extra_classes).to(device)
        self.decoder = GraphDecoderRNN(n_dimensions, n_classes, hidden_size, num_layers, embedding_size,
                                       is_bidirectional, n_extra_classes=n_extra_classes).to(device)
        #layernorm to normalize aggregated hidden states per-layer across hidden_size
        self.hidden_layer_norm = nn.LayerNorm(hidden_size).to(device)

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

        spatial_dims = min(3, self.n_dimensions)

        # Determine the number of layers of the encoder (depends on whether it is bidirectional or not)
        num_layers = self.num_layers * 2 if self.is_bidirectional else self.num_layers
        # Initialize the hidden state of the encoder
        batch_encoder_hidden = torch.zeros(num_layers, batch_size, self.hidden_size, device=self.device)

        # Determine the class that corresponds to zero relative coordinates. The same number of classes are attributed
        # to positive and negative relative coordinates. The middle point is the zero class.
        zero_class = (self.n_classes // 2) - 1

        # Iterate over a batch of input paths and encode them
        for batch_sample_idx in range(batch_size):
            sample = x[batch_sample_idx]

            #het mask for non-padding paths using spatial coordinates only.
            spatial_sample = sample[:, :, :spatial_dims]
            non_padding_paths = torch.amax(torch.abs(spatial_sample - zero_class), dim=(1, 2)) > 0

            #filter out all padding paths
            sample = sample[non_padding_paths]

            encoder_hidden = self.encoder.init_hidden(sample.size(0)).to(device=self.device)

            out, encoder_hidden = self.encoder(sample, encoder_hidden)
            #aggregate hidden states across the path batch by taking the mean over paths,
            aggregated = torch.mean(encoder_hidden, dim=1)
            #apply layer norm
            aggregated = self.hidden_layer_norm(aggregated)
            batch_encoder_hidden[:, batch_sample_idx, :] = aggregated

        # Aggregate context
        decoder_hidden = batch_encoder_hidden.to(device=self.device)

        # Initialize decoder start token.
        # xyz dims use zero_class; radius start from 0.
        decoder_start_classes = [zero_class] * spatial_dims + [0] * (self.n_dimensions - spatial_dims)
        decoder_start_input = torch.tensor([decoder_start_classes], device=self.device).long().unsqueeze(0).repeat(
            batch_size, 1, 1)

        if y is not None:
            # TRAINING: Perform teacher forcing.
            #trainer.py already selects a random path per sample and passes y with shape
            #(batch_size, max_output_nodes, n_dimensions). 
            teacher_forcing_steps = y[:, 0:-1]

            decoder_input = torch.cat([decoder_start_input, teacher_forcing_steps], dim=1)
            # Note that the aggregated context is being passed to the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # Return shape: (batch_size, seq_len, n_dimensions, n_classes) for trainer to reshape properly
        else:
            # INFERENCE: Sample classes for each dimension from multinomial distributions.
            decoder_input = decoder_start_input
            all_decoder_outputs = []
            for i in range(self.max_output_nodes):
                # The aggregated context is being passed to the decoder. The decoder input is ALWAYS the previous
                # output
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_logits = decoder_output[:, -1, :, :]  # (batch_size, n_dimensions, n_classes)

                # Mask invalid classes for radius before softmax
                if self.n_dimensions > 3:
                    step_logits[:, 3:, self.n_extra_classes:] = float('-inf')

                step_probs = step_logits.softmax(dim=2)

                sampled_indices = torch.multinomial(
                    step_probs.reshape(-1, self.n_classes),
                    1
                ).view(batch_size, self.n_dimensions)

                decoder_input = sampled_indices.unsqueeze(1)
                all_decoder_outputs.append(sampled_indices)

            decoder_output = torch.stack(all_decoder_outputs, dim=1)  #(batch_size, max_output_nodes, n_dimensions)
            if batch_size == 1:
                decoder_output = decoder_output.squeeze(0)

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
        pass
