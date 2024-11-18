import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from enum import Enum

from .encoding import encode_string


DEFAULT_CHAR_TO_IDX = {
    'A': 0,
    'N': 1,
    'S': 2,
    ' ': 3,
    '<PAD>': 4
}

class AggregatorMethod(Enum):
    AVG = 0
    LSTM = 1
    BILSTM = 2

class CharEmbeddingModule(nn.Module):
    def __init__(
        self,
        use_embedding: bool,
        aggregation_method: AggregatorMethod,
        # mode-specific params ------
        char_embedding_dim: int = 1,
        lstm_hidden_dim: int = 1,
        num_lstm_layer: int = 1,
        # ---------------------------
        device: torch.device = 'cpu',
        char_to_idx: dict = DEFAULT_CHAR_TO_IDX,
        padding_idx: int = len(DEFAULT_CHAR_TO_IDX)-1,
        max_seq_length:int = 256
    ):
        super(CharEmbeddingModule, self).__init__()
        self.use_embedding = use_embedding
        self.aggregation_method = aggregation_method
        self.char_embedding_dim = char_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layer = num_lstm_layer
        self.device = device
        self.char_to_idx = char_to_idx
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length

        self.vocab_size = len(char_to_idx)
        self.bidirectional = (self.aggregation_method == AggregatorMethod.BILSTM)

        # encoding components
        if self.use_embedding:
            self.input_dim = self.char_embedding_dim
            self.char_embeddings = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.char_embedding_dim,
                padding_idx=self.padding_idx
            )
        else:
            self.input_dim = self.vocab_size # one-hot input
        
        # aggregator components
        if self.aggregation_method in [AggregatorMethod.LSTM, AggregatorMethod.BILSTM]:
            self.lstm = nn.LSTM(
                input_size=(self.char_embedding_dim if self.use_embedding else self.vocab_size),
                hidden_size=self.lstm_hidden_dim,
                batch_first=True,
                num_layers=self.num_lstm_layer,
                bidirectional=self.bidirectional
            )
            self.output_dim = self.lstm_hidden_dim * (2 if self.bidirectional else 1)
        elif self.aggregation_method == AggregatorMethod.AVG:
            # -1 for one-hot due to removal of padding index (which should always be 0)
            self.output_dim = self.input_dim - (1 if not self.use_embedding else 0)
        else:
            raise ValueError("Invalid aggregation_method. Choose value from AggregatorMethod enum.")

    def __repr__(self):
        repr_string = "CharEmbeddingModule(\n"

        repr_string += f"  use_embedding={self.use_embedding},\n"
        repr_string += f"  aggregation_method={self.aggregation_method},\n"

        if self.use_embedding:
            repr_string += f"  char_embedding_dim={self.char_embedding_dim},\n"
        if self.aggregation_method in [AggregatorMethod.LSTM, AggregatorMethod.BILSTM]:
            repr_string += f"  lstm_hidden_dim={self.lstm_hidden_dim},\n"
            repr_string += f"  num_lstm_layer={self.num_lstm_layer},\n"

        repr_string += ")"

        return repr_string

    def preprocess(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]: # (preprocessed_tensor, seq_lengths)
        # encode input as configured
        # ex: ["he llo.", "AB-33"] -> ["AA AAAP", "AASNN"]
        encoded_texts = [encode_string(text) for text in texts]
        
        # true_seq_lengths processing for efficient torch rnn pack_sequence
        # empty string length is 1 as it will be processed as 1 padding token
        # non-empty string will be trimmed to max_seq_length
        true_seq_lengths = []
        for text in encoded_texts:
            if len(text) == 0:
                true_len = 1
            else:
                true_len = min(len(text), self.max_seq_length)
            true_seq_lengths.append(true_len)
        true_seq_lengths = torch.tensor(true_seq_lengths)

        # convert encoded char to index for model compatibility
        # ex: ["AA AAAP", "AASNN"] -> [[0,0,4,0,0,0,2], [0,0,2,1,1]]
        # for empty string, treat as 1 padding token
        sequences: List[torch.Tensor] = []
        for text in encoded_texts:
            if text == "":
                indices = [self.padding_idx]
            else:
                indices = []
                for char in text[:self.max_seq_length]:
                    if char in self.char_to_idx:
                        indices.append(self.char_to_idx[char])
            encoded_idx_tensor = torch.tensor(indices, dtype=torch.long)
            sequences.append(encoded_idx_tensor)

        # pad sequence for batch feeding
        padded_sequences = nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.padding_idx
        ).to(self.device)

        # for one-hot encoding
        if self.use_embedding == False:
            tensor = F.one_hot(padded_sequences, num_classes=self.vocab_size).float()
        else:
            tensor = padded_sequences
        return tensor, true_seq_lengths

    def forward(
        self,
        tensor: torch.Tensor,
        seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        device = self.device

        if self.use_embedding:
            # embedding
            tensor = self.char_embeddings(tensor.to(device))
        else:
            # already one-hot from preprocessing
            tensor = tensor.to(device)
        
        if self.aggregation_method in [AggregatorMethod.LSTM, AggregatorMethod.BILSTM]:
            # pack the padded sequence to reduce unnecessary computation
            # see https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
            # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
            packed_input = nn.utils.rnn.pack_padded_sequence(
                tensor, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (h_n, c_n) = self.lstm(packed_input)

            # Get the final hidden state
            if self.bidirectional:
                # h_n shape: (num_layers * 2, batch_size, hidden_size)
                # Reshape h_n to (num_layers, 2, batch_size, hidden_size)
                h_n = h_n.view(self.num_lstm_layer, 2, -1, self.lstm_hidden_dim)
                # Take the last layer's hidden states
                h_n_forward = h_n[-1, 0, :, :]  # Forward direction
                h_n_backward = h_n[-1, 1, :, :]  # Backward direction
                # Concatenate forward and backward hidden states
                h_n = torch.cat((h_n_forward, h_n_backward), dim=1)
            else:
                # h_n shape: (num_layers, batch_size, hidden_size)
                h_n = h_n[-1, :, :]  # Take the last layer's hidden state
            
            # LSTM text features
            text_features = h_n
        
        elif self.aggregation_method == AggregatorMethod.AVG:
            # create mask to exclude padding from averaging
            batch_size, padded_seq_length, num_features = tensor.shape
            mask = torch.arange(padded_seq_length).unsqueeze(0).expand(batch_size, padded_seq_length) < seq_lengths.unsqueeze(1)

            # apply mask and get masked mean
            masked_tensor = tensor*mask.unsqueeze(-1)
            masked_sum = masked_tensor.sum(dim=1)
            seq_length_divider = mask.sum(dim=1, keepdim=True)

            masked_mean = masked_sum/seq_length_divider

            # remove padding feature for one-hot mode
            if not self.use_embedding:
                masked_mean = torch.cat(
                    (
                        masked_mean[:, :self.padding_idx],
                        masked_mean[:, self.padding_idx+1:]
                    ),
                    dim=1
                )

            text_features = masked_mean

        else:
            raise ValueError("Invalid aggregation method specified.")

        return text_features
