import torch
import torch.nn as nn
from typing import List

from .encoding import encode_string


DEFAULT_CHAR_TO_IDX = {
    'A': 0,
    'N': 1,
    'P': 2,
    'S': 3,
    ' ': 4,
    '<UNK>': 5,
    '<PAD>': 6
}

class CharEmbeddingModule(nn.Module):
    def __init__(
        self,
        char_embedding_dim: int,
        lstm_hidden_dim: int,
        device: torch.device,
        char_to_idx: dict = DEFAULT_CHAR_TO_IDX,
        unknown_idx: int = len(DEFAULT_CHAR_TO_IDX)-2,
        padding_idx: int = len(DEFAULT_CHAR_TO_IDX)-1,
    ):
        super(CharEmbeddingModule, self).__init__()
        self.char_to_idx = dict
        self.char_embedding_dim = char_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        self.unknown_idx = unknown_idx
        self.padding_idx = padding_idx

        self.vocab_size = len(char_to_idx)
        self.char_embiddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.char_embedding_dim,
            padding_idx=self.padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

    def forward(self, texts: List[str]):
        # encode input as configured
        encoded_texts = [encode_string(text) for text in texts]

        # convert encoded char to index for model compatibility
        sequences: List[torch.Tensor] = []
        for text in encoded_texts:
            indices = []
            for char in text:
                if char in self.char_to_idx:
                    indices.append(self.char_to_idx[char])
                else:
                    indices.append(self.unknown_idx)
            encoded_idx_tensor = torch.tensor(indices, dtype=torch.long)
            sequences.append(encoded_idx_tensor)

        # pad sequence
        padded_sequences = nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.padding_idx
        ).to(self.device)

        # forward to embedding layer
        char_embeds = self.char_embeddings(padded_sequences)

        # LSTM model
        lstm_out, (h_n, c_n) = self.lstm(char_embeds)

        # Use the last hidden state as the text feature
        text_features = h_n[-1]  # Shape: [batch_size, lstm_hidden_dim]

        return text_features
