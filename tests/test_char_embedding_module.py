import unittest
import torch
import numpy as np

from src.models.char_embedding.model import CharEmbeddingModule, AggregatorMethod

class TestCharEmbeddingModule(unittest.TestCase):

    def setUp(self):
        self.dummy_texts = ['aBC.d 1234', '/', '22/08 11:02', 'A medium length sentence with alphabet.']
        self.dummy_texts_len = [len(s) for s in self.dummy_texts]
        self.max_dummy_len = max(self.dummy_texts_len)

    def test_init_simple(self):
        hist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.AVG
        )
        self.assertFalse(hasattr(hist_module, 'char_embeddings'))
        self.assertFalse(hasattr(hist_module, 'lstm'))

    def test_init_embs(self):
        embedding_avg_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.AVG
        )
        self.assertTrue(hasattr(embedding_avg_module, 'char_embeddings'))
        self.assertFalse(hasattr(embedding_avg_module, 'lstm'))

    def test_init_lstm(self):
        char_dist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.LSTM
        )
        embedding_dist_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.LSTM
        )
        self.assertTrue(hasattr(char_dist_module, 'lstm'))
        self.assertTrue(hasattr(embedding_dist_module, 'lstm'))
        # assert 1 directional
        self.assertFalse(char_dist_module.bidirectional)
        self.assertFalse(embedding_dist_module.bidirectional)
    
    def test_init_bi_lstm(self):
        embedding_bi_dist_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.BILSTM
        )
        self.assertTrue(hasattr(embedding_bi_dist_module, 'lstm'))
        self.assertTrue(embedding_bi_dist_module.bidirectional)

    def test_preprocess_one_hot(self):
        char_dist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.LSTM
        )

        # assert preprocess tensor shape
        preprocessed_tensor, seq_lengths = char_dist_module.preprocess(self.dummy_texts)
        self.assertEqual(
            seq_lengths.int().tolist(),
            self.dummy_texts_len
        )
        self.assertEqual(
            list(preprocessed_tensor.shape),
            [len(self.dummy_texts), self.max_dummy_len, char_dist_module.vocab_size]
        )
        # assert only 0 and 1 value
        unique_value = torch.unique(preprocessed_tensor)
        self.assertCountEqual(
            unique_value.int().tolist(),
            [0, 1]
        )
        # assert one-hot
        self.assertEqual(
            preprocessed_tensor[0, :self.dummy_texts_len[0], :].int().tolist(),
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0]
            ]
        )

    def test_preprocess_embedding(self):
        CHAR_EMBEDDING_DIM = 16
        embedding_dist_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.LSTM,
            char_embedding_dim=CHAR_EMBEDDING_DIM
        )

        # assert preprocess tensor shape
        preprocessed_tensor, seq_lengths = embedding_dist_module.preprocess(self.dummy_texts)
        self.assertEqual(
            seq_lengths.int().tolist(),
            self.dummy_texts_len
        )
        self.assertEqual(
            list(preprocessed_tensor.shape),
            [len(self.dummy_texts), self.max_dummy_len]
        )
        # assert processed as vocab index
        unique_value = torch.unique(preprocessed_tensor)
        self.assertTrue(min(unique_value) >= 0)
        self.assertTrue(max(unique_value) <= embedding_dist_module.vocab_size)

    def test_forward_average_one_hot(self):
        BATCH_SIZE = len(self.dummy_texts)
        hist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.AVG
        )
        preprocessed_tensor, seq_lengths = hist_module.preprocess(self.dummy_texts)
        # assert output shape
        avg_out = hist_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(avg_out.shape),
            [BATCH_SIZE, hist_module.vocab_size - 1]
        )
        # assert test case output
        np.testing.assert_almost_equal(
            avg_out[0].tolist(),
            [0.4, 0.4, 0.1, 0.0, 0.1, 0.0]
        )
        np.testing.assert_almost_equal(
            avg_out[1].tolist(),
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )

    def test_forward_average_embedding(self):
        BATCH_SIZE = len(self.dummy_texts)
        EMBEDDING_DIM = 3
        embedding_avg_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.AVG,
            char_embedding_dim=EMBEDDING_DIM
        )
        preprocessed_tensor, seq_lengths = embedding_avg_module.preprocess(self.dummy_texts)
        # assert output shape
        avg_out = embedding_avg_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(avg_out.shape),
            [BATCH_SIZE, embedding_avg_module.char_embedding_dim]
        )
        # assert average excluded padding
        preprocessed_tensor, seq_lengths = embedding_avg_module.preprocess([self.dummy_texts[1]])
        one_char_avg_output = embedding_avg_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            avg_out[1].tolist(),
            one_char_avg_output.squeeze(0).tolist()
        )

    def test_forward_lstm_one_hot(self):
        LSTM_HIDDEN_DIM = 16
        BATCH_SIZE = len(self.dummy_texts)
        char_dist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.LSTM,
            lstm_hidden_dim=LSTM_HIDDEN_DIM
        )
        preprocessed_tensor, seq_lengths = char_dist_module.preprocess(self.dummy_texts)
        # assert output shape
        lstm_out = char_dist_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(lstm_out.shape),
            [BATCH_SIZE, LSTM_HIDDEN_DIM]
        )
        # assert hidden state randomness (have many unique values)
        self.assertTrue(torch.unique(lstm_out).shape[0] >= 2)

    def test_forward_lstm_embedding(self):
        LSTM_HIDDEN_DIM = 48
        EMBEDDING_DIM = 16
        BATCH_SIZE = len(self.dummy_texts)
        embedding_dist_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.LSTM,
            char_embedding_dim=EMBEDDING_DIM,
            lstm_hidden_dim=LSTM_HIDDEN_DIM
        )
        preprocessed_tensor, seq_lengths = embedding_dist_module.preprocess(self.dummy_texts)
        # assert output shape
        lstm_out = embedding_dist_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(lstm_out.shape),
            [BATCH_SIZE, LSTM_HIDDEN_DIM]
        )
        # assert hidden state randomness (have many unique values)
        self.assertTrue(torch.unique(lstm_out).shape[0] >= 2)

    def test_forward_bilstm_one_hot(self):
        LSTM_HIDDEN_DIM = 16
        BATCH_SIZE = len(self.dummy_texts)
        char_dist_module = CharEmbeddingModule(
            use_embedding=False,
            aggregation_method=AggregatorMethod.BILSTM,
            lstm_hidden_dim=LSTM_HIDDEN_DIM
        )
        preprocessed_tensor, seq_lengths = char_dist_module.preprocess(self.dummy_texts)
        # assert bidirectional model output is double the size of hidden dim
        self.assertEqual(LSTM_HIDDEN_DIM*2, char_dist_module.output_dim)
        # assert output shape
        lstm_out = char_dist_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(lstm_out.shape),
            [BATCH_SIZE, LSTM_HIDDEN_DIM*2]
        )
        # assert hidden state randomness (have many unique values)
        self.assertTrue(torch.unique(lstm_out).shape[0] >= 2)

    def test_forward_bilstm_embedding(self):
        LSTM_HIDDEN_DIM = 48
        EMBEDDING_DIM = 16
        BATCH_SIZE = len(self.dummy_texts)
        embedding_dist_module = CharEmbeddingModule(
            use_embedding=True,
            aggregation_method=AggregatorMethod.BILSTM,
            char_embedding_dim=EMBEDDING_DIM,
            lstm_hidden_dim=LSTM_HIDDEN_DIM
        )
        preprocessed_tensor, seq_lengths = embedding_dist_module.preprocess(self.dummy_texts)
        # assert bidirectional model output is double the size of hidden dim
        self.assertEqual(LSTM_HIDDEN_DIM*2, embedding_dist_module.output_dim)
        # assert output shape
        lstm_out = embedding_dist_module(preprocessed_tensor, seq_lengths)
        self.assertEqual(
            list(lstm_out.shape),
            [BATCH_SIZE, LSTM_HIDDEN_DIM*2]
        )
        # assert hidden state randomness (have many unique values)
        self.assertTrue(torch.unique(lstm_out).shape[0] >= 2)


if __name__ == '__main__':
    unittest.main()
