import torch.nn as nn
import torch
from typing import Optional, Tuple
from torch import Tensor


class Model(nn.Module):
    """
    Класс Model представляет собой нейронную сеть на основе LSTM для обработки последовательностей, таких как текст.
    Она состоит из слоев эмбеддингов, LSTM и линейного слоя для получения логитов, соответствующих размерам словаря.

    Аргументы:
        vocab_size (int): Размер словаря (количество уникальных слов).
        emb_size (int, необязательный): Размерность эмбеддингов. По умолчанию 128.
        num_layers (int, необязательный): Количество слоев в LSTM. По умолчанию 1.
        hidden_size (int, необязательный): Размерность скрытого состояния LSTM. По умолчанию 256.
        dropout (float, необязательный): Вероятность отключения нейронов (dropout) между слоями LSTM. По умолчанию 0.0.

    Методы:
        forward(x, hx=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
            Проводит прямое распространение через сеть.
    """
    def __init__(
            self,
            vocab_size: int,
            emb_size: int = 128,
            num_layers: int = 1,
            hidden_size: int = 256,
            dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers, dropout=self.dropout)
        self.logits = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(
            self,
            x: Tensor,
            hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Проводит прямое распространение через сеть.

        Аргументы:
            x (Tensor): Входные данные (индексы слов) размером (batch_size, seq_len).
            hx (Optional[Tuple[Tensor, Tensor]]): Начальные скрытые состояния (h_n, c_n) для LSTM. По умолчанию None.

        Возвращает:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - Логиты (предсказания для каждого слова в последовательности) размером (batch_size, seq_len, vocab_size).
                - Пара скрытых состояний (h_n, c_n), где h_n и c_n — это последние скрытые и клеточные состояния LSTM.
        """
        embedded = self.embeddings(x)  # Размерность: (batch_size, seq_len, emb_size)

        lstm_out, (h_n, c_n) = self.lstm(embedded, hx)  # lstm_out размерностью: (batch_size, seq_len, hidden_size)

        logits = self.logits(lstm_out)  # Размерность: (batch_size, seq_len, vocab_size)

        return logits, (h_n, c_n)
