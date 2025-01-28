from typing import List, Optional
from tqdm import tqdm
from torch.utils.data import Dataset
from .ByteTokenizer import ByteTokenizer
class MyDataset(Dataset):
    """
    Класс MyDataset представляет собой набор данных для работы с текстами, закодированными с помощью токенизатора.

    Атрибуты:
    ----------
    max_length : Optional[int]
        Максимальная длина последовательности токенов (по умолчанию None, что означает отсутствие ограничения).
    data : List[List[int]]
        Список последовательностей токенов для каждого текста.

    Параметры:
    ----------
    texts : List[str]
        Список текстов для токенизации.
    tokenizer : ByteTokenizer
        Токенизатор, который преобразует текст в последовательность токенов.
    max_length : Optional[int], по умолчанию None
        Максимальная длина последовательности токенов (опционально). Если задано, обрезает последовательность до этой длины.

    Методы:
    -------
    __getitem__(idx: int) -> List[int]
        Возвращает последовательность токенов для текста по индексу, обрезанную до max_length.
    __len__() -> int
        Возвращает количество текстов в наборе данных

    """
    def __init__(self, texts: List[str], tokenizer: ByteTokenizer, max_length: Optional[int] = None):
        self.max_length = max_length
        self.data = []
        for text in tqdm(texts):
            token_ids = [tokenizer.bos_token]
            token_ids += tokenizer.encode(text)
            token_ids += [tokenizer.eos_token]
            self.data.append(token_ids)

    def __getitem__(self, idx: int) -> List[int]:
        """
        Возвращает последовательность токенов для текста по индексу, обрезанную до max_length.

        Параметры:
        ----------
        idx : int
            Индекс элемента в наборе данных, который нужно вернуть

        Возвращает:
        -----------
        List[int]
            Усеченный список номеров токенов
        """
        return self.data[idx][:self.max_length]

    def __len__(self) -> int:
        """Возвращает количество текстов в наборе данных."""
        return len(self.data)
