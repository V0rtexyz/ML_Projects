from typing import List, Tuple, Dict
from tqdm import tqdm
from .ByteTokenizer import ByteTokenizer
from .ByteTokenizer import merge
from .ByteTokenizer import count_pairs



class BpeTokenizer(ByteTokenizer):
    """
    Класс для токенизации текста с использованием байтового представления и BPE (Byte Pair Encoding) алгоритма.

    Этот класс является наследником ByteTokenizer и расширяет его функционал, реализуя механизм
    склеивания наиболее часто встречающихся пар байт в новые токены, что позволяет эффективно
    сжимать словарь и улучшать обработку текстов.

    Атрибуты:
    ----------
    merges : dict
        Словарь, в котором хранятся пары токенов и их новые индексы после склеивания (BPE).

    Методы:
    -------
    init_vocab() -> None
        Переинициализирует словарь, добавляя таблицу склеиваний BPE.
    train(texts: List[str], max_vocab: int) -> None
        Тренирует BPE-токенизатор, находя наиболее частотные пары токенов и склеивая их,
        пока не будет достигнут заданный размер словаря.
    encode(text: str) -> List[int]
        Преобразует строку в список байтов с применением BPE для наиболее частотных пар токенов.
    """
    def __init__(self):
        """
        Инициализирует BpeTokenizer, добавляя словарь для хранения склеиваний пар токенов (merges).
        """
        self.merges = {}
        super().__init__()

    def init_vocab(self) -> None:
        """
        Инициализирует словарь для токенизации и обнуляет таблицу склеиваний пар токенов.

        Вызывает родительский метод для создания исходного словаря с байтами и специальными токенами.
        """
        super().init_vocab()
        self.merges = {}

    def train(self, texts: List[str], max_vocab: int) -> None:
        """
        Тренирует BPE-токенизатор на предоставленных текстах, последовательно склеивая
        наиболее частотные пары токенов до достижения заданного размера словаря.

        Параметры:
        ----------
        texts : List[str]
            Список текстов для тренировки токенизатора.
        max_vocab : int
            Максимальный размер словаря, после достижения которого процесс тренировки остановится.

        Возвращает:
        -----------
        None
        """
        self.init_vocab()

        if max_vocab <= len(self.vocab):
            return
        progress_bar = tqdm(range(max_vocab - len(self.vocab)))

        # Формируем исходный список номеров токенов для каждого текста (изначально это байты в кодировке utf-8)
        list_of_ids = [list(text.encode('utf-8')) for text in texts]
        print(list_of_ids)
        for _ in progress_bar:
            # Находим наиболее частотную пару токенов для склеивания в один токен
            cnt = count_pairs(list_of_ids)
            pair = max(cnt, key=cnt.get)
            freq = cnt[pair]
            progress_bar.set_description(f'pair={pair}, freq={freq}')

            if freq == 1:
                break

            new_idx = len(self.vocab)
            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            for i, ids in enumerate(list_of_ids):
                list_of_ids[i] = merge(ids, pair, new_idx)

    def encode(self, text: str) -> List[int]:
        """
        Преобразует строку в последовательность идентификаторов с применением BPE.

        Параметры:
        ----------
        text : str
            Входная строка для токенизации.

        Возвращает:
        -----------
        List[int]
            Список идентификаторов с учётом частотных пар токенов, объединённых алгоритмом BPE.
        """
        ids = list(text.encode('utf-8'))
        while len(ids) > 1:
            cnt = count_pairs([ids])
            pair = max(cnt, key=cnt.get)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
