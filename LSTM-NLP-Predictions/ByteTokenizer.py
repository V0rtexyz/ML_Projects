from typing import List, Tuple, Dict


class ByteTokenizer:
    """
    Класс для токенизации текста с использованием байтового представления символов.

    Этот класс реализует базовый байтовый токенизатор, который преобразует строки
    в последовательности байтов и обратно. Также поддерживаются специальные токены,
    такие как <pad>, <bos> (начало последовательности) и <eos> (конец последовательности).

    Атрибуты:
    ----------
    pad_token : bytes
        Токен для заполнения (padding) последовательности, по умолчанию b'<pad>'.
    bos_token : bytes
        Токен начала последовательности, по умолчанию b'<bos>'.
    eos_token : bytes
        Токен конца последовательности, по умолчанию b'<eos>'.
    pad_token_id : int или None
        Идентификатор токена заполнения, назначается при инициализации словаря.
    bos_token_id : int или None
        Идентификатор токена начала последовательности, назначается при инициализации словаря.
    eos_token_id : int или None
        Идентификатор токена конца последовательности, назначается при инициализации словаря.
    special_tokens : List[bytes]
        Список специальных токенов, включающий pad_token, bos_token и eos_token.
    vocab : dict
        Словарь для сопоставления индексов и байтовых значений символов.

    Методы:
    -------
    init_vocab() -> None
        Инициализирует словарь (vocab), где ключами являются индексы, а значениями — байтовые представления символов.
    train(texts: List[str], max_vocab: int) -> None
        Переинициализирует словарь (переопределяется в потомках).
    encode(text: str) -> List[int]
        Преобразует строку в список идентификаторов (байтов) с использованием кодировки UTF-8.
    decode(ids: List[int]) -> str
        Преобразует список идентификаторов (байтов) обратно в строку.
    get_vocab_size() -> int
        Возвращает размер словаря (количество уникальных символов и токенов).
    """
    def __init__(self):
        self.pad_token = b'<pad>'
        self.bos_token = b'<bos>'
        self.eos_token = b'<eos>'
        self.pad_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token]
        self.vocab = {}
        self.init_vocab()

    def init_vocab(self) -> None:
        """Инициализирует словарь для токенизации, добавляя байтовые представления символов и специальные токены."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for token in self.special_tokens:
            idx = len(self.vocab)
            self.vocab[idx] = token
        token_to_id = {y: x for x, y in self.vocab.items()}
        self.pad_token_id = token_to_id[self.pad_token]
        self.bos_token_id = token_to_id[self.bos_token]
        self.eos_token_id = token_to_id[self.eos_token]

    def train(self, texts: List[str], max_vocab: int) -> None:
        """Тренирует токенизатор на данных текстах, переинициализируя словарь (пока без дополнительной логики)."""
        self.init_vocab()

    def encode(self, text: str) -> List[int]:
        """Преобразует строку в список байтов с использованием кодировки UTF-8.

        Параметры:
        ----------
        text : str
            Входная строка для преобразования.

        Возвращает:
        -----------
        List[int]
            Список идентификаторов (байтов), представляющих символы строки.
        """
        return list(text.encode('utf-8'))

    def decode(self, ids: List[int]) -> str:
        """Преобразует список идентификаторов обратно в строку.

        Параметры:
        ----------
        ids : List[int]
            Список байтовых идентификаторов.

        Возвращает:
        -----------
        str
            Декодированная строка.
        """
        text = b''.join(self.vocab[idx] for idx in ids).decode('utf-8', errors='replace')
        return text

    def get_vocab_size(self) -> int:
        """Возвращает количество уникальных символов и токенов в словаре.

        Возвращает:
        -----------
        int
            Размер словаря.
        """
        return len(self.vocab)


def count_pairs(data: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """
    Считает, сколько раз встречается каждая пара последовательных элементов (стоящих на соседних позициях) во всех списках чисел.

    Параметры:
    ----------
    data: List[List[int]]
        Список, содержащий списки целых чисел.

    Возвращает:
    -----------
        Dict[Tuple[int, int], int]
            Словарь, где ключами являются пары элементов (кортежи), а значениями - количество их появлений в списках.
    """
    Dict = {}
    for i in data:
        for j in range(len(i)-1):
            if tuple([i[j], i[j+1]]) in Dict:
                Dict[(i[j], i[j+1])] += 1
            else:
                Dict[(i[j], i[j+1])] = 1
    return Dict


def merge(numbers: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    Двигаясь слева направо, заменяет все вхождения заданной пары чисел в массиве на заданный индекс.
    Гарантируется, что заданный индекс не встречается в массиве чисел.

    Параметры:
    ----------
    numbers : List[int]
        Список целых чисел.
    pair : Tuple[int, int]
        Пара целых чисел, которую необходимо найти и заменить.
    idx : int
        Значение, на которое заменяется найденная пара.

    Возвращает:
    -----------
    List[int]
        Новый список, где каждая найденная пара заменена на значение idx.
    """
    i = 0
    while i != len(numbers)-1:
        if (numbers[i],numbers[i+1]) == pair:
            numbers[i] = idx
            del numbers[i+1]
        i += 1
    return numbers
