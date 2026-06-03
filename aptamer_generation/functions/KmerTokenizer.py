import torch
from typing import List
import itertools

class KMerTokenizer:
    def __init__(self, k: int = 5, nucleotides: str = "ATCG"):
        self.k = k
        self.nucleotides = nucleotides

        # Строим основной словарь k-меров
        self.vocab = self._build_vocab()

        # Добавляем специальные токены в начало
        self.special_tokens = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2}
        self.vocab = list(self.special_tokens.keys()) + self.vocab

        # Создаем отображения
        self.token_to_id_map = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token_map = {idx: token for idx, token in enumerate(self.vocab)}

        self.pad_id = self.token_to_id_map['[PAD]']
        self.sos_id = self.token_to_id_map['[SOS]']
        self.eos_id = self.token_to_id_map['[EOS]']

    def _build_vocab(self) -> List[str]:
        return [''.join(kmer) for kmer in itertools.product(self.nucleotides, repeat=self.k)]

    def token_to_id(self, token: str) -> int:
        """Возвращает ID для одного токена"""
        if token not in self.token_to_id_map:
            raise ValueError(f"Token '{token}' not in vocabulary")
##        print(f'Input token in token_to_id {token}')
##        print(self.token_to_id_map)
        return self.token_to_id_map[token]

    def id_to_token(self, token_id: int) -> str:
        """Возвращает токен по ID"""
        return self.id_to_token_map[token_id]

    def tokenize(self, sequence: str) -> List[str]:
        sequence = self.pad_sequence(sequence)
        step = self.k
        kmers = []
        for i in range(0, len(sequence) - self.k + 1, step):
            kmer = sequence[i:i+self.k]
            if all(c in self.nucleotides or c == '[PAD]' for c in kmer):
                kmers.append(kmer)
        return kmers

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a tensor of token IDs back to a sequence string"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()

        # Convert to integers and remove padding
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_id:  # Stop at EOS token
                break
            if token_id not in [self.pad_id, self.sos_id]:  # Skip PAD and SOS
                tokens.append(self.id_to_token_map[token_id])

        # Join k-mers back into sequence
        return ''.join(tokens).replace('[PAD]', '')

    def pad_sequence(self, sequence: str) -> str:
        pad_length = (self.k - len(sequence) % self.k) % self.k
        return sequence + '[PAD]' * pad_length

    def __len__(self) -> int:
        return len(self.vocab)


