import h5py
import json
import torch
import warnings

from torchbiggraph.model import DotComparator, CosComparator, L2Comparator, SquaredL2Comparator

from config import ENTITY_JSON, EMB_H5, MODEL_H5, DIM, RELATION_IDX

warnings.filterwarnings("ignore")


def load_all_emb():
    with h5py.File(EMB_H5, "r") as hf:
        all_emb = torch.from_numpy(hf["embeddings"][...]).float()
    return all_emb


def load_entity_map():
    with open(ENTITY_JSON, "rt") as tf:
        entity_names = json.load(tf)
    return entity_names


def get_comparator(name: str):
    return {
        "cos": CosComparator(),
        "dot": DotComparator(),
        "l2": L2Comparator(),
        "squared_l2": SquaredL2Comparator(),
    }[name]


def load_translation_from_h5() -> torch.Tensor:
    translation_path = f"model/relations/{RELATION_IDX}/operator/rhs/translation"

    with h5py.File(MODEL_H5, "r") as hf:
        if translation_path in hf:
            translation = torch.from_numpy(hf[translation_path][...]).float()
            if translation.shape == (DIM,):
                return translation

    raise RuntimeError(f"Не найден translation operator в H5 (RELATION_IDX={RELATION_IDX})")
