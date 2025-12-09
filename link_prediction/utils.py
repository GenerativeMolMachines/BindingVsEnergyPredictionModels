from fastapi import HTTPException
import joblib
import warnings
import numpy as np
import json, h5py, torch
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
# warnings.filterwarnings("ignore")
#
# def get_s3_fs():
#     """Создает и возвращает файловую систему S3"""
#     return s3fs.S3FileSystem(
#         key=ACCESS_KEY,
#         secret=SECRET_KEY,
#         client_kwargs={'endpoint_url': "http://" + MINIO_ENDPOINT}
#     )
#
#
# def download_file_from_s3(s3_path: str, force_download: bool = False) -> Path:
#     """
#     Скачивает файл из S3 в локальную директорию data, если его там нет.
#     Сохраняет с оригинальным именем файла.
#
#     Args:
#         s3_path: Путь к файлу в S3 (например, 'bucket-name/path/to/file.h5')
#         force_download: Принудительно перезаписать файл, даже если он уже есть
#
#     Returns:
#         Path: Путь к локальному файлу
#     """
#     # Получаем оригинальное имя файла из пути S3
#     filename = Path(s3_path).name
#     local_path = CACHE_DIR / filename
#
#     # Если файл уже есть и force_download=False, возвращаем путь
#     if local_path.exists() and not force_download:
#         print(f"Файл уже скачан: {local_path}")
#         return local_path
#
#     # Скачиваем файл из S3
#     print(f"Скачиваю файл из S3: {s3_path} -> {local_path}")
#
#     fs = get_s3_fs()
#
#     try:
#         # Проверяем существование файла в S3
#         if not fs.exists(s3_path):
#             raise FileNotFoundError(f"Файл {s3_path} не найден в MinIO S3")
#
#         # Скачиваем файл
#         fs.get(s3_path, str(local_path))
#
#         # Проверяем, что файл скачался
#         if not local_path.exists():
#             raise RuntimeError(f"Не удалось скачать файл {s3_path}")
#
#         file_size = local_path.stat().st_size
#         print(f"Файл успешно скачан: {local_path} ({file_size / (1024 * 1024):.2f} MB)")
#
#         return local_path
#
#     except Exception as e:
#         # Удаляем частично скачанный файл при ошибке
#         if local_path.exists():
#             local_path.unlink()
#         raise HTTPException(
#             status_code=500,
#             detail=f"Ошибка загрузки файла {s3_path} из S3: {str(e)}"
#         )
#
#
# def load_all_emb():
#     """Загружает эмбеддинги из H5 файла с кэшированием"""
#     try:
#         # Получаем оригинальное имя файла из S3 пути
#         filename = Path(EMB_H5).name
#         local_emb_path = CACHE_DIR / filename
#
#         # Если файла нет локально, скачиваем его
#         if not local_emb_path.exists():
#             local_emb_path = download_file_from_s3(EMB_H5)
#
#         # Читаем из локального файла
#         with h5py.File(local_emb_path, "r") as hf:
#             if "embeddings" not in hf:
#                 raise HTTPException(status_code=500, detail="Ключ 'embeddings' не найден в H5 файле")
#
#             all_emb = torch.from_numpy(hf["embeddings"][...]).float()
#
#         print(f"Эмбеддинги загружены успешно. Размер: {all_emb.shape}")
#         return all_emb
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка загрузки эмбеддингов: {str(e)}")
#
#
# def load_entity_map():
#     fs = get_s3_fs()
#     try:
#         if not fs.exists(ENTITY_JSON):
#             raise RuntimeError(status_code=404, detail=f"Файл {ENTITY_JSON} не найден в MinIO")
#
#         with fs.open(ENTITY_JSON, 'rt', encoding='utf-8') as f:
#             entity_names = json.load(f)
#         return entity_names
#     except json.JSONDecodeError as e:
#         raise RuntimeError(status_code=500, detail=f"Ошибка парсинга JSON: {str(e)}")
#     except Exception as e:
#         raise RuntimeError(status_code=500, detail=f"Ошибка загрузки entity map: {str(e)}")
#
# def get_comparator(name: str):
#     return {
#         "cos": CosComparator(),
#         "dot": DotComparator(),
#         "l2": L2Comparator(),
#         "squared_l2": SquaredL2Comparator(),
#     }[name]
#
def load_linear_w_from_h5() -> torch.Tensor:
    # fs = get_s3_fs()
    diagonal_path = f"model/relations/{RELATION_IDX}/operator/rhs/diagonal"
    # with fs.open(DIM, 'rb') as f:
    #     file_content = f.read()
    # # Создаем in-memory файловый объект для h5py
    # file_obj = io.BytesIO(file_content)

    with h5py.File(MODEL_H5, "r") as hf:
        if diagonal_path in hf:
            diagonal = torch.from_numpy(hf[diagonal_path][...]).float()
            if diagonal.shape == (DIM,):
                return diagonal  # Возвращаем как вектор для diagonal оператора

        # Если не diagonal, пробуем linear оператор (матрица [D, D])
        candidates = [
            f"model/relations/{RELATION_IDX}/operator/rhs/weight",
            f"model/relations/{RELATION_IDX}/operator/weight",
            f"model/relations/{RELATION_IDX}/operator/matrix",
            f"model/relations/{RELATION_IDX}/weight",
            "model/operator/weight",
        ]
        for key in candidates:
            if key in hf:
                W = torch.from_numpy(hf[key][...]).float()
                if tuple(W.shape) == (DIM, DIM):
                    return W

        # ищем любой [DIM,DIM] с operator
        target = None

        def visit(name, obj):
            nonlocal target
            if target is None and isinstance(obj, h5py.Dataset):
                if obj.shape == (DIM, DIM) and "operator" in name.lower():
                    target = torch.from_numpy(obj[...]).float()

        hf.visititems(visit)
        if target is not None:
            return target

    raise RuntimeError(f"Не найден оператор в H5 (RELATION_IDX={RELATION_IDX})")
#
