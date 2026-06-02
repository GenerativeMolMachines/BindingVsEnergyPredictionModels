import os

from dotenv import load_dotenv

load_dotenv()

ENTITY_JSON = os.getenv("ENTITY_JSON")
# ENTITY_JSON = '/mnt/tank/scratch/pbogdanov/BiopolymersKG/graph_link_prediction_pbg/data/partitions/entity_names_molecules_0.json'
EMB_H5 = os.getenv("EMB_H5")
# EMB_H5 = '/mnt/tank/scratch/pbogdanov/BiopolymersKG/graph_link_prediction_pbg/results_translation/l2_translation_checkpoint/epoch_22/embeddings_molecules_0.v352.h5'
MODEL_H5 = os.getenv("MODEL_H5")
# MODEL_H5 = '/mnt/tank/scratch/pbogdanov/BiopolymersKG/graph_link_prediction_pbg/results_translation/l2_translation_checkpoint/epoch_22/model.v352.h5'


DIM = int(os.getenv("DIM", 400))
TOPK = int(os.getenv("TOPK", 100))
COMPARATOR_TYPE = os.getenv("COMPARATOR_TYPE", "l2")
RELATION_IDX = int(os.getenv("RELATION_IDX", 0))
