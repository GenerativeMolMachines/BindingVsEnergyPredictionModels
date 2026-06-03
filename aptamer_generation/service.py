# Python version: 3.10.12


import numpy as np
from pathlib import Path

import torch

from functions.gen_modes import greedy_decode, beam_search_decode, beam_search_decode_2, top_k_decode, top_p_decode, top_p_decode_2
from functions.Transformer_model import build_transformer
from functions.KmerTokenizer import KMerTokenizer


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "Transformer_model.pt"


def get_config():
    # Default configuration
    config = {
        'experiment_name': 'Transformer_new_try',
        'model_basename': 'Transformer_new_try_beam_2',
        'batch_size': 32,
        'num_epochs': 30,
        'save_every': 10,
        'lr': 1e-4,
        'lr_gamma': 0.95,
        "weight_decay": 1.67e-05,
        't_max': 50,
        'eta_min': 1e-5,
        'seq_len': 80,
        'num_heads': 8,
        'num_layers': 1,
        'd_model': 512,
        'd_ff': 256,
        'input_dim':2048,
        'save_100_steps': False,
        'save_first_epoch': True,
        'model_folder': 'checkpoints',
        'preload': "-",
        'attention_log_interval': 50,
        'validation_log_interval': 1,
        'gen_mode': 'beam_2',
        'kmer': 3,
        'comments': '-'
    }
    
    return config 
    

def predict(embedding, gen_mode="beam_2", max_len=80):
    """
    Generate an aptamer sequence from a precomputed joint protein-antibody embedding.

    Parameters
    ----------
    embedding : numpy.ndarray or torch.Tensor of shape (2048,)
        Precomputed embedding representing a protein-antibody pair.
        The embedding is obtained by concatenating embeddings generated
        by ProtBERT (protein sequence) and IgT5 (antibody sequence),
        resulting in a 2048-dimensional feature vector.

    gen_mode : str, default="beam_2"
        Decoding strategy used for sequence generation.

        Supported values:
        - "beam"
        - "beam_2"
        - "greedy"
        - "top_k"
        - "top_p"
        - "top_p_2"

    max_len : int, default=80
        Maximum length of the generated aptamer sequence in tokens.

    Returns
    -------
    str
        Generated aptamer sequence.

    """

    num_heads = model.config.num_heads if hasattr(model, 'config') else 8


    with torch.no_grad():
        encoder_input = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        encoder_mask = torch.ones((1, num_heads, 1, 1), dtype=torch.bool, device=device)

        decode_modes = {
            "beam": beam_search_decode,
            "beam_2": beam_search_decode_2,
            "greedy": greedy_decode,
            "top_k": top_k_decode,
            "top_p": top_p_decode,
            "top_p_2": top_p_decode_2
        }

        if gen_mode not in decode_modes:
            raise ValueError(f"Unknown gen_mode: {gen_mode}")

        model_out = decode_modes[gen_mode](
            model,
            encoder_input,
            encoder_mask,
            tokenizer_tgt,
            max_len,
            device
        )

        if model_out.dim() == 1:
            model_out = model_out.unsqueeze(0)
        sequence = tokenizer_tgt.decode(model_out[0].detach().cpu().numpy())

    return sequence





def get_model(config, vocab_tgt_len):
    """Создает модель трансформера с учетом нового формата входных данных"""
    model = build_transformer(
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        h = config['num_heads'],
        N=config['num_layers'],
        d_model=config['d_model'],
        d_ff = config['d_ff'],
        input_dim = config['input_dim']
    )
    return model



config = get_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer_tgt = KMerTokenizer(k=config["kmer"])


model = get_model(config, vocab_tgt_len=len(tokenizer_tgt)).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


if __name__ == "__main__":
    example_embedding = np.random.randn(
        80,
        2048
    ).astype(np.float32)

    result = predict(
        example_embedding
    )

    print(result)