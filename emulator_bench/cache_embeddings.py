import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    ensure_parent,
    ligand_cache_path,
    normalize_threshold_args,
    normalize_sequence,
    protein_cache_path,
    read_table,
    save_json,
)
from emulator_bench.feature_pipeline import (
    build_prot_t5_batches,
    embed_prot_t5_batch,
    ligand_cache_items,
    load_esm_model,
    load_prot_t5,
    load_smiles_transformer,
    protein_cache_item,
)


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected columns `{sequence_col}` and `{smiles_col}` in {job[split_key]}")
            sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
            smiles_values.update(str(value) for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def _save_npz(path: Path, item: dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp_path.replace(path)


def cache_proteins(args, sequences):
    pending = [seq for seq in sequences if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()]
    if not pending:
        print("Protein cache is already complete.")
        return {"proteins_total": len(sequences), "proteins_written": 0}

    device = torch.device(args.device)
    print(f"Protein cache device: {device}")
    prot_t5_model, prot_t5_tokenizer = load_prot_t5(args.prot_t5_model, device)
    esm_model, batch_converter = load_esm_model(device)

    written = 0
    prot_t5_batches = build_prot_t5_batches(
        pending,
        max_residues=args.prot_t5_max_residues,
        max_seq_len=args.prot_t5_max_seq_len,
        max_batch=args.prot_t5_max_batch,
    )
    batch_iter = tqdm(prot_t5_batches, desc="Caching protein embeddings", unit="batch")
    for batch in batch_iter:
        prot5_by_sequence = embed_prot_t5_batch(prot_t5_model, prot_t5_tokenizer, batch, device)
        for sequence in batch:
            cache_item = protein_cache_item(
                sequence=sequence,
                prot5_array=prot5_by_sequence[sequence],
                esm_model=esm_model,
                batch_converter=batch_converter,
                device=device,
                protein_dtype=args.protein_dtype,
            )
            _save_npz(protein_cache_path(args.embeddings_dir, sequence), cache_item)
            written += 1
            batch_iter.set_postfix(written=written, remaining=len(pending) - written)

    return {"proteins_total": len(sequences), "proteins_written": written}


def cache_ligands(args, smiles_values):
    pending = [smiles for smiles in smiles_values if args.overwrite or not ligand_cache_path(args.embeddings_dir, smiles).exists()]
    if not pending:
        print("Ligand cache is already complete.")
        return {"ligands_total": len(smiles_values), "ligands_written": 0}

    device = torch.device(args.device)
    print(f"Ligand cache device: {device}")
    trfm_model = load_smiles_transformer(device=device)
    written = 0
    batch_iter = tqdm(range(0, len(pending), args.ligand_batch_size), desc="Caching ligand embeddings", unit="batch")
    for start in batch_iter:
        batch = pending[start : start + args.ligand_batch_size]
        items = ligand_cache_items(batch, trfm_model)
        for smiles, item in items.items():
            _save_npz(ligand_cache_path(args.embeddings_dir, smiles), item)
            written += 1
        batch_iter.set_postfix(written=written, remaining=len(pending) - written)

    return {"ligands_total": len(smiles_values), "ligands_written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache reusable KcatNet protein and ligand embeddings for emulator bench runs.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_seq_len", type=int, default=1000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=32)
    parser.add_argument("--ligand_batch_size", type=int, default=256)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    started = time.time()
    sequences, smiles_values = _collect_unique_values(jobs, sequence_col=args.sequence_col, smiles_col=args.smiles_col)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique normalized sequences: {len(sequences)}")
    print(f"Unique smiles: {len(smiles_values)}")

    protein_stats = cache_proteins(args, sequences)
    ligand_stats = cache_ligands(args, smiles_values)

    manifest = {
        "cache_version": 1,
        "base_dir": str(args.base_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "split_groups": list(args.split_groups),
        "thresholds": args.thresholds,
        "protein_dtype": args.protein_dtype,
        "prot_t5_model": args.prot_t5_model,
        "protein_max_len": 1000,
        "ligand_smiles_encoder": "smiles_transformer_trfm_12_23000",
        "protein_model": "esm2_t33_650M_UR50D + ProtT5",
        "protein_cache": protein_stats,
        "ligand_cache": ligand_stats,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)

    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")
    print(
        f"Protein cache: wrote {protein_stats['proteins_written']} / {protein_stats['proteins_total']} | "
        f"Ligand cache: wrote {ligand_stats['ligands_written']} / {ligand_stats['ligands_total']}"
    )


if __name__ == "__main__":
    main()
