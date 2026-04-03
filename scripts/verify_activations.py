#!/usr/bin/env python3
"""Verify phoneme-aligned activations are meaningful.

Runs diagnostic checks on captured activations:
1. Sanity checks: counts, norms, degenerate vectors
2. Phoneme discriminability: cosine similarity between centroids
3. Intra/inter-class variance ratio (silhouette-like)
4. PCA visualization of phoneme centroids

Usage:
    python scripts/verify_activations.py --input data/activations_aligned/all
    python scripts/verify_activations.py --input data/activations_aligned/all --layer 4 --plot
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_phoneme_activations(input_dir: Path, layer_idx: int):
    """Load all phoneme .npy files for a given layer.

    Returns dict: phoneme -> np.ndarray of shape (n_frames, d_model)
    """
    layer_dir = input_dir / f"layer_{layer_idx:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

    activations = {}
    for npy_file in sorted(layer_dir.glob("phoneme_*.npy")):
        phoneme = npy_file.stem.replace("phoneme_", "")
        acts = np.load(npy_file)
        activations[phoneme] = acts.astype(np.float32)

    return activations


def check_sanity(activations: dict, inventory: dict):
    """Basic sanity checks on activations."""
    logger.info("=" * 60)
    logger.info("1. SANITY CHECKS")
    logger.info("=" * 60)

    total_frames = sum(a.shape[0] for a in activations.values())
    logger.info(f"Phonemes loaded: {len(activations)}")
    logger.info(f"Total frames: {total_frames}")

    if not activations:
        logger.error("No activations loaded!")
        return False

    # Check dimensions are consistent
    dims = set(a.shape[1] for a in activations.values())
    if len(dims) != 1:
        logger.error(f"Inconsistent d_model across phonemes: {dims}")
        return False
    d_model = dims.pop()
    logger.info(f"d_model: {d_model}")

    # Check for degenerate (all-zero) vectors
    zero_phonemes = []
    for ph, acts in activations.items():
        if np.allclose(acts, 0):
            zero_phonemes.append(ph)
    if zero_phonemes:
        logger.warning(f"All-zero activations for: {zero_phonemes}")
    else:
        logger.info("No degenerate (all-zero) phonemes")

    # Check count distribution
    counts = {ph: a.shape[0] for ph, a in activations.items()}
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    logger.info("\nTop 10 phonemes by frame count:")
    for ph, c in sorted_counts[:10]:
        pct = 100 * c / total_frames
        logger.info(f"  {ph:>6s}: {c:>6d} ({pct:.1f}%)")

    logger.info("\nBottom 5 phonemes by frame count:")
    for ph, c in sorted_counts[-5:]:
        logger.info(f"  {ph:>6s}: {c:>6d}")

    # Check if one phoneme dominates
    top_pct = sorted_counts[0][1] / total_frames
    if top_pct > 0.5:
        logger.warning(f"Top phoneme '{sorted_counts[0][0]}' has {top_pct:.0%} of all frames — suspicious")
    else:
        logger.info(f"Top phoneme accounts for {top_pct:.1%} of frames (healthy)")

    # Activation norm statistics
    all_norms = []
    for ph, acts in activations.items():
        norms = np.linalg.norm(acts, axis=1)
        all_norms.append((ph, norms.mean(), norms.std()))

    norms_array = np.array([n[1] for n in all_norms])
    logger.info(f"\nActivation L2 norms: mean={norms_array.mean():.2f}, std={norms_array.std():.2f}")
    logger.info(f"  range: [{norms_array.min():.2f}, {norms_array.max():.2f}]")

    return True


def check_discriminability(activations: dict, min_count: int = 10):
    """Check if different phonemes have distinguishable activation patterns."""
    logger.info("\n" + "=" * 60)
    logger.info("2. PHONEME DISCRIMINABILITY (cosine similarity between centroids)")
    logger.info("=" * 60)

    # Compute centroids for phonemes with enough samples
    centroids = {}
    for ph, acts in activations.items():
        if acts.shape[0] >= min_count:
            centroids[ph] = acts.mean(axis=0)

    logger.info(f"Phonemes with >= {min_count} frames: {len(centroids)}")

    if len(centroids) < 2:
        logger.warning("Not enough phonemes for discriminability analysis")
        return None

    # Pairwise cosine similarity
    phonemes = sorted(centroids.keys())
    n = len(phonemes)
    centroid_matrix = np.stack([centroids[ph] for ph in phonemes])

    # Normalize
    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    centroid_normed = centroid_matrix / norms

    sim_matrix = centroid_normed @ centroid_normed.T

    # Extract upper triangle (exclude diagonal)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    logger.info(f"\nPairwise cosine similarity (between centroids):")
    logger.info(f"  mean: {upper_tri.mean():.4f}")
    logger.info(f"  std:  {upper_tri.std():.4f}")
    logger.info(f"  min:  {upper_tri.min():.4f}")
    logger.info(f"  max:  {upper_tri.max():.4f}")

    # If mean similarity is very high (>0.95), activations are not discriminative
    if upper_tri.mean() > 0.95:
        logger.warning("High mean similarity — activations may not discriminate phonemes well")
    elif upper_tri.mean() > 0.85:
        logger.info("Moderate similarity — some discriminability present")
    else:
        logger.info("Good discriminability — centroids are spread out")

    # Show most similar and most different pairs
    logger.info("\nMost similar pairs (potential confusion):")
    pair_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sims.append((phonemes[i], phonemes[j], sim_matrix[i, j]))
    pair_sims.sort(key=lambda x: -x[2])

    for ph1, ph2, sim in pair_sims[:5]:
        logger.info(f"  {ph1:>6s} ↔ {ph2:<6s}: {sim:.4f}")

    logger.info("\nMost different pairs:")
    for ph1, ph2, sim in pair_sims[-5:]:
        logger.info(f"  {ph1:>6s} ↔ {ph2:<6s}: {sim:.4f}")

    return sim_matrix, phonemes


def check_cluster_quality(activations: dict, min_count: int = 10, sample_size: int = 200):
    """Compute intra-class vs inter-class distance ratio."""
    logger.info("\n" + "=" * 60)
    logger.info("3. CLUSTER QUALITY (intra/inter-class distance)")
    logger.info("=" * 60)

    # Filter phonemes with enough samples
    filtered = {ph: acts for ph, acts in activations.items() if acts.shape[0] >= min_count}
    if len(filtered) < 2:
        logger.warning("Not enough phonemes for cluster analysis")
        return

    # Compute centroids
    centroids = {ph: acts.mean(axis=0) for ph, acts in filtered.items()}
    global_centroid = np.mean(list(centroids.values()), axis=0)

    # Intra-class: average distance of frames to their own centroid
    intra_distances = []
    for ph, acts in filtered.items():
        # Subsample if too many frames
        if acts.shape[0] > sample_size:
            idx = np.random.choice(acts.shape[0], sample_size, replace=False)
            acts_sub = acts[idx]
        else:
            acts_sub = acts
        dists = np.linalg.norm(acts_sub - centroids[ph], axis=1)
        intra_distances.append(dists.mean())

    # Inter-class: average distance between centroids
    centroid_list = list(centroids.values())
    inter_distances = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_distances.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))

    mean_intra = np.mean(intra_distances)
    mean_inter = np.mean(inter_distances)
    ratio = mean_inter / mean_intra if mean_intra > 0 else float("inf")

    logger.info(f"Mean intra-class distance: {mean_intra:.4f}")
    logger.info(f"Mean inter-class distance: {mean_inter:.4f}")
    logger.info(f"Ratio (inter/intra):       {ratio:.4f}")

    if ratio > 2.0:
        logger.info("Excellent separation — clusters are well-defined")
    elif ratio > 1.0:
        logger.info("Reasonable separation — clusters overlap partially")
    else:
        logger.warning("Poor separation — clusters highly overlapping")


def plot_pca(activations: dict, min_count: int = 10, output_path: Path = None):
    """PCA visualization of phoneme centroids."""
    logger.info("\n" + "=" * 60)
    logger.info("4. PCA VISUALIZATION")
    logger.info("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib/sklearn not available — skipping PCA plot")
        return

    # Compute centroids
    centroids = {}
    for ph, acts in activations.items():
        if acts.shape[0] >= min_count:
            centroids[ph] = acts.mean(axis=0)

    if len(centroids) < 3:
        logger.warning("Need at least 3 phonemes for PCA")
        return

    phonemes = sorted(centroids.keys())
    matrix = np.stack([centroids[ph] for ph in phonemes])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    # Simple phoneme type classification for coloring
    vowels = set("aeiouæɛɪɔʊʌəɑɐɒœøyː")
    colors = []
    for ph in phonemes:
        if any(c in vowels for c in ph.lower()):
            colors.append("red")
        else:
            colors.append("blue")

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, ph in enumerate(phonemes):
        ax.scatter(coords[i, 0], coords[i, 1], c=colors[i], s=50, alpha=0.7)
        ax.annotate(ph, (coords[i, 0], coords[i, 1]), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of Phoneme Centroids (red=vowel-like, blue=consonant-like)")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = Path("phoneme_pca.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"PCA plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify phoneme-aligned activations")
    parser.add_argument("--input", type=Path, required=True, help="Activations directory")
    parser.add_argument("--layer", type=int, default=4, help="Layer index to analyze (default: 4)")
    parser.add_argument("--min-count", type=int, default=10, help="Min frames per phoneme")
    parser.add_argument("--plot", action="store_true", help="Generate PCA plot")
    parser.add_argument("--all-layers", action="store_true", help="Run checks on all layers")
    args = parser.parse_args()

    # Load inventory
    inventory_path = args.input / "phoneme_inventory.json"
    inventory = {}
    if inventory_path.exists():
        with open(inventory_path) as f:
            inventory = json.load(f)
        logger.info(f"Language: {inventory.get('language', 'unknown')}")
        logger.info(f"Total phonemes: {len(inventory.get('phoneme_counts', {}))}")
        logger.info(f"Total frames: {inventory.get('total_frames', 0)}")

    # Determine layers to check
    if args.all_layers:
        layer_dirs = sorted(args.input.glob("layer_*"))
        layers = [int(d.name.split("_")[1]) for d in layer_dirs]
    else:
        layers = [args.layer]

    for layer_idx in layers:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# LAYER {layer_idx}")
        logger.info(f"{'#' * 60}")

        try:
            activations = load_phoneme_activations(args.input, layer_idx)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        ok = check_sanity(activations, inventory)
        if not ok:
            continue

        check_discriminability(activations, min_count=args.min_count)
        check_cluster_quality(activations, min_count=args.min_count)

        if args.plot:
            plot_path = args.input / f"pca_layer_{layer_idx:02d}.png"
            plot_pca(activations, min_count=args.min_count, output_path=plot_path)


if __name__ == "__main__":
    main()
