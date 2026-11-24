# When SaCo Changes Its Mind

Code and results for the paper "When SaCo Changes Its Mind: A Large-Scale Re-Evaluation of Faithfulness Verdicts" by Lauria Rubega and Sibylle Hess (Eindhoven University of Technology).

The paper tests whether the SaCo faithfulness metric (Wu et al., CVPR 2024) gives the same verdict when its perturbation operator changes. It compares mean replacement with four alternatives: blur, noisy fill, pixel shuffling and inpainting. The grid covers three Vision Transformers (ViT-B/16, ViT-L/16, DeiT-B/16), four datasets (Imagenette, ImageNet-1k, CIFAR-10, CIFAR-100) and four explanation methods plus a random baseline.

## Contents

| Folder | What it holds |
|---|---|
| `src/saco/` | Models, data loading, perturbations, explanation methods and the SaCo metric. |
| `src/` | Entry points. `run_experiment.py` runs one architecture and dataset. `analyze_results.py` computes margins and tests. `multiple_comparisons.py` applies the Benjamini-Hochberg and Holm corrections. The `debug_*.py` scripts are the attention diagnostics from the paper. |
| `src/configs/` | Which checkpoint is used for each CIFAR combination. |
| `jobs/` | SLURM scripts used to run the grid. |
| `environment/` | Conda environment. |
| `results/` | Per-image SaCo scores for all 12 combinations, five methods and five operators. Integrated Gradients uses 200 steps. |
| `analysis/` | Summary statistics per combination, the corrected p-values, the robustness pilot summaries and the diagnostic outputs. |

## Setup

```
conda env create -f environment/environment.yml -n saco
conda activate saco
```

The results in the paper were produced with PyTorch 2.13.0, torchvision 0.28.0, timm 1.0.28, transformers 5.14.1, SciPy 1.17.1 and OpenCV 5.0.0.

On a cluster, check that PyTorch sees the GPU after the environment is built. Conda can silently install a CPU-only build. If that happens, reinstall with `pip install --force-reinstall "torch>=2.6" torchvision`. The comments in `environment/environment.yml` explain this.

## Running

Prepare the data, then run one combination:

```
python src/prepare_data.py --datasets imagenette cifar10 cifar100 --data-root ./data
python src/run_experiment.py --arch vit_b16 --dataset imagenette --num-images 1000 --ig-steps 200
```

Use `--ig-steps 200` to match the paper. The default is 20, which does not converge well (see the paper's implementation details). ImageNet-1k needs a local copy of the validation set (`jobs/prepare_imagenet.job`).

Three CIFAR checkpoints were fine-tuned by us. They are not included because of their size. Recreate them with:

```
python src/finetune_cifar_checkpoint.py --arch vit_l16 --dataset cifar10
python src/finetune_cifar_checkpoint.py --arch deit_b16 --dataset cifar10
python src/finetune_cifar_checkpoint.py --arch deit_b16 --dataset cifar100
```

To run the whole grid with SLURM, set your account in `jobs/*.job` (replace `YOUR_SLURM_ACCOUNT`) and run `bash jobs/submit_matrix.sh` from the repository root.

## Statistics

```
python src/analyze_results.py --results results/vit_b16_imagenette/results.json --output-dir analysis
cd src && python add_hodges_lehmann.py && python multiple_comparisons.py
```

Each test compares a method with Random Attribution on the same images. The test is a two-sided Wilcoxon signed-rank test on the paired margins. All 240 tests are corrected together with Benjamini-Hochberg. Holm is used as a robustness check.

## Licence

MIT, see `LICENSE`.

## Citation

If you use this code, please cite the paper. The full reference will be added here after publication.
