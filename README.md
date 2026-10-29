# ImageLab AI — Image-to-Image Translation Studio

An end-to-end college project demonstrating image-to-image translation with a real pretrained **CycleGAN Horse → Zebra** generator. The Streamlit application provides image upload, actual PyTorch inference, downloadable results, model information, evaluation guidance, and a short theory reference. The reverse direction is enabled only when its own compatible checkpoint is present.

## What is implemented

- Streamlit studio with Home, Translate, Gallery, Model Details, Evaluation, and Theory sections.
- Official Berkeley CycleGAN 9-block ResNet generator architecture and strict state-dictionary loading.
- RGB preprocessing, 256 × 256 resize, normalization to `[-1, 1]`, output conversion, and original-size display.
- CPU inference by default, automatic CUDA selection when available, cached model loading, and `torch.inference_mode()`.
- Official pretrained Horse → Zebra and Zebra → Horse checkpoint download options (each about 43 MB).
- Optional unpaired CycleGAN training script with generator/discriminator losses, identity/cycle-consistency loss, replay pool, checkpoint saving, and CSV loss history.
- Dataset downloader for the Horse2Zebra unpaired training data (about 111 MB).
- Basic image preprocessing/output validation tests.

The demo does not train the pretrained model. No training result or model-quality score is claimed by this repository.

## Requirements

- Python 3.11 recommended (3.10+ should work with compatible PyTorch/torchvision wheels).
- CPU is supported. An NVIDIA CUDA installation is optional.
- Internet is needed once to download pretrained weights. Dataset download is only needed for training.

## Install and run (Windows PowerShell)

From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python download_model.py --task horse2zebra
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). If PowerShell blocks environment activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py` instead.

The first install command selects CPU-only PyTorch wheels. To use CUDA, install the PyTorch/torchvision build matching your installed CUDA version from the official [PyTorch install selector](https://pytorch.org/get-started/locally/) instead, then install the remaining packages from `requirements.txt`.

## Demo flow

1. Open **Translate** and upload a JPG/JPEG/PNG, or choose a Gallery sample.
2. Select **Horse → Zebra** and click **Translate Image**. If the checkpoint is missing, use the explicit download button.
3. Inspect the side-by-side input/output, inference time, and output dimensions; download the generated PNG.
4. Open **Model Details** to review the checkpoint source and **Evaluation** to inspect the result. Since this is unpaired translation, do not report MAE/PSNR without a spatially aligned target.

The included horse and zebra sample photos are small public-domain/CC0 photos from Wikimedia Commons, intended only as demo inputs. Their source and license are listed below. They are not the CycleGAN training set and are not model-generated outputs.

## Checkpoints and provenance

This project loads the public model-zoo weights published by the original CycleGAN project at UC Berkeley:

- Horse → Zebra: <https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth>
- Zebra → Horse: <https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/zebra2horse.pth>
- Upstream PyTorch architecture and code: <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
- Original CycleGAN paper: Zhu, Park, Isola, and Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017, <https://arxiv.org/abs/1703.10593>.

The upstream PyTorch repository uses the BSD 3-Clause license for code. The Berkeley checkpoint archive does not provide a separate model-card license next to each weight file; this project downloads the weights locally and does not commit or redistribute them. Review the current upstream terms before redistributing checkpoints or outputs.

Model settings expected by these checkpoints: RGB input, 256 × 256 test resize, values normalized to `[-1, 1]`, ResNet generator with 64 base channels and 9 residual blocks, instance normalization, and tanh output. The application verifies all checkpoint tensor keys and shapes before loading. Checkpoints live under `models/` and are excluded from Git.

## Optional: train CycleGAN locally

Download the official unpaired dataset only if you plan to train:

```powershell
python download_dataset.py
```

This creates `data/horse2zebra/{trainA,trainB,testA,testB}` from the public Horse2Zebra archive at <https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip>.

Start with a short smoke training run only if you have adequate compute; full training is expensive and typically needs a CUDA GPU:

```powershell
python train.py --data data/horse2zebra --epochs 200 --decay-epoch 100 --batch-size 1
```

The script saves generator state dictionaries and `losses.csv` under `outputs/checkpoints/`. To use locally trained weights in the app, copy the desired `horse2zebra.pth` (and, if available, `zebra2horse.pth`) into `models/`. If you change the architecture or preprocessing, the checkpoint must match it exactly. The training script uses separate unpaired A/B image sets, LSGAN losses, cycle weight 10, identity weight 5, Adam (`lr=0.0002`, `beta1=0.5`), and a 50-image replay pool.

Run one command-line inference with a downloaded checkpoint:

```powershell
python infer.py --task horse2zebra --input assets/sample_horse.jpg --output results/sample_translation.png
```

Training requires a long run; the included public pretrained checkpoint is the reproducible live-demo path. A partial training run is not presented as a completed experiment.

## Dataset and sample photos

- Horse2Zebra training data: unpaired horse and zebra photos, distributed through the original CycleGAN dataset endpoint. Dataset content is downloaded on demand and is not committed.
- Demo input, horse: “Horse in a pen.jpg” by Jojovavasasa, Wikimedia Commons, CC0 1.0: <https://commons.wikimedia.org/wiki/File:Horse_in_a_pen.jpg>.
- Demo input, zebra: “Zebra at the zoo.jpg” by Rivadavia.vila, Wikimedia Commons, CC0 1.0: <https://commons.wikimedia.org/wiki/File:Zebra_at_the_zoo.jpg>.

## Evaluation and limitations

- The app reports inference time and shows generated examples; users should review them qualitatively.
- On the included CC0 horse sample, this host measured 0.632 seconds for a CPU translation (preprocessing, generator pass, and output conversion; model load excluded). Timing depends on hardware and system load.
- MAE and PSNR are enabled only when the user supplies a spatially aligned paired target. They are not meaningful quality metrics for unpaired Horse → Zebra outputs.
- A low cycle-reconstruction error, if added in an experiment, would measure reconstruction consistency rather than target-domain realism.
- The model was trained on a particular collection of horse/zebra images. It can alter details, generate artifacts, or fail on unusual inputs and is not suitable for scientific, forensic, or identity-sensitive use.
- Visual examples do not establish general accuracy. For a formal evaluation, use held-out data, an explicitly defined protocol, and suitable distribution-level and human review metrics.
- Synthetic outputs should be labelled as generated. Check image and dataset rights for any redistribution or public deployment.

## Tests

```powershell
python -m pytest -q
```

## Troubleshooting

- **No checkpoint:** Run `python download_model.py --task "Horse → Zebra"`, or use the button in Translate. The download is checked by loading it strictly into the expected architecture.
- **CUDA out of memory:** The web demo uses a single 256 × 256 image. Select a CPU-only PyTorch build or close other GPU workloads if necessary.
- **Checkpoint key/shape mismatch:** Do not rename a Pix2Pix/CycleGAN checkpoint to this project's filename. Only weights matching the listed ResNet generator are compatible.
- **Torch/torchvision import error:** Install a matched pair through the official PyTorch selector, then rerun `pip install -r requirements.txt`.
- **Poor output:** CycleGAN works best on images with a clear animal subject and similar framing to its training examples; it does not detect or segment the horse automatically.

## Suggested CA-2 viva walkthrough

1. Explain that unpaired data lacks aligned image pairs, which is why CycleGAN adds cycle-consistency loss.
2. Point to the two generators and two discriminators in the training script; the live demo loads one direction's pretrained generator only.
3. Show preprocessing (`RGB → resize → normalize`) and the generator's tanh output mapping.
4. Run one inference and distinguish a generated translation from ground-truth evaluation.
5. Discuss domain shift, artifacts, checkpoint provenance, and the fact that pretrained weights were not trained by this app.

## References

- Zhu et al. (2017), CycleGAN paper: <https://arxiv.org/abs/1703.10593>
- Isola et al. (2017), Pix2Pix paper: <https://arxiv.org/abs/1611.07004>
- Original PyTorch CycleGAN and Pix2Pix repository: <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
- Berkeley CycleGAN project page: <https://junyanz.github.io/CycleGAN/>


## Presentation

The CA-2 presentation is included here: [CA2_ImageLab_AI.pptx](deliverables/CA2_ImageLab_AI.pptx). Replace the team/roll-number/section placeholders before submission.
