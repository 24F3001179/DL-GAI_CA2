from __future__ import annotations

import time
import hashlib
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from download_model import download
from utils.evaluation import paired_metrics
from utils.image_processing import image_to_png_bytes, is_valid_output
from utils.inference import MODEL_SOURCES, get_device, load_generator, translate


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"
st.set_page_config(page_title="ImageLab AI", page_icon="🧪", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 12% 0%, #1b2140 0, #11131a 40%, #0b0d12 100%); color:#e9ecf5;}
[data-testid="stSidebar"] {background:#10131c; border-right:1px solid #252b3b;}
h1,h2,h3 {letter-spacing:-.025em;}
div[data-testid="stMetric"] {background:#171b27; border:1px solid #2a3041; padding:14px; border-radius:12px;}
.stButton>button {border-radius:9px; font-weight:650;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_generator(task: str, modified: float):
    del modified
    return load_generator(MODEL_DIR / MODEL_SOURCES[task]["file"], get_device())


def checkpoint_path(task: str) -> Path:
    return MODEL_DIR / MODEL_SOURCES[task]["file"]


def page_home() -> None:
    st.title("ImageLab AI")
    st.subheader("Image-to-Image Translation Studio")
    st.write("Image-to-image translation takes a picture in one style or category and uses a neural network to generate a version in another, such as a horse image with zebra-like markings.")
    st.info("The studio runs real pretrained CycleGAN generators. Outputs are model predictions, not hand-edited effects.")
    a, b, c = st.columns(3)
    a.metric("Model", "CycleGAN ResNet-9")
    b.metric("Input size", "256 × 256")
    b.caption("Input is resized for inference; output is returned at the uploaded image dimensions.")
    c.metric("Inference", "CPU or CUDA")
    st.markdown("### Translation tasks")
    task_cols = st.columns(2)
    for column, task in zip(task_cols, MODEL_SOURCES):
        installed = checkpoint_path(task).is_file()
        with column.container(border=True):
            st.markdown(f"#### {task}")
            st.write("Pretrained checkpoint ready" if installed else "Checkpoint needs download")
    st.markdown("### Pipeline")
    st.code("Upload image → RGB + resize/normalize → ResNet generator → [-1, 1] output → RGB image", language="text")
    st.caption("Training is optional and separate from the pretrained demo. This application does not claim to have trained the pretrained checkpoint.")


def page_translate() -> None:
    st.title("Translate an image")
    task = st.selectbox("Translation task", list(MODEL_SOURCES))
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="translate_upload")
    sample_key = st.session_state.get("sample_image_path")
    image = None
    image_name = ""
    image_id = ""
    if uploaded is not None:
        try:
            with Image.open(uploaded) as decoded:
                image = decoded.convert("RGB")
            image_name = uploaded.name
            image_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")
    elif sample_key:
        path = Path(sample_key)
        if path.is_file():
            with Image.open(path) as decoded:
                image = decoded.convert("RGB")
            image_name = path.name
            image_id = hashlib.sha256(path.read_bytes()).hexdigest()
            st.success(f"Using gallery sample: {image_name}")

    if image is not None:
        left, right = st.columns(2)
        left.image(image, caption=f"Input · {image.width} × {image.height}", use_container_width=True)
        model_path = checkpoint_path(task)
        if not model_path.is_file():
            st.warning(f"The {task} checkpoint is not installed yet. Download it from the official Berkeley model archive to enable real inference.")
            if st.button(f"Download {task} checkpoint", type="primary"):
                with st.spinner("Downloading and validating the public checkpoint…"):
                    try:
                        download(task, MODEL_DIR)
                        cached_generator.clear()
                        st.success("Checkpoint verified. Click Translate Image to run inference.")
                    except Exception as exc:
                        st.error(f"Checkpoint download or validation failed: {exc}")
            return
        if st.button("Translate Image", type="primary", use_container_width=True):
            progress = st.progress(10, text="Loading model")
            try:
                generator = cached_generator(task, model_path.stat().st_mtime)
                progress.progress(45, text="Running generator")
                started = time.perf_counter()
                if get_device().type == "cuda":
                    torch.cuda.synchronize()
                result = translate(generator, image, get_device())
                if get_device().type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not is_valid_output(result):
                    raise ValueError("The model returned an invalid image")
                progress.progress(100, text="Translation complete")
                st.session_state["last_result"] = result
                st.session_state["last_input"] = image.copy()
                st.session_state["last_time"] = elapsed
                st.session_state["last_task"] = task
                st.session_state["last_input_name"] = image_name
                st.session_state["last_input_id"] = image_id
            except Exception as exc:
                progress.empty()
                st.error(f"Inference failed: {exc}")
        result = st.session_state.get("last_result")
        if (result is not None and st.session_state.get("last_task") == task
                and st.session_state.get("last_input_id") == image_id):
            right.image(result, caption=f"Generated · {result.width} × {result.height}", use_container_width=True)
            st.success(f"Inference completed in {st.session_state['last_time']:.2f} seconds on {get_device()}.")
            st.download_button("Download translated PNG", image_to_png_bytes(result),
                               file_name="imagelab_translation.png", mime="image/png")
            st.caption("CycleGAN is unpaired: no pixel-aligned ground truth is expected for this translation.")
        else:
            right.info("Choose Translate Image to run the pretrained generator.")
    else:
        st.info("Upload a JPG or PNG, or choose a sample from Gallery.")


def page_gallery() -> None:
    st.title("Sample gallery")
    samples = [path for path in ASSET_DIR.glob("sample_*.jpg")]
    if not samples:
        st.info("Sample photos are not bundled yet. The project avoids downloading the full training dataset for a demo; upload a horse image in Translate, or download the dataset only if you plan to train.")
        return
    cols = st.columns(min(3, len(samples)))
    for index, path in enumerate(samples):
        with cols[index % len(cols)]:
            st.image(path, caption=path.stem.replace("sample_", "").replace("_", " ").title(), use_container_width=True)
            if st.button("Use this sample", key=f"sample_{path.name}"):
                st.session_state["sample_image_path"] = str(path)
                st.success("Sample selected. Open Translate to run it through the model.")


def page_model_details() -> None:
    st.title("Model details")
    task = st.selectbox("Inspect task", list(MODEL_SOURCES), key="model_detail_task")
    source = MODEL_SOURCES[task]
    path = checkpoint_path(task)
    st.write(f"**Model:** CycleGAN ResNet generator (9 residual blocks)  ")
    st.write(f"**Framework:** PyTorch  ")
    st.write(f"**Input:** RGB, resized to {256} × {256}, normalized to [-1, 1]  ")
    st.write(f"**Checkpoint source:** [{source['url']}]({source['url']})  ")
    st.write(f"**Direction:** {source['direction']}  ")
    st.write(f"**Device:** {get_device()}  ")
    st.write(f"**Weights installed:** {'Yes' if path.is_file() else 'No'}  ")
    st.write("**Training status:** pretrained public weights; this app did not train them. A separate optional training script is included.")
    st.write("**Architecture:** 7×7 convolution → 2 downsampling blocks → 9 residual blocks → 2 upsampling blocks → tanh output.")
    st.write("**Provenance/license:** weights are served by the original CycleGAN project at UC Berkeley. The upstream code repository is BSD-3-Clause; the weight archive does not publish a separate model-card license, so this project downloads it locally and does not redistribute it.")
    st.caption("The official CycleGAN paper and repository are linked in README.md. Check upstream terms before external redistribution.")


def page_evaluation() -> None:
    st.title("Evaluation")
    input_image = st.session_state.get("last_input")
    output_image = st.session_state.get("last_result")
    if input_image is None or output_image is None:
        st.info("Run a translation first to see its input, output and inference time here.")
        return
    left, right = st.columns(2)
    left.image(input_image, caption="Input", use_container_width=True)
    right.image(output_image, caption="CycleGAN output", use_container_width=True)
    st.metric("Inference time", f"{st.session_state.get('last_time', 0):.2f} s")
    reference = st.file_uploader("Optional aligned reference image", type=["jpg", "jpeg", "png"], key="reference_upload")
    if reference:
        with Image.open(reference) as decoded:
            reference_image = decoded.convert("RGB")
        try:
            metrics = paired_metrics(output_image, reference_image)
            st.write(f"MAE: `{metrics['mae']:.5f}` · PSNR: `{metrics['psnr_db']:.2f} dB`")
            st.caption("These metrics are meaningful only if the reference is spatially aligned with the generated target.")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.info("No aligned target was supplied. MAE/PSNR are not reported for this unpaired CycleGAN translation.")
    st.write("Qualitative review: inspect whether the target domain cues appear and whether the source scene structure remains plausible. A visual example alone does not establish model accuracy.")


def page_theory() -> None:
    st.title("Deep learning theory")
    sections = [
        ("Image-to-image translation", "Maps an input image in one visual domain to an image in another, such as a horse photograph to a zebra-like rendering."),
        ("Generative adversarial network", "A generator creates candidate images while a discriminator learns to distinguish generated images from real examples. Their competing objectives improve realism."),
        ("Generator and discriminator", "The generator transforms pixels. The discriminator gives patch-level feedback about whether translated regions resemble the target domain."),
        ("Paired and unpaired data", "Pix2Pix learns from aligned input/target image pairs. CycleGAN can learn from two separate collections without one-to-one matching."),
        ("CycleGAN losses", "Adversarial loss encourages target-domain style. Cycle-consistency loss asks G_BA(G_AB(A)) ≈ A and G_AB(G_BA(B)) ≈ B. Identity loss can discourage unnecessary color changes."),
        ("Pix2Pix architecture", "The common Pix2Pix setup uses a U-Net generator with skip connections and a PatchGAN discriminator, trained with paired targets and a reconstruction term."),
        ("Applications and limits", "Uses include artistic translation, domain adaptation and data augmentation. The model can change or invent details, fail outside its training distribution, and inherit dataset bias."),
        ("Ethical use", "Label synthetic outputs clearly, avoid presenting them as documentary evidence, check data rights, and consider consent and downstream misuse."),
    ]
    for title, body in sections:
        with st.expander(title, expanded=title == "CycleGAN losses"):
            st.write(body)
    st.caption("CycleGAN: Zhu et al., ICCV 2017. Pix2Pix: Isola et al., CVPR 2017. References are in README.md.")


PAGES = {
    "Home": page_home,
    "Translate": page_translate,
    "Gallery": page_gallery,
    "Model Details": page_model_details,
    "Evaluation": page_evaluation,
    "Theory": page_theory,
}
with st.sidebar:
    st.markdown("## 🧪 ImageLab AI")
    st.caption("GAN image translation studio")
    selected_page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Inference device · {get_device()}")
PAGES[selected_page]()
