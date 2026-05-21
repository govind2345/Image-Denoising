"""
app.py — Gradio Web UI for Image Denoising with Improved U-Net
================================================================
Upload a noisy/degraded image and optionally a defect mask.
The model restores the image while preserving defect regions.

Deploy on Hugging Face Spaces or run locally:
    pip install gradio torch torchvision pillow numpy
    python app.py
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr

# ── Model Architecture (duplicated here for standalone deployment) ─────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class ImprovedUNet(nn.Module):
    def __init__(self, skip_connections=None):
        super().__init__()
        self.skip_connections = [True, True, True, True] if skip_connections is None else skip_connections

        self.enc1 = self._conv_block(4, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.att1 = ChannelAttention(64)
        self.att2 = ChannelAttention(128)
        self.att3 = ChannelAttention(256)
        self.att4 = ChannelAttention(512)

        self.bottleneck = self._conv_block(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024 if self.skip_connections[3] else 512, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512 if self.skip_connections[2] else 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256 if self.skip_connections[1] else 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128 if self.skip_connections[0] else 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.mask_attention = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x, mask):
        mask_weights = self.mask_attention(mask)
        x = torch.cat([x, mask], dim=1)

        e1 = self.att1(self.enc1(x))
        e2 = self.att2(self.enc2(self.pool(e1)))
        e3 = self.att3(self.enc3(self.pool(e2)))
        e4 = self.att4(self.enc4(self.pool(e3)))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1) if self.skip_connections[3] else d4)
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1) if self.skip_connections[2] else d3)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1) if self.skip_connections[1] else d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1) if self.skip_connections[0] else d1)

        out = self.final(d1)
        return out * mask_weights + out * (1 - mask_weights)


# ── Load Model ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

model = ImprovedUNet(skip_connections=[False, True, True, True]).to(DEVICE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[OK] Model loaded from {MODEL_PATH} (best epoch: {checkpoint.get('epoch', '?')})")
else:
    print(f"[WARN] Model file '{MODEL_PATH}' not found - running with random weights")

model.eval()

# ── Inference helpers ──────────────────────────────────────────────────────────

IMG_SIZE = 256  # resize for inference

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def denoise_image(input_image, mask_image=None):
    """Run denoising inference on the uploaded image."""
    if input_image is None:
        raise gr.Error("Please upload an image first!")

    # Convert input
    input_pil = Image.fromarray(input_image).convert("RGB")
    original_size = input_pil.size  # (W, H)

    # Prepare image tensor
    img_tensor = transform(input_pil).unsqueeze(0).to(DEVICE)

    # Prepare mask tensor
    if mask_image is not None:
        mask_pil = Image.fromarray(mask_image).convert("L")
        mask_tensor = transform(mask_pil).unsqueeze(0).to(DEVICE)
    else:
        # No mask provided — use zeros (no defect regions)
        mask_tensor = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # Ensure mask is single channel
    if mask_tensor.shape[1] == 3:
        mask_tensor = mask_tensor.mean(dim=1, keepdim=True)

    # Inference
    with torch.no_grad():
        output = model(img_tensor, mask_tensor)

    # Post-process output
    output_np = output[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    output_pil = Image.fromarray(output_np).resize(original_size, Image.LANCZOS)
    output_final = np.array(output_pil)

    # Compute PSNR (between input & output at processing resolution)
    input_resized = np.array(input_pil.resize((IMG_SIZE, IMG_SIZE)))
    psnr_value = calculate_psnr(input_resized, output_np)

    info_text = f"🔍 **Processing Details**\n\n"
    info_text += f"- **Input size**: {original_size[0]}×{original_size[1]}\n"
    info_text += f"- **Processing resolution**: {IMG_SIZE}×{IMG_SIZE}\n"
    info_text += f"- **Defect mask**: {'Provided' if mask_image is not None else 'None (full denoising)'}\n"
    info_text += f"- **PSNR**: {psnr_value:.2f} dB\n"
    info_text += f"- **Device**: {DEVICE}\n"

    return output_final, info_text


# ── Example images ─────────────────────────────────────────────────────────────

def get_examples():
    """Find example images from the dataset if available."""
    examples = []
    dataset_dir = "Denoising_Dataset_prepared"
    if os.path.isdir(dataset_dir):
        for category in sorted(os.listdir(dataset_dir)):
            deg_dir = os.path.join(dataset_dir, category, "Val", "Degraded_image")
            mask_dir = os.path.join(dataset_dir, category, "Val", "Defect_mask")
            if os.path.isdir(deg_dir):
                imgs = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])
                if imgs:
                    img_path = os.path.join(deg_dir, imgs[0])
                    stem = os.path.splitext(imgs[0])[0]
                    mask_path = os.path.join(mask_dir, f"{stem}_mask.png")
                    if os.path.exists(mask_path):
                        examples.append([img_path, mask_path])
                    else:
                        examples.append([img_path, None])
            if len(examples) >= 5:
                break
    return examples if examples else None


# ── Gradio UI ──────────────────────────────────────────────────────────────────

TITLE = "🔬 Image Denoising with Improved U-Net"

DESCRIPTION = """
### Restore noisy & degraded images while preserving defect regions

This model uses an **Improved U-Net architecture** with channel attention mechanisms
to denoise images. It was trained on the **MVTec Anomaly Detection** dataset across
15 industrial object categories.

**How to use:**
1. **Upload a noisy/degraded image** (required)
2. **Upload a defect mask** (optional) — white regions indicate defects to preserve
3. Click **Denoise Image** to see the result

> 💡 *Without a mask, the model performs full-image denoising.*
"""

ARTICLE = """
### 📊 Model Performance (Validation Set)

| Metric | Value |
|--------|-------|
| Overall PSNR | 15.96 dB |
| Overall SSIM | 0.46 |
| Defect PSNR | 33.57 dB |
| **Defect SSIM** | **0.97** |

### 🏗️ Architecture

- **Improved U-Net** with 4 encoder/decoder levels
- **Channel Attention** modules at each encoder stage
- **Defect-Aware Mask Attention** for preserving anomalies
- **31M+ parameters** with batch normalization & dropout

---
*Built with PyTorch • Trained on MVTec AD Dataset •
[GitHub Repository](https://github.com/atharvarya12/Image-Denoising-with-Improved-U-Net-Architecture)*
"""

# Build theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
)

examples = get_examples()

with gr.Blocks(title="Image Denoising | U-Net") as demo:

    gr.Markdown(f"# {TITLE}", elem_id="title")
    gr.Markdown(DESCRIPTION, elem_id="description")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📷 Upload Noisy / Degraded Image",
                type="numpy",
                height=350,
            )
            mask_image = gr.Image(
                label="🎭 Upload Defect Mask (optional)",
                type="numpy",
                height=200,
            )
            denoise_btn = gr.Button(
                "✨ Denoise Image",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🖼️ Restored Image",
                type="numpy",
                height=350,
            )
            info_output = gr.Markdown(
                label="Processing Info",
                value="*Upload an image and click 'Denoise Image' to see results.*",
            )

    # Wire up the button
    denoise_btn.click(
        fn=denoise_image,
        inputs=[input_image, mask_image],
        outputs=[output_image, info_output],
    )

    # Examples section
    if examples:
        gr.Markdown("### 📁 Example Images")
        gr.Examples(
            examples=examples,
            inputs=[input_image, mask_image],
            outputs=[output_image, info_output],
            fn=denoise_image,
            cache_examples=False,
        )

    gr.Markdown(ARTICLE)

# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css="""
            .gradio-container { max-width: 1200px !important; }
            #title { text-align: center; margin-bottom: 0; }
            #description { text-align: center; }
            .output-image { border-radius: 12px; }
            footer { display: none !important; }
        """,
    )
