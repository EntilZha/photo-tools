import os
import numpy as np
import rawpy
from PIL import Image
import typer
import matplotlib.pyplot as plt


def main(
    raw_path: str = typer.Argument(..., help="Path to raw file"),
    output_dir: str = typer.Argument(..., help="Output directory"),
):
    os.makedirs(output_dir, exist_ok=True)

    # Get the raw filename for prefixing
    raw_filename = os.path.basename(raw_path)
    prefix = f"{raw_filename}."

    # Load raw image
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess()

    # Save as JPEG
    img = Image.fromarray(rgb)
    img.save(os.path.join(output_dir, f"{prefix}image.jpg"))

    # Grayscale version
    gray_img = img.convert("L")
    gray = np.array(gray_img)
    gray_img.save(os.path.join(output_dir, f"{prefix}grayscale.jpg"))

    # Thresholded grayscale: 255 -> white, else black
    thresh = np.where(gray == 255, 255, 0).astype(np.uint8)
    thresh_img = Image.fromarray(thresh)
    thresh_img.save(os.path.join(output_dir, f"{prefix}grayscale_threshold.jpg"))

    # Channel maxed out images
    for i, channel in enumerate(["red", "green", "blue"]):
        channel_data = rgb[..., i]
        maxed = np.where(channel_data == 255, 255, 0).astype(np.uint8)
        maxed_img = Image.fromarray(maxed)
        maxed_img.save(os.path.join(output_dir, f"{prefix}{channel}_maxed.jpg"))

    # RGB histograms
    plt.figure(figsize=(10, 6))
    colors = ["r", "g", "b"]
    channel_names = ["Red", "Green", "Blue"]

    for i, (color, channel_name) in enumerate(zip(colors, channel_names)):
        channel_data = rgb[..., i].ravel()
        total_pixels = len(channel_data)

        # Calculate exposure statistics
        overexposed = np.sum(channel_data == 255)
        underexposed = np.sum(channel_data == 0)
        overexposed_pct = (overexposed / total_pixels) * 100
        underexposed_pct = (underexposed / total_pixels) * 100

        plt.hist(
            channel_data, bins=256, color=color, alpha=0.5, label=f"{channel_name}"
        )

        # Add text annotations for this channel
        y_pos = 0.9 - i * 0.1
        plt.text(
            0.02,
            y_pos,
            f"{channel_name}: Overexp: {overexposed_pct:.2f}% | Underexp: {underexposed_pct:.2f}%",
            transform=plt.gca().transAxes,
            fontsize=9,
            color=color,
        )

    plt.legend()
    plt.title("RGB Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, f"{prefix}rgb_histogram.png"))
    plt.close()

    # Grayscale histogram
    plt.figure(figsize=(10, 6))
    total_gray_pixels = gray.size
    gray_overexposed = np.sum(gray == 255)
    gray_underexposed = np.sum(gray == 0)
    gray_overexposed_pct = (gray_overexposed / total_gray_pixels) * 100
    gray_underexposed_pct = (gray_underexposed / total_gray_pixels) * 100

    plt.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
    plt.title("Grayscale Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")

    # Add exposure statistics text
    plt.text(
        0.02,
        0.9,
        f"Overexposed: {gray_overexposed_pct:.2f}% | Underexposed: {gray_underexposed_pct:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
    )

    plt.savefig(os.path.join(output_dir, f"{prefix}grayscale_histogram.png"))
    plt.close()


if __name__ == "__main__":
    typer.run(main)
