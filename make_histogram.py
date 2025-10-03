import os
import numpy as np
from PIL import Image
import typer
import matplotlib.pyplot as plt


def main(
    image_path: str = typer.Argument(..., help="Path to JPEG image"),
    output_dir: str = typer.Argument(
        ".", help="Output directory (default: current directory)"
    ),
):
    os.makedirs(output_dir, exist_ok=True)

    # Get the image filename for prefixing (without extension)
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        # Color image
        gray_img = img.convert("L")
        gray = np.array(gray_img)
        is_color = True
    else:
        # Already grayscale
        gray = img_array
        is_color = False

    # Grayscale histogram with exposure statistics
    plt.figure(figsize=(10, 6))
    total_gray_pixels = gray.size
    gray_overexposed = np.sum(gray == 255)
    gray_underexposed = np.sum(gray == 0)
    gray_overexposed_pct = (gray_overexposed / total_gray_pixels) * 100
    gray_underexposed_pct = (gray_underexposed / total_gray_pixels) * 100
    gray_avg = np.mean(gray)

    plt.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
    plt.title(f"Grayscale Exposure Histogram - {os.path.basename(image_path)}")
    plt.xlabel("Value")
    plt.ylabel("Count")

    # Add exposure statistics text
    plt.text(
        0.02,
        0.9,
        f"Overexposed: {gray_overexposed_pct:.2f}% | Underexposed: {gray_underexposed_pct:.2f}% | Avg: {gray_avg:.1f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.savefig(
        os.path.join(output_dir, f"{image_filename}_grayscale_histogram.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # RGB histograms if color image
    if is_color:
        plt.figure(figsize=(12, 6))
        colors = ["r", "g", "b"]
        channel_names = ["Red", "Green", "Blue"]

        for i, (color, channel_name) in enumerate(zip(colors, channel_names)):
            channel_data = img_array[..., i].ravel()
            total_pixels = len(channel_data)

            # Calculate exposure statistics
            overexposed = np.sum(channel_data == 255)
            underexposed = np.sum(channel_data == 0)
            overexposed_pct = (overexposed / total_pixels) * 100
            underexposed_pct = (underexposed / total_pixels) * 100
            channel_avg = np.mean(channel_data)

            plt.hist(
                channel_data, bins=256, color=color, alpha=0.5, label=f"{channel_name}"
            )

            # Add text annotations for this channel
            y_pos = 0.9 - i * 0.08
            plt.text(
                0.02,
                y_pos,
                f"{channel_name}: Overexp: {overexposed_pct:.2f}% | Underexp: {underexposed_pct:.2f}% | Avg: {channel_avg:.1f}",
                transform=plt.gca().transAxes,
                fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        plt.legend()
        plt.title(f"RGB Exposure Histogram - {os.path.basename(image_path)}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.savefig(
            os.path.join(output_dir, f"{image_filename}_rgb_histogram.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Generated histograms for color image: {image_filename}_grayscale_histogram.png and {image_filename}_rgb_histogram.png"
        )
    else:
        print(
            f"Generated histogram for grayscale image: {image_filename}_grayscale_histogram.png"
        )


if __name__ == "__main__":
    typer.run(main)
