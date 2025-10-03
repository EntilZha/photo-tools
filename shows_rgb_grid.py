import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import typer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


def rgb_to_grayscale(r, g, b):
    """Convert RGB values to grayscale using standard formula"""
    return int(0.299 * r + 0.587 * g + 0.114 * b)


def main(
    image_path: str = typer.Argument(..., help="Path to JPEG image"),
    x: int = typer.Argument(..., help="X coordinate for crop start"),
    y: int = typer.Argument(..., help="Y coordinate for crop start"),
    size: int = typer.Argument(..., help="Size of the crop (width and height)"),
    output_dir: str = typer.Argument(
        ".", help="Output directory (default: current directory)"
    ),
):
    os.makedirs(output_dir, exist_ok=True)

    # Get the image filename for output
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Load and crop the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Crop the image
    cropped = img_array[y : y + size, x : x + size]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        print(f"Warning: Crop size is {cropped.shape[:2]}, requested {size}x{size}")

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Display the cropped image
    ax.imshow(cropped)

    # Add grid lines between pixels
    for i in range(cropped.shape[1] + 1):
        ax.axvline(x=i - 0.5, color="white", linewidth=0.5, alpha=0.7)
    for i in range(cropped.shape[0] + 1):
        ax.axhline(y=i - 0.5, color="white", linewidth=0.5, alpha=0.7)

    # Select 4 pixel locations to annotate (corners and center-ish)
    h, w = cropped.shape[:2]
    pixel_locations = [
        (w // 4, h // 4),  # Top-left quadrant
        (3 * w // 4, h // 4),  # Top-right quadrant
        (w // 4, 3 * h // 4),  # Bottom-left quadrant
        (3 * w // 4, 3 * h // 4),  # Bottom-right quadrant
    ]

    # Colors for the annotation boxes
    box_colors = ["red", "blue", "green", "orange"]

    for i, (px, py) in enumerate(pixel_locations):
        # Ensure pixel coordinates are within bounds
        px = min(px, w - 1)
        py = min(py, h - 1)

        # Get RGB values
        r, g, b = cropped[py, px]
        gray_val = rgb_to_grayscale(r, g, b)

        # Create a box around the pixel
        box = patches.Rectangle(
            (px - 0.4, py - 0.4),
            0.8,
            0.8,
            linewidth=2,
            edgecolor=box_colors[i],
            facecolor="none",
        )
        ax.add_patch(box)

        # Determine arrow position (outside the image)
        if i == 0:  # Top-left pixel -> arrow from left
            arrow_start = (-w * 0.3, py)
            arrow_end = (px - 0.5, py)
            ha, va = "right", "center"
            text_x, text_y = -w * 0.35, py
        elif i == 1:  # Top-right pixel -> arrow from top
            arrow_start = (px, -h * 0.3)
            arrow_end = (px, py - 0.5)
            ha, va = "center", "bottom"
            text_x, text_y = px, -h * 0.4
        elif i == 2:  # Bottom-left pixel -> arrow from bottom
            arrow_start = (px, h + h * 0.2)
            arrow_end = (px, py + 0.5)
            ha, va = "center", "top"
            text_x, text_y = px, h + h * 0.25
        else:  # Bottom-right pixel -> arrow from right
            arrow_start = (w + w * 0.2, py)
            arrow_end = (px + 0.5, py)
            ha, va = "left", "center"
            text_x, text_y = w + w * 0.25, py

        # Draw arrow
        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(arrowstyle="->", color=box_colors[i], lw=2),
        )

        # Add text label with RGB and grayscale values
        label = f"RGB: ({r}, {g}, {b})\nGray: {gray_val}"
        ax.text(
            text_x,
            text_y,
            label,
            ha=ha,
            va=va,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=box_colors[i],
                alpha=0.9,
            ),
            color=box_colors[i],
            weight="bold",
        )

    # Set title and labels
    ax.set_title(
        f"RGB Pixel Analysis - {os.path.basename(image_path)}\nCrop: ({x}, {y}) Size: {size}x{size}",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    # Extend axis limits to show annotations
    ax.set_xlim(-w * 0.5, w + w * 0.4)
    ax.set_ylim(h + h * 0.4, -h * 0.5)

    # Save the figure
    output_path = os.path.join(output_dir, f"{image_filename}_rgb_pixels.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Generated RGB pixel analysis: {output_path}")


if __name__ == "__main__":
    typer.run(main)
