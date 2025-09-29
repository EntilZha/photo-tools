
import os
import numpy as np
import rawpy
from PIL import Image
import typer
import matplotlib.pyplot as plt

def main(
	raw_path: str = typer.Argument(..., help="Path to raw file"),
	output_dir: str = typer.Argument(..., help="Output directory")
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
	gray_img = img.convert('L')
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
	plt.figure()
	for i, color in enumerate(["r", "g", "b"]):
		plt.hist(rgb[..., i].ravel(), bins=256, color=color, alpha=0.5, label=f"{color.upper()}")
	plt.legend()
	plt.title("RGB Histogram")
	plt.xlabel("Value")
	plt.ylabel("Count")
	plt.savefig(os.path.join(output_dir, f"{prefix}rgb_histogram.png"))
	plt.close()

	# Grayscale histogram
	plt.figure()
	plt.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
	plt.title("Grayscale Histogram")
	plt.xlabel("Value")
	plt.ylabel("Count")
	plt.savefig(os.path.join(output_dir, f"{prefix}grayscale_histogram.png"))
	plt.close()

if __name__ == "__main__":
	typer.run(main)

