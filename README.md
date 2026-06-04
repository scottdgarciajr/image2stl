# Image to STL Converter

A local web app for turning images into printable STL models. It includes standard height-map modes plus a product-focused mode for extracting a subject, smoothing its outline, optionally removing the image background, editing the mask with an eraser, and exporting a closed STL.

## Features

- Browser-based interface with live 3D preview.
- Standard grayscale and color height-map conversion modes.
- Product mode for subject-shaped relief models.
- Adjustable subject crop, placement, pickup thresholds, smoothing, relief, and optional frame.
- Toggleable image background remover before mask/STL generation.
- Manual mask eraser for cleaning up unwanted areas.
- Direct command-line export for repeatable jobs.

## Requirements

- Python 3.10 or newer.
- A modern browser.
- Python packages listed in `requirements.txt`.

No paid API keys or cloud services are required. The current background remover is local and dependency-light; SAM2/potrace are not required for the included workflow.

## Setup

1. Clone or download this repository.

2. Open a terminal in the project folder.

3. Create a virtual environment.

```powershell
python -m venv .venv
```

4. Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

5. Install dependencies.

```bash
pip install -r requirements.txt
```

## Run The App

```bash
python image_to_stl.py
```

Then open:

```text
http://localhost:5000
```

To use a different port:

```bash
python image_to_stl.py --port 5050
```

## Basic Workflow

1. Upload an image.
2. Choose `Product` mode for a subject-shaped model, or use the standard modes for a rectangular height-map.
3. Use `Subject Shape` controls first:
   - `IMAGE BG REMOVE`: toggle early background removal.
   - `IMAGE BG STRENGTH`: increase if background remains; reduce if subject edges are being eaten.
   - `EDGE SMOOTH`: rounds the subject outline.
   - `MASK GROW` and `MASK CLOSE`: expand and clean the generated mask.
4. Use `Place & Crop` to frame the subject.
5. Use `Pickup Thresholds` only if the mask still grabs too much or too little.
6. Use the eraser panel under the uploaded image to manually remove unwanted mask areas.
7. Export STL.

## Product Controls

`Subject Shape` contains the most important controls for the final silhouette.

`Place & Crop` changes how the uploaded image is cropped and fitted into the product area.

`Pickup Thresholds` adjusts the image analysis used to identify the subject.

`Relief & Detail` changes raised details and surface height.

`Frame` controls the optional circular backing, diagonal braces, and corner holes.

## Command-Line Export

You can export without launching the browser.

```bash
python image_to_stl.py --input examples/testcar2.png --output model.stl --filter product
```

Useful product flags:

```bash
python image_to_stl.py --input examples/testcar2.png --output model.stl --filter product --image-bg-remove --image-bg-strength 0.55 --edge-smooth 6
```

Disable the frame:

```bash
python image_to_stl.py --input examples/testcar2.png --output model.stl --filter product --no-frame
```

Standard rectangular height-map export:

```bash
python image_to_stl.py --input examples/test_car.png --output heightmap.stl --filter standard --detail high --width 80 --height 80 --depth 5
```

## Editing The App

The project is intentionally contained in one Python file:

- `image_to_stl.py`: Flask server, STL generation code, image processing, and embedded HTML/CSS/JavaScript UI.
- `requirements.txt`: Python dependencies.
- `examples/`: sample input images.

Important areas inside `image_to_stl.py`:

- Image processing helpers near the top: background removal, mask cleanup, smoothing.
- `make_product_height_and_mask(...)`: product-mode mask and height generation.
- `masked_heightfield_to_stl(...)`: STL mesh generation for subject-shaped products.
- `HTML_PAGE`: embedded frontend UI and controls.
- `main()`: command-line arguments and app startup.

After editing, run:

```bash
python -m py_compile image_to_stl.py
```

Then start the app and test a preview/export.

## Troubleshooting

If `ModuleNotFoundError` appears, activate the virtual environment and rerun:

```bash
pip install -r requirements.txt
```

If the app says the port is already in use, start it on another port:

```bash
python image_to_stl.py --port 5050
```

If a product looks like a blob, turn `IMAGE BG REMOVE` off or lower `IMAGE BG STRENGTH`, then adjust `MASK GROW`, `MASK CLOSE`, and `EDGE SMOOTH`.

If unwanted background remains, turn `IMAGE BG REMOVE` on, raise `IMAGE BG STRENGTH` slightly, crop tighter, or paint it out with the eraser.

## GitHub Notes

Generated STL files and debug/tuning images are ignored by `.gitignore`, so the repository stays small. Keep source images in `examples/` only if they are okay to publish.
