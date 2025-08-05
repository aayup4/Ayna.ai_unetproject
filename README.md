# Ayna.ai_unetproject
Design and train a model that can colorize grayscale polygon shapes based on a given color name. The model should take two inputs:  A grayscale image of a polygon  A color name (e.g., "red", "blue")  And generate an output image where the polygon is filled with the correct colour


✅ README.md — Polygon Colorization using Conditional UNet
🎯 Project Title
Polygon Colorization via Conditional UNet

📝 Problem Statement
Design and train a model that can colorize grayscale polygon shapes based on a given color name.
The model should take two inputs:

A grayscale image of a polygon

A color name (e.g., "red", "blue")

And generate an output image where the polygon is filled with the correct color.

📁 Dataset
inputs/: Grayscale images of polygons

outputs/: Colored ground truth images

data.json: Maps each input polygon image and color name to the corresponding output image

Split into:

training/

validation/

⚙️ Architecture & Conditioning
🔧 Model
A custom U-Net architecture with:

3-level encoder-decoder structure

Skip connections between corresponding encoder/decoder blocks

Input image: 1×128×128 (grayscale polygon)

🎨 Conditioning Strategy
We tried two methods:

CLIP Embedding + FiLM Conditioning (Initial Approach):

Text tokenization and encoding via Hugging Face CLIP

Used [CLS] or mean-pooled token embedding

Injected into the bottleneck via Feature-wise Linear Modulation (FiLM)

(Recommended) RGB Concatenation (Simplified & Final):

Converted color name to a 3D RGB tensor

Concatenated with grayscale input → 4-channel input to the UNet

Led to much more stable training and visually correct outputs

🧪 Hyperparameters
Hyperparameter	Value	Notes
Optimizer	Adam	With default β values
Learning Rate	1e-4	Tuned from 1e-3 → 1e-4 for smoother training
Batch Size	1	Due to GPU memory in Colab
Image Size	128×128	Fixed across all stages
Loss Function	L1 / MSE	L1 performed better for color fidelity
Epochs	50	Enough for convergence on clean dataset

📈 Training Dynamics
Loss decreased steadily over 50 epochs

wandb logs show smooth convergence without spikes

Using CLIP-based conditioning alone gave unstable color outputs

Switching to RGB vector conditioning improved training speed and accuracy

🖼️ Qualitative Trends
Condition	Prediction Output	Observation
red	Polygon mostly red	✅ Accurate
green	Polygon greenish, background unchanged	✅ Good separation
blue	Faint blue polygon (CLIP version failed)	❌ CLIP conditioning failed

🚨 Failure Modes & Fixes
Issue	Fix Applied
CLIP embedding not aligning with output	Switched to RGB-based direct conditioning
Outputs were grayscale	Added correct output channel config (3-C)
Output color mismatch	Used L1 loss over MSE for better color match
Embedding shape mismatch	Replaced [CLS] with mean-pooled embedding

🧠 Key Learnings
Conditioning U-Nets with language embeddings can work, but may underperform without large data and deep supervision.

Direct RGB conditioning (via concatenation) is often better for visual generation tasks with fixed vocab like "red", "blue".

Visualization of average color values helped debug issues early in training.

Using FiLM layers requires careful tuning and strong supervision signals.
