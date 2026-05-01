# Music Genre Classifier

An audio classification system using the AST (Audio Spectrogram Transformer) model, fine-tuned to classify music into 10 genres. Includes a Jupyter notebook for model development and a Gradio web interface for inference.

## Features

- Audio classification into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
- Pre-trained / fine-tuned AST model: `chopadaansh/ast-music-genre-classifier`.
- Gradio web interface for quick audio upload and inference.
- Development notebook for experiments and analysis.

## Project Structure

```
├── deployment/
│   ├── app.py              # Gradio web interface
│   └── requirements.txt    # Python dependencies
├── notebooks/
│   └── main.ipynb          # Model development & analysis
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- `pip` (or `conda`)

### Installation

1. Clone the repository (replace with your repo URL):

```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classifier.git
cd music-genre-classifier
```

2. (Recommended) Create and activate a virtual environment:

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r deployment/requirements.txt
```

## Usage

### Run the Web App

Start the Gradio interface:

```bash
python deployment/app.py
```

The app will start at `http://localhost:7860` (or another available port). Upload an audio clip and view the predicted genre and confidence scores.

### Live demo

A hosted demo of this project is available on Hugging Face Spaces:

https://huggingface.co/spaces/chopadaansh/music-genre-classifier

### Run the Notebook

Open the development notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

## Model Details

- Fine-tuned model: `chopadaansh/ast-music-genre-classifier`
- Audio format: 16kHz mono
- Output classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Dependencies

See `deployment/requirements.txt`. Key packages include `torch`, `transformers`, `librosa`, `soundfile`, `gradio`, and `numpy`.

## Pushing to GitHub — Step-by-step

1. Initialize git (if not already a repo):

```bash
git init
```

2. Create a helpful `.gitignore` (Python typical) and add files:

```bash
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
git add .gitignore
git add .
```

3. Commit changes:

```bash
git commit -m "feat: initial commit — Music Genre Classifier"
```

4. Create a remote repository on GitHub (choose one):

- Using the GitHub website: create a new repo under your account and copy the remote URL.
- Or using GitHub CLI:

```bash
gh repo create YOUR_USERNAME/music-genre-classifier --public --source=. --remote=origin --push
```

5. Or add remote and push manually:

```bash
git remote add origin https://github.com/YOUR_USERNAME/music-genre-classifier.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and the repo name with your choices.

## Contributing

Contributions welcome — please open issues or PRs.

## License

Add a license of your choice (e.g., MIT). Replace this section with the actual license text or a link.

## Author

Your name or GitHub handle

## Acknowledgments

- AST model and HuggingFace Transformers
- Gradio for the web UI

