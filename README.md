# Multimodal Data Pipeline

A comprehensive toolkit for processing multimodal data across speech, vision, and text modalities.

## Installation

### Prerequisites
- Python 3.12
- Poetry
- Git

### Basic Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd multimodal-data-pipeline
   ```

2. Create a Poetry environment with Python 3.12:
   ```
   poetry env use python3.12
   ```

3. Activate the Poetry shell:
   ```
   poetry shell
   ```

4. Install dependencies with Poetry:
   ```
   poetry install
   ```

5. Install specific dependency groups as needed:
   ```
   poetry install --with speech,vision,text
   ```

## Usage

Import modules based on your needs:

```python
# Example for speech processing
import speechbrain as sb
from src.speech.processor import SpeechProcessor

# Example for vision processing
import mediapipe as mp
from src.vision.pose_analyzer import PoseAnalyzer

# Example for text analysis
from sentence_transformers import SentenceTransformer
from src.text.embedding_generator import TextEmbedder
```

## Model Categories

- **Speech**: Speech emotion recognition, transcription, and audio feature extraction
- **Vision**: Pose estimation, facial expression analysis, and motion tracking
- **Text**: Sentence embeddings, contextual representations, and semantic analysis
- **Multimodal**: Combined audio-visual analysis and integration
