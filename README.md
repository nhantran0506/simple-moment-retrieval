# Simple Moment Retrieval

A Python-based video moment retrieval system that enables semantic search within video content using natural language queries. The system uses CLIP (Contrastive Language-Image Pre-training) embeddings and Qdrant vector database to find specific moments in videos based on textual descriptions.

## 🎯 Features

- **Scene Detection**: Automatically detects scene changes in videos
- **Semantic Search**: Search for video moments using natural language queries
- **CLIP Integration**: Uses OpenAI's CLIP model for multimodal understanding
- **Vector Database**: Leverages Qdrant for efficient similarity search
- **Visual Results**: Displays matching frames with timestamps and similarity scores

## 🔧 Requirements

- **Python**: 3.11
- **CUDA**: Optional (for GPU acceleration)
- **Docker**: Required for Qdrant database

## 📋 Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Qdrant Database

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

This will start Qdrant on:
- HTTP API: `http://localhost:6333`
- gRPC API: `http://localhost:6334`

## 🚀 Usage

### 1. Prepare Your Video

Place your video file in the `data/` directory or update the `CONFIG` section in `main.ipynb`:

```python
CONFIG = {
    "collection_name": "search_image_2025",
    "clip_model": "openai/clip-vit-large-patch14",
    "video_path": "path/to/your/video.mp4"  # Update this path
}
```

### 2. Run the Notebook

Open `main.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook main.ipynb
```

### 3. Process Your Video

The system will:
1. Detect scenes in your video
2. Extract representative frames
3. Generate CLIP embeddings
4. Store embeddings in Qdrant

### 4. Search for Moments

Use natural language queries to find specific moments:
- "person talking"
- "outdoor scene"
- "close-up shot"
- "news anchor"
- etc.

## 🔄 System Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Video   │───▶│  Scene Detection │───▶│ Frame Extraction│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Search Results  │◀───│   User Query     │    │ CLIP Embeddings │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│Frame Display &  │    │  Vector Search   │◀───│ Qdrant Database │
│   Timestamps    │    │   (Similarity)   │    │   (Storage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Detailed Process Flow:

1. **Scene Detection**: Uses PySceneDetect with AdaptiveDetector to identify scene boundaries
2. **Frame Extraction**: Extracts middle frame from each detected scene
3. **Embedding Generation**: Converts frames to 768-dimensional vectors using CLIP
4. **Vector Storage**: Stores embeddings in Qdrant with metadata (timestamps, frame indices)
5. **Query Processing**: Converts text queries to embeddings using CLIP
6. **Similarity Search**: Finds most similar frames using cosine similarity
7. **Result Display**: Shows matching frames with timestamps and confidence scores

## 📁 Project Structure

```
simple-moment-retrieval/
├── data/                          # Video files directory
│   └── your_video.mp4            # Place your videos here
├── main.ipynb                    # Main notebook with all functionality
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License file
```

## ⚙️ Configuration

Key configuration options in `main.ipynb`:

- `collection_name`: Qdrant collection name
- `clip_model`: CLIP model variant (default: "openai/clip-vit-large-patch14")
- `video_path`: Path to your input video

## 🎥 Supported Video Formats

The system supports common video formats including:
- MP4
- AVI
- MOV
- MKV
- And other OpenCV-supported formats

## 🚨 Important Notes

1. **Video Input**: You must provide your own video file(s) in the `data/` directory
2. **Qdrant Setup**: Ensure Qdrant is running before executing the notebook
3. **GPU Support**: CUDA-compatible GPU recommended for faster processing
4. **Memory Usage**: Large videos may require significant RAM for processing

## 🔍 Example Queries

Try these example queries:
- "person speaking into microphone"
- "outdoor landscape"
- "close-up face"
- "text on screen"
- "group of people"
- "indoor scene"

## 🛠️ Troubleshooting

### Common Issues:

1. **Qdrant Connection Error**: Ensure Docker container is running on port 6333
2. **CUDA Issues**: System will fallback to CPU if CUDA is unavailable
3. **Video Loading Error**: Check video file path and format compatibility
4. **Memory Issues**: Consider processing smaller video segments for large files

### Dependencies Issues:

If you encounter package conflicts, try creating a fresh virtual environment:

```bash
# Remove existing environment
rm -rf venv

# Create new environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Note**: This system is designed for educational and research purposes. Ensure you have appropriate rights to process any video content you use with this system.