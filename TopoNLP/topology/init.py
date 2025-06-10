"""
TopologicalNLP: A Revolutionary Framework for Topological Deep Learning in NLP
==============================================================================

TopologicalNLP is a comprehensive framework that combines mathematical topology,
deep learning, and natural language processing to create next-generation AI systems
that understand the geometric structure of language.

Key Components:
- Topological analysis tools (persistent homology, Hodge Laplacians, simplicial complexes)
- Topology-enhanced neural architectures (topological transformers, attention mechanisms)
- Comprehensive training and evaluation infrastructure
- Rich visualization and analysis utilities

Quick Start:
-----------
```python
import toponlp
from toponlp.topology import WordManifold
from toponlp.models import TopologicalTransformer

# Construct word manifold
manifold = WordManifold()
manifold.fit(texts)

# Extract topological features
features = manifold.persistent_homology()

# Train topological model
model = TopologicalTransformer(use_hodge_attention=True)
```

For more information, see the documentation at: https://toponlp.readthedocs.io
"""

__version__ = "0.1.0"
__author__ = "TopologicalNLP Research Team"
__email__ = "team@toponlp.org"
__license__ = "MIT"
__copyright__ = "Copyright 2024, TopologicalNLP Research Team"

# Core imports for easy access
from toponlp.core.config import TopoNLPConfig
from toponlp.core.base import BaseTopologicalModel

# Topology module
from toponlp.topology.manifolds import WordManifold
from toponlp.topology.persistent_homology import PersistentHomology
from toponlp.topology.hodge_laplacian import HodgeLaplacian
from toponlp.topology.complexes import SimplicialComplex

# NLP module
from toponlp.nlp.embeddings import TopologicalEmbedding
from toponlp.nlp.tokenizers import TopologicalTokenizer
from toponlp.nlp.preprocessing import TopologicalPreprocessor

# Models module
from toponlp.models.topological_transformers import TopologicalTransformer
from toponlp.models.topological_attention import HodgeAttention
from toponlp.models.gnn_models import TopologicalGNN

# Training module
from toponlp.training.trainer import TopologicalTrainer
from toponlp.training.metrics import TopologicalMetrics

# Utilities
from toponlp.utils.visualization import plot_persistence_diagram, plot_manifold
from toponlp.utils.logging import get_logger

# Version information
__all__ = [
    # Core
    "TopoNLPConfig",
    "BaseTopologicalModel",
    
    # Topology
    "WordManifold",
    "PersistentHomology", 
    "HodgeLaplacian",
    "SimplicialComplex",
    
    # NLP
    "TopologicalEmbedding",
    "TopologicalTokenizer",
    "TopologicalPreprocessor",
    
    # Models
    "TopologicalTransformer",
    "HodgeAttention",
    "TopologicalGNN",
    
    # Training
    "TopologicalTrainer",
    "TopologicalMetrics",
    
    # Utilities
    "plot_persistence_diagram",
    "plot_manifold",
    "get_logger",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Package-level configuration
import logging
import os
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up package logger
logger = logging.getLogger(__name__)

# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Check for required dependencies
def _check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import gudhi
    except ImportError:
        missing_deps.append("gudhi")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        logger.warning(
            f"Missing dependencies: {', '.join(missing_deps)}. "
            "Some functionality may not be available. "
            "Install with: pip install -r requirements.txt"
        )

# Check GPU availability
def _check_gpu():
    """Check GPU availability and log information."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU acceleration available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            logger.info("GPU acceleration not available, using CPU")
    except ImportError:
        pass

# Initialize package
def _initialize_package():
    """Initialize the package."""
    try:
        _check_dependencies()
        _check_gpu()
        
        # Set environment variables for better performance
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        
        logger.info(f"TopologicalNLP v{__version__} initialized successfully")
        
    except Exception as e:
        logger.warning(f"Package initialization warning: {e}")

# Initialize when imported
_initialize_package()

# Convenience functions
def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_config():
    """Get default configuration object."""
    return TopoNLPConfig()

# Add convenience functions to __all__
__all__.extend([
    "get_device",
    "set_random_seed", 
    "get_config"
])
