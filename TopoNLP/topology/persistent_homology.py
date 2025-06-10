"""
Persistent Homology Module for Topological Data Analysis in NLP.

This module provides tools for computing persistent homology from NLP data,
extracting topological features, and analyzing the structure of language.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import pdist, squareform

try:
    import gudhi as gd
    from gudhi.point_cloud.timedelay import TimeDelayEmbedding
    GUDHI_AVAILABLE = True
except ImportError:
    logger.warning("GUDHI not available. Some functionality will be limited.")
    GUDHI_AVAILABLE = False

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    logger.warning("Ripser not available. Falling back to GUDHI if available.")
    RIPSER_AVAILABLE = False

try:
    import persim
    PERSIM_AVAILABLE = True
except ImportError:
    logger.warning("Persim not available. Persistence diagram visualization will be limited.")
    PERSIM_AVAILABLE = False

try:
    from giotto.diagrams import PersistenceEntropy, BettiCurve, PersistenceLandscape
    GIOTTO_AVAILABLE = True
except ImportError:
    logger.warning("Giotto-TDA not available. Some feature extraction methods will be limited.")
    GIOTTO_AVAILABLE = False


class PersistentHomology:
    """Persistent homology computation for text data.
    
    This class provides tools for computing persistent homology on NLP data,
    extracting topological features from the resulting persistence diagrams,
    and analyzing the topological structure of language.
    
    Attributes:
        max_dimension (int): Maximum homology dimension to compute.
        metric (str): Distance metric to use for constructing distance matrix.
        backend (str): Backend to use for computation ('gudhi', 'ripser', or 'auto').
        filtration_type (str): Type of filtration to use ('vietoris_rips', 'alpha', 'witness').
        max_edge_length (float): Maximum length of edges to include in the filtration.
        n_jobs (int): Number of parallel jobs for computation.
        verbose (bool): Whether to display verbose output.
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        metric: str = "cosine",
        backend: str = "auto",
        filtration_type: str = "vietoris_rips",
        max_edge_length: float = np.inf,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the PersistentHomology calculator.
        
        Args:
            max_dimension: Maximum homology dimension to compute.
            metric: Distance metric to use ('cosine', 'euclidean', etc.).
            backend: Computation backend ('gudhi', 'ripser', or 'auto').
            filtration_type: Type of filtration to use.
            max_edge_length: Maximum edge length for filtration.
            n_jobs: Number of parallel jobs.
            verbose: Whether to display verbose output.
        """
        self.max_dimension = max_dimension
        self.metric = metric
        self.filtration_type = filtration_type
        self.max_edge_length = max_edge_length
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set the backend based on availability
        if backend == "auto":
            if RIPSER_AVAILABLE and max_dimension <= 2:
                self.backend = "ripser"  # Ripser is faster for low dimensions
            elif GUDHI_AVAILABLE:
                self.backend = "gudhi"
            else:
                raise ImportError(
                    "No persistent homology backend available. "
                    "Please install GUDHI or Ripser."
                )
        else:
            self.backend = backend
            if backend == "gudhi" and not GUDHI_AVAILABLE:
                raise ImportError("GUDHI backend requested but not available.")
            if backend == "ripser" and not RIPSER_AVAILABLE:
                raise ImportError("Ripser backend requested but not available.")
        
        if self.verbose:
            logger.info(f"Using {self.backend} backend for persistent homology computation")
    
    def fit_transform(
        self, 
        X: Union[np.ndarray, torch.Tensor, List[List[float]]],
        y: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """Compute persistent homology and return persistence diagrams.
        
        Args:
            X: Input data with shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            List of persistence diagrams, one for each homology dimension.
        """
        # Convert to numpy array if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif isinstance(X, list):
            X = np.array(X)
        
        # Check if we already have a distance matrix
        if len(X.shape) == 2 and X.shape[0] == X.shape[1]:
            is_distance_matrix = np.allclose(X, X.T) and np.all(np.diag(X) == 0)
            if is_distance_matrix:
                distance_matrix = X
            else:
                # Compute distance matrix
                distance_matrix = squareform(pdist(X, metric=self.metric))
        else:
            # Compute distance matrix
            distance_matrix = squareform(pdist(X, metric=self.metric))
        
        # Compute persistent homology
        if self.backend == "ripser":
            diagrams = self._compute_ph_ripser(distance_matrix)
        else:  # gudhi
            diagrams = self._compute_ph_gudhi(distance_matrix)
        
        return diagrams
    
    def _compute_ph_ripser(self, distance_matrix: np.ndarray) -> List[np.ndarray]:
        """Compute persistent homology using Ripser.
        
        Args:
            distance_matrix: Pairwise distance matrix.
            
        Returns:
            List of persistence diagrams, one for each homology dimension.
        """
        if self.verbose:
            logger.info(f"Computing persistent homology with Ripser (max dim: {self.max_dimension})")
        
        result = ripser.ripser(
            distance_matrix,
            maxdim=self.max_dimension,
            distance_matrix=True,
            thresh=self.max_edge_length
        )
        
        # Extract diagrams from Ripser output
        diagrams = result['dgms']
        
        # Filter out points with infinite persistence
        filtered_diagrams = []
        for diag in diagrams:
            # Get points with finite persistence
            finite_mask = np.isfinite(diag[:, 1])
            filtered_diag = diag[finite_mask]
            filtered_diagrams.append(filtered_diag)
        
        return filtered_diagrams
    
    def _compute_ph_gudhi(self, distance_matrix: np.ndarray) -> List[np.ndarray]:
        """Compute persistent homology using GUDHI.
        
        Args:
            distance_matrix: Pairwise distance matrix.
            
        Returns:
            List of persistence diagrams, one for each homology dimension.
        """
        if self.verbose:
            logger.info(f"Computing persistent homology with GUDHI (max dim: {self.max_dimension})")
        
        if self.filtration_type == "vietoris_rips":
            # Construct Vietoris-Rips complex
            rips_complex = gd.RipsComplex(
                distance_matrix=distance_matrix,
                max_edge_length=self.max_edge_length
            )
            
            # Create simplex tree
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension + 1)
            
        elif self.filtration_type == "alpha":
            # Alpha complex requires point cloud, not distance matrix
            # This is an approximation that works reasonably well for NLP
            from sklearn.manifold import MDS
            
            mds = MDS(n_components=3, dissimilarity="precomputed", n_jobs=self.n_jobs)
            points = mds.fit_transform(distance_matrix)
            
            # Construct Alpha complex
            alpha_complex = gd.AlphaComplex(points=points)
            simplex_tree = alpha_complex.create_simplex_tree()
            
        elif self.filtration_type == "witness":
            # Witness complex requires landmarks
            # Use farthest point sampling for landmarks
            n_landmarks = min(50, distance_matrix.shape[0])
            from gudhi.clustering.tomato import Tomato
            
            tomato = Tomato(points=None, distance_matrix=distance_matrix, n_jobs=self.n_jobs)
            landmarks = tomato.get_n_farthest_points(n_landmarks)
            
            # Construct Witness complex
            witness_complex = gd.EuclideanWitnessComplex()
            simplex_tree = witness_complex.create_simplex_tree(
                landmarks=landmarks,
                witnesses=range(distance_matrix.shape[0]),
                distances=distance_matrix,
                max_alpha=self.max_edge_length,
            )
        else:
            raise ValueError(f"Unknown filtration type: {self.filtration_type}")
        
        # Compute persistence
        simplex_tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
        
        # Extract persistence diagrams
        diagrams = []
        for dim in range(self.max_dimension + 1):
            persistence_pairs = simplex_tree.persistence_pairs_in_dimension(dim)
            
            # Convert to numpy array
            if persistence_pairs:
                birth_death_pairs = []
                for birth, death in persistence_pairs:
                    birth_value = simplex_tree.filtration(birth)
                    death_value = simplex_tree.filtration(death)
                    birth_death_pairs.append([birth_value, death_value])
                
                diagrams.append(np.array(birth_death_pairs))
            else:
                diagrams.append(np.empty((0, 2)))
        
        return diagrams
    
    def extract_features(
        self, 
        diagrams: List[np.ndarray], 
        feature_type: str = "statistics"
    ) -> np.ndarray:
        """Extract features from persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams from fit_transform.
            feature_type: Type of features to extract:
                - 'statistics': Basic statistics (birth, death, persistence)
                - 'vectorized': Vectorized representations (persistence landscape, etc.)
                - 'entropy': Persistence entropy
                - 'betti': Betti curves
                - 'all': All available features
            
        Returns:
            Array of features with shape (n_diagrams_dimensions, n_features).
        """
        features = []
        
        for dim, diagram in enumerate(diagrams):
            if diagram.shape[0] == 0:
                # No features in this dimension
                if feature_type in ["statistics", "all"]:
                    # Add zeros for basic statistics
                    features.append(np.zeros(6))  # 6 basic statistics
                if feature_type in ["entropy", "all"] and GIOTTO_AVAILABLE:
                    # Add zero for entropy
                    features.append(np.zeros(1))
                if feature_type in ["betti", "all"] and GIOTTO_AVAILABLE:
                    # Add zeros for Betti curve
                    features.append(np.zeros(10))  # 10 samples for Betti curve
                if feature_type in ["vectorized", "all"] and GIOTTO_AVAILABLE:
                    # Add zeros for persistence landscape
                    features.append(np.zeros(20))  # 20 values for landscape
                continue
            
            # Always compute basic statistics (very cheap)
            if feature_type in ["statistics", "all"]:
                # Basic statistics
                birth = diagram[:, 0]
                death = diagram[:, 1]
                persistence = death - birth
                
                stats = np.array([
                    np.mean(persistence),
                    np.std(persistence),
                    np.max(persistence),
                    np.sum(persistence),
                    np.mean(birth),
                    np.mean(death)
                ])
                
                features.append(stats)
            
            # More advanced features if requested
            if GIOTTO_AVAILABLE:
                # Convert to Giotto format (may need reshaping)
                giotto_diagram = np.zeros((diagram.shape[0], 3))
                giotto_diagram[:, 0] = dim  # homology dimension
                giotto_diagram[:, 1:] = diagram  # birth-death pairs
                
                if feature_type in ["entropy", "all"]:
                    # Persistence entropy
                    entropy_extractor = PersistenceEntropy()
                    entropy = entropy_extractor.fit_transform([giotto_diagram])
                    features.append(entropy.flatten())
                
                if feature_type in ["betti", "all"]:
                    # Betti curves
                    betti_extractor = BettiCurve(n_bins=10)
                    betti_curve = betti_extractor.fit_transform([giotto_diagram])
                    features.append(betti_curve.flatten())
                
                if feature_type in ["vectorized", "all"]:
                    # Persistence landscapes
                    landscape_extractor = PersistenceLandscape(
                        n_layers=2,
                        n_bins=10,
                        n_jobs=self.n_jobs
                    )
                    landscape = landscape_extractor.fit_transform([giotto_diagram])
                    features.append(landscape.flatten())
        
        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.array([])
    
    def persistence_image(
        self, 
        diagrams: List[np.ndarray],
        resolution: Tuple[int, int] = (20, 20),
        sigma: float = 0.1,
        weight_function: Optional[Callable] = None
    ) -> np.ndarray:
        """Generate a persistence image from the diagrams.
        
        Args:
            diagrams: List of persistence diagrams.
            resolution: Resolution of the persistence image.
            sigma: Bandwidth for the Gaussian kernel.
            weight_function: Custom weight function for the persistence image.
            
        Returns:
            Persistence image array with shape (n_dimensions, *resolution).
        """
        if not PERSIM_AVAILABLE:
            raise ImportError("Persim is required for persistence images.")
        
        images = []
        
        for dim, diagram in enumerate(diagrams):
            if diagram.shape[0] == 0:
                # No features in this dimension
                images.append(np.zeros(resolution))
                continue
            
            # Default weight function: linear weighting by persistence
            if weight_function is None:
                def weight_function(birth, death):
                    return death - birth
            
            # Generate persistence image
            pim = persim.PersistenceImager(
                pixels=resolution,
                weight_function=weight_function,
                kernel_params={"sigma": sigma}
            )
            
            image = pim.transform(diagram)
            images.append(image)
        
        return np.array(images)
    
    def betti_numbers(self, diagrams: List[np.ndarray], threshold: float) -> List[int]:
        """Compute Betti numbers at a specific threshold.
        
        Args:
            diagrams: List of persistence diagrams.
            threshold: Filtration value at which to compute Betti numbers.
            
        Returns:
            List of Betti numbers, one for each homology dimension.
        """
        betti = []
        
        for diagram in diagrams:
            if diagram.shape[0] == 0:
                betti.append(0)
                continue
            
            # Count features that are born before and die after the threshold
            birth = diagram[:, 0]
            death = diagram[:, 1]
            
            count = np.sum((birth <= threshold) & (death > threshold))
            betti.append(int(count))
        
        return betti
    
    def distance_matrix(
        self, 
        diagrams1: List[np.ndarray], 
        diagrams2: List[np.ndarray],
        metric: str = "wasserstein"
    ) -> np.ndarray:
        """Compute distance between persistence diagrams.
        
        Args:
            diagrams1: First list of persistence diagrams.
            diagrams2: Second list of persistence diagrams.
            metric: Distance metric ('wasserstein', 'bottleneck', or 'landscape').
            
        Returns:
            Distance matrix between diagrams.
        """
        if not PERSIM_AVAILABLE:
            raise ImportError("Persim is required for diagram distances.")
        
        n_diagrams1 = len(diagrams1)
        n_diagrams2 = len(diagrams2)
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_diagrams1, n_diagrams2))
        
        for i, diag1 in enumerate(diagrams1):
            for j, diag2 in enumerate(diagrams2):
                # Skip if both diagrams are empty
                if diag1.shape[0] == 0 and diag2.shape[0] == 0:
                    distance_matrix[i, j] = 0
                    continue
                
                # Handle case where one diagram is empty
                if diag1.shape[0] == 0 or diag2.shape[0] == 0:
                    # For Wasserstein, use infinity
                    if metric == "wasserstein":
                        distance_matrix[i, j] = np.inf
                    # For bottleneck, use infinity
                    elif metric == "bottleneck":
                        distance_matrix[i, j] = np.inf
                    # For landscape, use 0 (no features)
                    else:
                        distance_matrix[i, j] = 0
                    continue
                
                # Compute distance based on the metric
                if metric == "wasserstein":
                    distance_matrix[i, j] = persim.wasserstein(diag1, diag2)
                elif metric == "bottleneck":
                    distance_matrix[i, j] = persim.bottleneck(diag1, diag2)
                elif metric == "landscape":
                    # Landscape distance requires Giotto-TDA
                    if GIOTTO_AVAILABLE:
                        from giotto.diagrams import PersistenceLandscape
                        
                        # Convert to Giotto format
                        giotto_diag1 = np.zeros((diag1.shape[0], 3))
                        giotto_diag1[:, 0] = i  # homology dimension
                        giotto_diag1[:, 1:] = diag1  # birth-death pairs
                        
                        giotto_diag2 = np.zeros((diag2.shape[0], 3))
                        giotto_diag2[:, 0] = j  # homology dimension
                        giotto_diag2[:, 1:] = diag2  # birth-death pairs
                        
                        # Compute landscape distance
                        landscape = PersistenceLandscape(n_layers=5, n_bins=100)
                        landscape1 = landscape.fit_transform([giotto_diag1])
                        landscape2 = landscape.fit_transform([giotto_diag2])
                        
                        distance_matrix[i, j] = np.linalg.norm(landscape1 - landscape2)
                    else:
                        raise ImportError("Giotto-TDA required for landscape distance.")
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        return distance_matrix
    
    def plot_diagram(
        self, 
        diagrams: List[np.ndarray],
        title: str = "Persistence Diagram",
        ax: Optional[Any] = None,
        dimensions: Optional[List[int]] = None
    ) -> Any:
        """Plot persistence diagram.
        
        Args:
            diagrams: List of persistence diagrams.
            title: Title for the plot.
            ax: Matplotlib axis to plot on (optional).
            dimensions: Which dimensions to plot (default: all).
            
        Returns:
            Matplotlib axis with the plot.
        """
        if not PERSIM_AVAILABLE:
            raise ImportError("Persim is required for plotting diagrams.")
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        if dimensions is None:
            dimensions = list(range(len(diagrams)))
        
        # Plot the diagrams
        persim.plot_diagrams(
            [diagrams[dim] for dim in dimensions],
            labels=[f"H{dim}" for dim in dimensions],
            ax=ax
        )
        
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        return ax
