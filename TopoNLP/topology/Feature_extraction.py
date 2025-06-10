class TopologicalFeatureExtractor:
    def __init__(self, max_dimension=3):
        self.max_dimension = max_dimension
        
    def extract_persistent_homology_features(self, point_cloud, max_epsilon=1.0):
        """Extract persistence diagrams from text embeddings"""
        import ripser
        from ripser import ripser as rips
        
        # Compute Vietoris-Rips persistence
        diagrams = rips(point_cloud, maxdim=self.max_dimension, 
                       thresh=max_epsilon)['dgms']
        
        features = {}
        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                # Extract birth, death, and persistence values
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                persistence = deaths - births
                
                features[f'dim_{dim}'] = {
                    'births': births,
                    'deaths': deaths, 
                    'persistence': persistence,
                    'betti_number': len(diagram)
                }
        
        return features
    
    def extract_word_manifold_features(self, word_manifold):
        """Extract topological invariants from word manifolds"""
        features = {}
        
        # Betti numbers across dimensions
        for dim in word_manifold.skeleta:
            features[f'betti_{dim}'] = len(word_manifold.skeleta[dim])
        
        # Euler characteristic
        euler_char = sum((-1)**dim * features[f'betti_{dim}'] 
                        for dim in word_manifold.skeleta)
        features['euler_characteristic'] = euler_char
        
        # Hodge Laplacian spectral features
        hodge_lap = HodgeLaplacian(word_manifold)
        for dim in range(min(3, len(word_manifold.skeleta))):
            laplacian = hodge_lap.compute_hodge_laplacian(dim)
            eigenvals = np.linalg.eigvals(laplacian)
            features[f'hodge_spectrum_{dim}'] = eigenvals
        
        return features
