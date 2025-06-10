class PersistenceDiagramProcessor:
    def __init__(self, resolution=50):
        self.resolution = resolution
        
    def compute_persistence_diagrams(self, embeddings):
        """Compute persistence diagrams from text embeddings"""
        from ripser import ripser
        
        # Compute Vietoris-Rips filtration
        result = ripser(embeddings, maxdim=2)
        diagrams = result['dgms']
        
        return diagrams
    
    def vectorize_persistence_diagram(self, diagram):
        """Convert persistence diagram to feature vector"""
        if len(diagram) == 0:
            return np.zeros(self.resolution * self.resolution)
        
        # Create persistence image
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistence = deaths - births
        
        # Discretize into grid
        birth_range = (births.min(), births.max()) if len(births) > 0 else (0, 1)
        death_range = (deaths.min(), deaths.max()) if len(deaths) > 0 else (0, 1)
        
        image = np.zeros((self.resolution, self.resolution))
        
        for birth, death, pers in zip(births, deaths, persistence):
            # Map to grid coordinates
            birth_idx = int((birth - birth_range[0]) / 
                           (birth_range[1] - birth_range[0]) * 
                           (self.resolution - 1))
            death_idx = int((death - death_range[0]) / 
                           (death_range[1] - death_range[0]) * 
                           (self.resolution - 1))
            
            birth_idx = max(0, min(birth_idx, self.resolution - 1))
            death_idx = max(0, min(death_idx, self.resolution - 1))
            
            # Weight by persistence
            image[birth_idx, death_idx] += pers
        
        return image.flatten()
    
    def extract_topological_signatures(self, text_embeddings):
        """Extract comprehensive topological signatures"""
        diagrams = self.compute_persistence_diagrams(text_embeddings)
        
        signatures = {}
        for dim, diagram in enumerate(diagrams):
            signatures[f'persistence_image_{dim}'] = self.vectorize_persistence_diagram(diagram)
            
            if len(diagram) > 0:
                signatures[f'total_persistence_{dim}'] = np.sum(diagram[:, 1] - diagram[:, 0])
                signatures[f'max_persistence_{dim}'] = np.max(diagram[:, 1] - diagram[:, 0])
                signatures[f'num_features_{dim}'] = len(diagram)
        
        return signatures
