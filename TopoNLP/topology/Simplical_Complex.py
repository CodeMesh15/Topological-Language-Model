class SimplicialComplex:
    def __init__(self):
        self.skeleta = {}  # Dictionary mapping dimension to simplices
        self.orientations = {}
        
    def add_simplex(self, simplex, dimension=None):
        """Add a simplex and all its faces to the complex"""
        if dimension is None:
            dimension = len(simplex) - 1
            
        # Add the simplex itself
        if dimension not in self.skeleta:
            self.skeleta[dimension] = set()
        self.skeleta[dimension].add(tuple(sorted(simplex)))
        
        # Add all faces recursively
        if dimension > 0:
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i+1:]
                self.add_simplex(face, dimension - 1)
    
    def from_text_corpus(self, corpus, window_size=3):
        """Construct simplicial complex from text using sliding windows"""
        for sentence in corpus:
            for i in range(len(sentence) - window_size + 1):
                window = sentence[i:i + window_size]
                # Create simplex from co-occurring words
                self.add_simplex(window)
        
        return self
    
    def get_betti_numbers(self):
        """Compute Betti numbers (topological invariants)"""
        betti_numbers = {}
        for dim in self.skeleta:
            # Simplified computation - full implementation requires
            # homology computation using boundary matrices
            betti_numbers[dim] = len(self.skeleta[dim])
        return betti_numbers
