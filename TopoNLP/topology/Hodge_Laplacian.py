class HodgeLaplacian:
    def __init__(self, simplicial_complex):
        self.complex = simplicial_complex
        self.boundary_matrices = {}
        self.laplacians = {}
    
    def compute_boundary_matrices(self):
        """Compute boundary operators ∂_p for each dimension"""
        for p in range(len(self.complex.skeleta)):
            self.boundary_matrices[p] = self._build_boundary_matrix(p)
    
    def _build_boundary_matrix(self, p):
        """Build p-th boundary matrix B_p"""
        if p == 0:
            return np.zeros((1, len(self.complex.skeleta[0])))
        
        p_simplices = list(self.complex.skeleta[p])
        p_minus_1_simplices = list(self.complex.skeleta[p-1])
        
        boundary_matrix = np.zeros((len(p_minus_1_simplices), len(p_simplices)))
        
        for j, simplex in enumerate(p_simplices):
            for i, face in enumerate(self._get_faces(simplex)):
                if face in p_minus_1_simplices:
                    face_idx = p_minus_1_simplices.index(face)
                    # Orientation coefficient (-1)^i
                    boundary_matrix[face_idx, j] = (-1) ** i
        
        return boundary_matrix
    
    def compute_hodge_laplacian(self, p):
        """Compute p-th Hodge Laplacian: L_p = B_p^T B_p + B_{p+1} B_{p+1}^T"""
        B_p = self.boundary_matrices.get(p, np.zeros((1, 1)))
        B_p_plus_1 = self.boundary_matrices.get(p+1, np.zeros((1, 1)))
        
        # Lower adjacency Laplacian
        L_lower = B_p.T @ B_p
        
        # Upper adjacency Laplacian  
        L_upper = B_p_plus_1 @ B_p_plus_1.T
        
        return L_lower + L_upper
