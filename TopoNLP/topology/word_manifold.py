class WordManifold:
    def __init__(self, corpus, window_size=5):
        self.corpus = corpus
        self.window_size = window_size
        self.skeleta = {}
        self.boundary_matrices = {}
        
    def construct_skeleta(self):
        """Generate n-skeleta from n-gram patterns"""
        # 0-skeleton: lexicon extraction
        self.skeleta[0] = set(word for sentence in self.corpus 
                             for word in sentence)
        
        # Higher-dimensional skeleta from n-grams
        for n in range(1, self.window_size):
            self.skeleta[n] = self._extract_ngram_simplices(n)
    
    def _extract_ngram_simplices(self, n):
        """Extract (n+1)-gram simplices satisfying subsequence conditions"""
        simplices = set()
        for sentence in self.corpus:
            for i in range(len(sentence) - n):
                ngram = tuple(sentence[i:i+n+1])
                if self._validate_simplex(ngram):
                    simplices.add(ngram)
        return simplices
