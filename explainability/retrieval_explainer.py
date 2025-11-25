import numpy as np
from collections import Counter

class RetrievalExplainer:
    """Explains why specific documents were retrieved."""
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def explain_retrieval(self, query, retrieved_docs, scores):
        """
        Explain why documents were retrieved.
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved document dicts
            scores: List of similarity scores
            
        Returns:
            List of explanations for each retrieved document
        """
        query_words = set(query.lower().split())
        explanations = []
        
        for doc, score in zip(retrieved_docs, scores):
            doc_text = doc['text'].lower()
            doc_words = set(doc_text.split())
            
            # Find overlapping keywords
            overlap = query_words.intersection(doc_words)
            
            # Calculate keyword coverage
            coverage = len(overlap) / len(query_words) if query_words else 0
            
            # Extract key matching phrases
            matching_phrases = self._find_matching_phrases(query, doc_text)
            
            explanations.append({
                'document': doc,
                'score': float(score),
                'overlapping_keywords': list(overlap),
                'keyword_coverage': float(coverage),
                'matching_phrases': matching_phrases,
                'explanation': self._generate_explanation(overlap, score, coverage)
            })
        
        return explanations
    
    def _find_matching_phrases(self, query, doc_text, context_window=5):
        """Find phrases in document that match query terms."""
        query_words = query.lower().split()
        doc_words = doc_text.split()
        
        matching_phrases = []
        
        for q_word in query_words:
            for i, d_word in enumerate(doc_words):
                if q_word in d_word.lower():
                    # Extract context around match
                    start = max(0, i - context_window)
                    end = min(len(doc_words), i + context_window + 1)
                    phrase = ' '.join(doc_words[start:end])
                    matching_phrases.append({
                        'keyword': q_word,
                        'context': phrase,
                        'position': i
                    })
        
        return matching_phrases[:3]  # Limit to top 3
    
    def _generate_explanation(self, overlap, score, coverage):
        """Generate human-readable explanation."""
        if score > 0.7:
            strength = "strong"
        elif score > 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        if overlap:
            keywords_str = ', '.join(list(overlap)[:3])
            return f"This document has a {strength} match (score: {score:.2f}) based on keywords: {keywords_str}"
        else:
            return f"This document has a {strength} semantic match (score: {score:.2f}) based on meaning similarity"
    
    def get_confidence_breakdown(self, scores):
        """Break down confidence into components."""
        if not scores:
            return {
                'overall': 0.0,
                'top_match': 0.0,
                'consistency': 0.0,
                'interpretation': 'No matches found'
            }
        
        top_score = max(scores)
        avg_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        
        # Consistency: lower std = more consistent
        consistency = 1.0 - min(std_score, 1.0)
        
        overall = (top_score * 0.6) + (avg_score * 0.3) + (consistency * 0.1)
        
        if overall > 0.7:
            interpretation = "High confidence - strong relevant matches found"
        elif overall > 0.5:
            interpretation = "Moderate confidence - relevant information available"
        else:
            interpretation = "Low confidence - limited relevant information"
        
        return {
            'overall': float(overall),
            'top_match': float(top_score),
            'average_match': float(avg_score),
            'consistency': float(consistency),
            'interpretation': interpretation
        }
