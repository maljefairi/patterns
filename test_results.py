import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json

class TestPatternAnalysis(unittest.TestCase):
    def setUp(self):
        # Load test dataset and benchmark results
        self.test_data = pd.read_csv('thinking_patterns.csv')
        with open('benchmark_results.json', 'r') as f:
            self.benchmark_results = json.load(f)
        
    def test_pattern_generation_quality(self):
        """Test if generated patterns meet quality criteria"""
        # Check that patterns are non-empty strings
        self.assertTrue(all(isinstance(p, str) and len(p) > 0 
                          for p in self.test_data['Pattern']))
        
        # Check that patterns contain expected AGI/architectural keywords
        agi_keywords = ['intelligence', 'learning', 'cognition', 'reasoning', 'adaptation',
                       'emergence', 'consciousness', 'scalability', 'architecture']
        patterns_with_keywords = 0
        for pattern in self.test_data['Pattern']:
            if any(keyword in pattern.lower() for keyword in agi_keywords):
                patterns_with_keywords += 1
        keyword_rate = patterns_with_keywords / len(self.test_data)
        self.assertGreater(keyword_rate, 0.7, "Patterns should contain AGI/architectural keywords")
        
    def test_vector_quality(self):
        """Test vector quality metrics from benchmarks"""
        vectors = []
        
        # Safely evaluate string representations of vectors
        for v in self.test_data['Vector']:
            try:
                if isinstance(v, str):
                    vector = eval(v)
                else:
                    vector = v
                vectors.append(vector)
            except:
                print(f"Warning: Failed to parse vector: {v[:100]}...")
                continue
        
        # Print vector statistics for debugging
        lengths = [len(v) for v in vectors]
        print(f"\nVector length statistics:")
        print(f"Unique lengths found: {sorted(set(lengths))}")
        print(f"Length counts: {pd.Series(lengths).value_counts().to_dict()}")
        
        # Test if majority of vectors have consistent length
        most_common_length = max(set(lengths), key=lengths.count)
        consistent_vectors = sum(1 for l in lengths if l == most_common_length)
        consistency_rate = consistent_vectors / len(vectors)
        
        self.assertGreater(consistency_rate, 0.9, 
                          f"Most vectors should have consistent length of {most_common_length}")
        
    def test_pattern_benchmarks(self):
        """Test pattern benchmark scores"""
        # Calculate average scores for each benchmark dimension
        agi_scores = np.mean([b['agi_alignment'] for b in self.benchmark_results])
        arch_scores = np.mean([b['architectural_coherence'] for b in self.benchmark_results])
        web_scores = np.mean([b['web_recognition'] for b in self.benchmark_results])
        
        # Check benchmark thresholds based on target accuracy of 0.8
        self.assertGreater(agi_scores, 0.7, "AGI alignment score below threshold")
        self.assertGreater(arch_scores, 0.7, "Architectural coherence score below threshold")
        self.assertGreater(web_scores, 0.7, "Web recognition score below threshold")
                
    def test_timestamp_validity(self):
        """Test timestamp formatting and validity"""
        timestamps = pd.to_datetime(self.test_data['Timestamp'])
        
        # Check timestamps are in UTC
        self.assertTrue(all(ts.tzinfo is not None for ts in timestamps))
        
        # Check timestamps are not in future
        now = datetime.now(timezone.utc)
        self.assertTrue(all(ts <= now for ts in timestamps))
        
    def test_llm_analysis_quality(self):
        """Test LLM analysis content"""
        analyses = self.test_data['LLM_Analysis']
        
        # Check analyses contain expected evaluation dimensions
        required_aspects = ['cognitive architecture', 'intelligence emergence',
                          'scaling characteristics', 'safety', 'ethical considerations']
        
        for analysis in analyses:
            analysis_lower = analysis.lower()
            matches = sum(1 for aspect in required_aspects 
                        if aspect in analysis_lower)
            self.assertGreater(matches / len(required_aspects), 0.6,
                             "Analysis missing key evaluation aspects")

if __name__ == '__main__':
    unittest.main()
