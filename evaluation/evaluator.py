"""
Main Evaluation Orchestrator
Integrates evaluation metrics with query templates for comprehensive testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import List, Dict, Optional

from .metrics import RAGEvaluationSystem
from query_template import UAMQueryTemplateLibrary, ResearchDomain


class UAMResearchEvaluator:
    """Integrated system for evaluating UAM RAG using domain-specific templates"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.evaluator = RAGEvaluationSystem(rag_system)
        self.template_library = UAMQueryTemplateLibrary()
        self.benchmark_results = []
    
    def run_domain_benchmark(self, domain: ResearchDomain, n_queries: int = 5) -> pd.DataFrame:
        """Run benchmark evaluation for a specific research domain"""
        print(f"\n{'='*60}")
        print(f"Running benchmark for domain: {domain.value}")
        print(f"{'='*60}\n")
        
        # Get templates for domain
        domain_templates = self.template_library.get_template_by_domain(domain)
        selected_templates = domain_templates[:n_queries]
        
        results = []
        for template in selected_templates:
            # Generate query
            query = template.example_query
            print(f"Testing: {query[:80]}...")
            
            # Evaluate query
            eval_result = self.evaluator.evaluate_query(query)
            
            # Validate against template criteria
            response, _ = self.rag_system.answer_literature_query(query)
            validation = self.template_library.validate_query_response(
                query, response, template.id
            )
            
            # Compile results
            result = {
                'domain': domain.value,
                'template_id': template.id,
                'query': query[:50] + '...',
                'overall_score': eval_result.overall_score,
                'retrieval_f1': np.mean(list(eval_result.retrieval_metrics.f1_at_k.values())),
                'generation_coherence': eval_result.generation_metrics.coherence_score,
                'template_validation_score': validation['score'],
                'missing_elements': len(validation['missing_elements']),
                'statistical_focus': template.statistical_focus
            }
            
            results.append(result)
            self.benchmark_results.append(result)
        
        return pd.DataFrame(results)
    
    def test_progressive_complexity(self, topic: str = "trust") -> pd.DataFrame:
        """Test system performance on progressively complex queries"""
        print(f"\n{'='*60}")
        print(f"Testing progressive complexity for topic: {topic}")
        print(f"{'='*60}\n")
        
        progressive_queries = self.template_library.get_progressive_queries(topic)
        results = []
        
        for complexity_level, (query_type, query) in enumerate(progressive_queries, 1):
            print(f"\nLevel {complexity_level} - {query_type}")
            print(f"Query: {query}")
            
            # Evaluate
            start_time = datetime.now()
            eval_result = self.evaluator.evaluate_query(query)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'complexity_level': complexity_level,
                'query_type': query_type,
                'query': query[:50] + '...',
                'overall_score': eval_result.overall_score,
                'execution_time': execution_time,
                'retrieval_coverage': eval_result.retrieval_metrics.coverage,
                'generation_completeness': eval_result.generation_metrics.completeness_score,
                'recommendations': len(eval_result.recommendations)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_statistical_reporting(self) -> pd.DataFrame:
        """Specifically evaluate statistical reporting accuracy"""
        print("\n" + "="*60)
        print("Evaluating Statistical Reporting Accuracy")
        print("="*60 + "\n")
        
        # Get all statistical templates
        stat_templates = self.template_library.get_statistical_templates()
        results = []
        
        for template in stat_templates[:10]:  # Test first 10
            query = template.example_query
            
            # Get response
            response, sources = self.rag_system.answer_literature_query(query)
            
            # Extract and verify statistics
            reported_stats = self.evaluator._extract_statistics(response)
            
            # Check statistical reporting quality
            stat_quality_metrics = {
                'has_coefficients': any(s['type'] == 'coefficient' for s in reported_stats),
                'has_p_values': any(s['type'] == 'p_value' for s in reported_stats),
                'has_effect_sizes': any(s['type'] == 'effect_size' for s in reported_stats),
                'has_sample_sizes': any(s['type'] == 'sample_size' for s in reported_stats),
                'total_statistics': len(reported_stats),
                'unique_stat_types': len(set(s['type'] for s in reported_stats))
            }
            
            result = {
                'template_id': template.id,
                'domain': template.domain.value,
                **stat_quality_metrics,
                'statistical_score': sum(stat_quality_metrics.values()) / 6  # Normalize
            }
            
            results.append(result)
        
        return pd.DataFrame(results)

