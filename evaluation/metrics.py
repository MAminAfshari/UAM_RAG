# evaluation/metrics.py
"""
Core Evaluation Metrics
Comprehensive metrics for evaluating RAG system performance.
"""
import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality"""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    map: float = 0.0  # Mean Average Precision
    coverage: float = 0.0  # Percentage of corpus accessed
    diversity: float = 0.0  # Diversity of retrieved sources


@dataclass
class GenerationMetrics:
    """Metrics for response generation quality"""
    citation_accuracy: float = 0.0
    citation_coverage: float = 0.0
    statistical_accuracy: float = 0.0
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    factual_consistency: float = 0.0


@dataclass
class TemporalMetrics:
    """Metrics for temporal analysis quality"""
    trend_accuracy: float = 0.0
    temporal_coverage: float = 0.0
    change_point_precision: float = 0.0
    period_balance: float = 0.0


@dataclass
class SynthesisMetrics:
    """Metrics for synthesis and contradiction detection"""
    contradiction_precision: float = 0.0
    contradiction_recall: float = 0.0
    consensus_quality: float = 0.0
    clustering_quality: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    query: str
    timestamp: datetime
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    temporal_metrics: Optional[TemporalMetrics]
    synthesis_metrics: Optional[SynthesisMetrics]
    overall_score: float
    recommendations: List[str]


class RAGEvaluationSystem:
    """Comprehensive evaluation system for UAM Literature Review RAG"""
    
    def __init__(self, rag_system, ground_truth_path: Optional[str] = None):
        self.rag_system = rag_system
        self.ground_truth = self._load_ground_truth(ground_truth_path) if ground_truth_path else {}
        self.evaluation_history = []
        
        # Evaluation weights for overall score
        self.weights = {
            'retrieval': 0.3,
            'generation': 0.4,
            'temporal': 0.15,
            'synthesis': 0.15
        }
    
    def _load_ground_truth(self, path: str) -> Dict:
        """Load ground truth data for evaluation"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def evaluate_query(self, 
                      query: str, 
                      expected_papers: Optional[List[str]] = None,
                      expected_findings: Optional[List[Dict]] = None,
                      evaluate_all: bool = True) -> EvaluationResult:
        """Evaluate a single query comprehensively"""
        
        # Execute query
        start_time = datetime.now()
        retrieved_docs = self.rag_system.enhanced_retrieval(query)
        response, sources = self.rag_system.answer_literature_query(query)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate retrieval
        retrieval_metrics = self._evaluate_retrieval(
            query, retrieved_docs, expected_papers
        )
        
        # Evaluate generation
        generation_metrics = self._evaluate_generation(
            response, sources, retrieved_docs, expected_findings
        )
        
        # Evaluate temporal analysis if applicable
        temporal_metrics = None
        if evaluate_all and self._is_temporal_query(query):
            temporal_metrics = self._evaluate_temporal_analysis(query, response)
        
        # Evaluate synthesis if applicable
        synthesis_metrics = None
        if evaluate_all:
            synthesis_metrics = self._evaluate_synthesis(query, response)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            retrieval_metrics, generation_metrics, 
            temporal_metrics, synthesis_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            retrieval_metrics, generation_metrics,
            temporal_metrics, synthesis_metrics
        )
        
        result = EvaluationResult(
            query=query,
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            temporal_metrics=temporal_metrics,
            synthesis_metrics=synthesis_metrics,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        self.evaluation_history.append(result)
        return result
    
    def _evaluate_retrieval(self, 
                           query: str, 
                           retrieved_docs: List,
                           expected_papers: Optional[List[str]] = None) -> RetrievalMetrics:
        """Evaluate retrieval quality"""
        metrics = RetrievalMetrics()
        
        # Get retrieved paper IDs
        retrieved_papers = [doc.metadata.get('source') for doc in retrieved_docs]
        unique_papers = list(set(retrieved_papers))
        
        if expected_papers:
            # Calculate precision, recall, F1 at different k values
            for k in [5, 10, 20]:
                retrieved_at_k = set(retrieved_papers[:k])
                expected_set = set(expected_papers)
                
                precision = len(retrieved_at_k & expected_set) / k if k > 0 else 0
                recall = len(retrieved_at_k & expected_set) / len(expected_set) if expected_set else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.precision_at_k[k] = precision
                metrics.recall_at_k[k] = recall
                metrics.f1_at_k[k] = f1
            
            # Calculate MRR
            for i, paper in enumerate(retrieved_papers):
                if paper in expected_papers:
                    metrics.mrr = 1 / (i + 1)
                    break
        
        # Calculate coverage
        try:
            total_papers = len(self.rag_system.get_corpus_statistics().get('papers_processed', []))
            metrics.coverage = len(unique_papers) / total_papers if total_papers > 0 else 0
        except:
            metrics.coverage = 0.0
        
        # Diversity score
        metrics.diversity = len(unique_papers) / len(retrieved_papers) if retrieved_papers else 0
        
        return metrics
    
    def _evaluate_generation(self,
                           response: str,
                           sources: List[str],
                           retrieved_docs: List,
                           expected_findings: Optional[List[Dict]] = None) -> GenerationMetrics:
        """Evaluate generation quality"""
        metrics = GenerationMetrics()
        
        # Citation accuracy
        citations_in_response = self._extract_citations(response)
        metrics.citation_accuracy = self._calculate_citation_accuracy(
            citations_in_response, sources
        )
        
        # Citation coverage
        metrics.citation_coverage = len(citations_in_response) / len(sources) if sources else 0
        
        # Statistical accuracy
        stats_in_response = self._extract_statistics(response)
        metrics.statistical_accuracy = min(1.0, len(stats_in_response) / 3)  # Normalize
        
        # Completeness score
        if expected_findings:
            metrics.completeness_score = self._calculate_completeness(
                response, expected_findings
            )
        else:
            metrics.completeness_score = 0.7  # Default reasonable score
        
        # Coherence score
        metrics.coherence_score = self._calculate_coherence(response)
        
        # Factual consistency
        metrics.factual_consistency = self._calculate_factual_consistency(
            response, retrieved_docs
        )
        
        return metrics
    
    def _evaluate_temporal_analysis(self, query: str, response: str) -> TemporalMetrics:
        """Evaluate temporal analysis quality"""
        metrics = TemporalMetrics()
        
        # Extract temporal claims
        temporal_claims = self._extract_temporal_claims(response)
        
        # Temporal coverage
        years_mentioned = self._extract_years(response)
        if years_mentioned:
            year_range = max(years_mentioned) - min(years_mentioned)
            metrics.temporal_coverage = min(1.0, year_range / 20)  # Normalize by 20 years
        
        # Period balance
        metrics.period_balance = self._calculate_period_balance(years_mentioned)
        
        return metrics
    
    def _evaluate_synthesis(self, query: str, response: str) -> SynthesisMetrics:
        """Evaluate synthesis quality"""
        metrics = SynthesisMetrics()
        
        # Consensus quality
        consensus_statements = self._extract_consensus_statements(response)
        metrics.consensus_quality = self._evaluate_consensus_quality(consensus_statements)
        
        # Clustering quality
        metrics.clustering_quality = self._evaluate_clustering_quality(response)
        
        return metrics
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from response text"""
        pattern = r'\[([^\]]+)\]'
        citations = re.findall(pattern, text)
        return [c for c in citations if not c.isdigit()]
    
    def _extract_statistics(self, text: str) -> List[Dict]:
        """Extract statistical values from text"""
        statistics = []
        
        patterns = {
            'coefficient': r'(?:β|beta)\s*=\s*([-+]?\d*\.?\d+)',
            'p_value': r'p\s*[<>=]\s*(\d*\.?\d+)',
            'effect_size': r'd\s*=\s*([-+]?\d*\.?\d+)',
            'r_squared': r'R[²2]\s*=\s*(\d*\.?\d+)',
            'sample_size': r'[nN]\s*=\s*(\d+)'
        }
        
        for stat_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match) if stat_type != 'sample_size' else int(match)
                    statistics.append({
                        'type': stat_type,
                        'value': value
                    })
                except ValueError:
                    continue
        
        return statistics
    
    def _calculate_citation_accuracy(self, citations: List[str], sources: List[str]) -> float:
        """Calculate citation accuracy"""
        if not citations:
            return 0.0
        
        valid_citations = sum(1 for c in citations if c in sources)
        return valid_citations / len(citations)
    
    def _calculate_completeness(self, response: str, expected_findings: List[Dict]) -> float:
        """Calculate completeness score"""
        if not expected_findings:
            return 0.7  # Default
        
        covered_findings = 0
        for finding in expected_findings:
            key_terms = finding.get('key_terms', [])
            if all(term.lower() in response.lower() for term in key_terms):
                covered_findings += 1
        
        return covered_findings / len(expected_findings)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score"""
        score = 0.5  # Base score
        
        # Check for structured sections
        if any(marker in response for marker in ['First,', 'Second,', 'Finally,', '1.', '2.']):
            score += 0.1
        
        # Check for transition words
        transitions = ['however', 'moreover', 'furthermore', 'additionally', 'similarly', 
                      'in contrast', 'on the other hand', 'consequently']
        transition_count = sum(1 for t in transitions if t in response.lower())
        score += min(0.2, transition_count * 0.05)
        
        # Check for balanced paragraphs
        paragraphs = response.split('\n\n')
        if len(paragraphs) >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_factual_consistency(self, response: str, docs: List) -> float:
        """Calculate factual consistency"""
        if not docs:
            return 0.5
        
        response_sentences = response.split('.')
        doc_content = ' '.join([doc.page_content for doc in docs])
        
        consistent_claims = 0
        total_claims = 0
        
        for sentence in response_sentences:
            if len(sentence.strip()) > 20:
                total_claims += 1
                key_terms = [word for word in sentence.split() 
                           if len(word) > 4 and word.isalpha()]
                if key_terms:
                    matching_terms = sum(1 for term in key_terms 
                                       if term.lower() in doc_content.lower())
                    if matching_terms / len(key_terms) > 0.3:
                        consistent_claims += 1
        
        return consistent_claims / total_claims if total_claims > 0 else 0.5
    
    def _extract_temporal_claims(self, text: str) -> List[str]:
        """Extract temporal claims"""
        temporal_patterns = [
            r'(?:increased|decreased|changed|evolved|grew|declined)\s+(?:over|from|since|between)',
            r'trend[s]?\s+(?:show|indicate|suggest)',
            r'(?:early|recent|later)\s+studies'
        ]
        
        claims = []
        for pattern in temporal_patterns:
            matches = re.findall(f'([^.]*{pattern}[^.]*)', text, re.IGNORECASE)
            claims.extend(matches)
        
        return claims
    
    def _extract_years(self, text: str) -> List[int]:
        """Extract years from text"""
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        return [int(f"{prefix}{year[2:]}") for prefix, year in 
                zip(years[::2], years[1::2])] if len(years) % 2 == 0 else []
    
    def _calculate_period_balance(self, years: List[int]) -> float:
        """Calculate period balance"""
        if not years:
            return 0.0
        
        periods = {'early': 0, 'middle': 0, 'recent': 0}
        
        for year in years:
            if year <= 2010:
                periods['early'] += 1
            elif year <= 2017:
                periods['middle'] += 1
            else:
                periods['recent'] += 1
        
        total = sum(periods.values())
        if total == 0:
            return 0.0
        
        proportions = [count / total for count in periods.values()]
        return 1 - np.std(proportions) * 3  # Balance metric
    
    def _extract_consensus_statements(self, text: str) -> List[str]:
        """Extract consensus statements"""
        consensus_patterns = [
            r'(?:consensus|agreement|consistent)\s+(?:across|among|between)\s+studies',
            r'(?:all|most|majority)\s+(?:studies|research|literature)\s+(?:agree|confirm|support)',
            r'(?:widely|generally|commonly)\s+(?:accepted|supported|confirmed)'
        ]
        
        statements = []
        for pattern in consensus_patterns:
            matches = re.findall(f'([^.]*{pattern}[^.]*)', text, re.IGNORECASE)
            statements.extend(matches)
        
        return statements
    
    def _evaluate_consensus_quality(self, statements: List[str]) -> float:
        """Evaluate consensus quality"""
        if not statements:
            return 0.0
        
        quality_score = 0.0
        
        for statement in statements:
            if re.search(r'\d+\s*(?:studies|papers|%)', statement):
                quality_score += 0.3
            if re.search(r'(?:p\s*[<>=]|β\s*=|effect)', statement, re.IGNORECASE):
                quality_score += 0.4
            if len(statement.split()) > 15:
                quality_score += 0.3
        
        return min(1.0, quality_score / len(statements))
    
    def _evaluate_clustering_quality(self, response: str) -> float:
        """Evaluate clustering quality"""
        grouping_patterns = [
            r'(?:first|second|third)\s+(?:group|cluster|category)',
            r'can\s+be\s+(?:grouped|categorized|classified)',
            r'(?:type|kind|category)\s+(?:of|includes)'
        ]
        
        grouping_count = sum(1 for pattern in grouping_patterns 
                           if re.search(pattern, response, re.IGNORECASE))
        
        return min(1.0, grouping_count * 0.25)
    
    def _calculate_overall_score(self, 
                               retrieval: RetrievalMetrics,
                               generation: GenerationMetrics,
                               temporal: Optional[TemporalMetrics],
                               synthesis: Optional[SynthesisMetrics]) -> float:
        """Calculate weighted overall score"""
        scores = {}
        
        # Retrieval score
        retrieval_scores = []
        if retrieval.f1_at_k:
            retrieval_scores.append(np.mean(list(retrieval.f1_at_k.values())))
        if retrieval.coverage:
            retrieval_scores.append(retrieval.coverage)
        if retrieval.diversity:
            retrieval_scores.append(retrieval.diversity)
        scores['retrieval'] = np.mean(retrieval_scores) if retrieval_scores else 0.5
        
        # Generation score
        generation_scores = [
            generation.citation_accuracy,
            generation.statistical_accuracy,
            generation.coherence_score,
            generation.factual_consistency,
            generation.completeness_score
        ]
        scores['generation'] = np.mean(generation_scores)
        
        # Temporal score
        if temporal:
            temporal_scores = [
                temporal.temporal_coverage,
                temporal.period_balance
            ]
            scores['temporal'] = np.mean(temporal_scores)
        
        # Synthesis score
        if synthesis:
            synthesis_scores = [
                synthesis.consensus_quality,
                synthesis.clustering_quality
            ]
            scores['synthesis'] = np.mean(synthesis_scores)
        
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        
        for component, score in scores.items():
            weight = self.weights.get(component, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _generate_recommendations(self,
                                retrieval: RetrievalMetrics,
                                generation: GenerationMetrics,
                                temporal: Optional[TemporalMetrics],
                                synthesis: Optional[SynthesisMetrics]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Retrieval recommendations
        if retrieval.coverage < 0.1:
            recommendations.append("Low corpus coverage - expand search parameters")
        if retrieval.diversity < 0.5:
            recommendations.append("Low source diversity - improve ranking algorithm")
        
        # Generation recommendations
        if generation.citation_accuracy < 0.8:
            recommendations.append("Citation accuracy low - verify citation extraction")
        if generation.statistical_accuracy < 0.5:
            recommendations.append("Statistical reporting needs improvement")
        if generation.coherence_score < 0.6:
            recommendations.append("Response structure could be more coherent")
        
        # Temporal recommendations
        if temporal and temporal.temporal_coverage < 0.5:
            recommendations.append("Limited temporal coverage - include more time periods")
        
        # Synthesis recommendations
        if synthesis and synthesis.consensus_quality < 0.5:
            recommendations.append("Consensus statements lack supporting evidence")
        
        return recommendations
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query involves temporal analysis"""
        temporal_keywords = [
            'temporal', 'trend', 'evolution', 'change', 'over time',
            'timeline', 'historical', 'progression', 'development'
        ]
        return any(keyword in query.lower() for keyword in temporal_keywords)
