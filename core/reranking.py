# core/reranking.py
"""
Enhanced Re-ranking Engine Module with Journal Impact Integration
Re-ranks retrieved documents with journal impact awareness, quality weighting,
paper diversity enforcement, and comprehensive coverage optimization.
"""

import logging
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import numpy as np
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from config import Config

logger = logging.getLogger(__name__)


class ReRankingEngine:
    """Enhanced re-ranking engine with journal impact integration"""
    
    def __init__(self, journal_manager=None):
        """Initialize enhanced re-ranking model with journal impact support"""
        logger.info(f"Initializing enhanced re-ranking model: {Config.RERANKER_MODEL}")
        
        try:
            self.reranker = CrossEncoder(Config.RERANKER_MODEL)
            logger.info("Re-ranking model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize re-ranking model: {e}")
            self.reranker = None
        
        # Journal impact manager
        self.journal_manager = journal_manager
        self.journal_impact_enabled = bool(journal_manager)
        
        # Reranking statistics
        self.reranking_stats = {
            'total_documents_processed': 0,
            'documents_reranked': 0,
            'quality_tier_distribution': defaultdict(int),
            'paper_diversity_enforced': 0,
            'statistical_content_boosted': 0
        }
        
        logger.info(f"Enhanced re-ranking engine initialized with journal impact: {self.journal_impact_enabled}")
    
    def rerank(self, query: str, documents: List[Document], chapter_topic: str = None) -> List[Document]:
        """
        Enhanced re-rank documents with journal impact, quality weighting, and diversity enforcement
        
        Args:
            query: The search query
            documents: List of documents to re-rank
            chapter_topic: Optional chapter topic for specialized ranking
            
        Returns:
            List of re-ranked documents with comprehensive coverage
        """
        if not documents:
            return []
        
        logger.info(f"Re-ranking {len(documents)} documents with enhanced quality-aware algorithm")
        
        # Update statistics
        self.reranking_stats['total_documents_processed'] += len(documents)
        
        # Phase 1: Calculate base relevance scores
        scored_docs = self._calculate_base_scores(query, documents)
        
        # Phase 2: Apply journal impact and quality boosts
        if self.journal_impact_enabled:
            scored_docs = self._apply_journal_impact_boosts(scored_docs, query, chapter_topic)
        
        # Phase 3: Apply section and content boosts
        scored_docs = self._apply_content_boosts(scored_docs, chapter_topic)
        
        # Phase 4: Enforce paper diversity and quality tier representation
        final_docs = self._enforce_diversity_and_coverage(scored_docs, query)
        
        # Phase 5: Final ranking with comprehensive coverage
        final_ranking = self._final_ranking_with_coverage(final_docs)
        
        # Update statistics
        self.reranking_stats['documents_reranked'] = len(final_ranking)
        self._update_ranking_stats(final_ranking)
        
        logger.info(f"Re-ranking complete: {len(final_ranking)} documents with enhanced quality weighting")
        
        return final_ranking
    
    def _calculate_base_scores(self, query: str, documents: List[Document]) -> List[Tuple[float, Document, Dict]]:
        """Calculate base relevance scores for documents"""
        scored_docs = []
        
        for doc in documents:
            # Calculate base relevance score
            if self.reranker:
                try:
                    base_score = self.reranker.predict([(query, doc.page_content)])[0]
                except Exception as e:
                    logger.warning(f"Reranker prediction failed: {e}")
                    base_score = 0.5  # Default score
            else:
                base_score = 0.5  # Default when reranker not available
            
            # Create scoring metadata
            scoring_metadata = {
                'base_relevance_score': base_score,
                'document_id': doc.metadata.get('chunk_id', 'unknown'),
                'source': doc.metadata.get('source', 'unknown'),
                'section_type': doc.metadata.get('section_type', 'other'),
                'has_statistics': doc.metadata.get('has_statistics', False),
                'boosts_applied': []
            }
            
            scored_docs.append((base_score, doc, scoring_metadata))
        
        return scored_docs
    
    def _apply_journal_impact_boosts(self, scored_docs: List[Tuple[float, Document, Dict]], 
                                   query: str, chapter_topic: str = None) -> List[Tuple[float, Document, Dict]]:
        """Apply journal impact and quality boosts to scores"""
        enhanced_scored_docs = []
        
        for score, doc, metadata in scored_docs:
            enhanced_score = score
            
            # Get journal impact information
            journal_quality_score = doc.metadata.get('journal_quality_score', 0.2)
            journal_tier = doc.metadata.get('journal_quality_tier', 'unknown')
            journal_boost_multiplier = doc.metadata.get('journal_boost_multiplier', 0.2)
            
            # Apply journal impact boost
            if self.journal_impact_enabled:
                journal_boost = self._calculate_journal_impact_boost(
                    journal_quality_score, journal_tier, doc.metadata
                )
                enhanced_score += journal_boost
                metadata['boosts_applied'].append(f"journal_impact: +{journal_boost:.3f}")
            
            # Apply quality tier boost
            tier_boost = self._calculate_quality_tier_boost(journal_tier, doc.metadata)
            enhanced_score += tier_boost
            metadata['boosts_applied'].append(f"quality_tier: +{tier_boost:.3f}")
            
            # Apply chapter-specific journal boost
            if chapter_topic:
                chapter_boost = self._calculate_chapter_journal_boost(
                    chapter_topic, journal_tier, doc.metadata
                )
                enhanced_score += chapter_boost
                metadata['boosts_applied'].append(f"chapter_journal: +{chapter_boost:.3f}")
            
            # Update metadata
            metadata['journal_quality_score'] = journal_quality_score
            metadata['journal_tier'] = journal_tier
            metadata['journal_boost_multiplier'] = journal_boost_multiplier
            metadata['enhanced_score'] = enhanced_score
            
            enhanced_scored_docs.append((enhanced_score, doc, metadata))
        
        return enhanced_scored_docs
    
    def _calculate_journal_impact_boost(self, quality_score: float, tier: str, doc_metadata: Dict) -> float:
        """Calculate journal impact boost based on quality metrics"""
        # Base journal impact boost
        base_boost = quality_score * Config.JOURNAL_IMPACT_WEIGHT
        
        # Tier-specific multipliers
        tier_multiplier = Config.QUALITY_TIER_WEIGHTS.get(tier, 0.2)
        
        # Section-specific boost for high-quality journals
        section_type = doc_metadata.get('section_type', 'other')
        section_boost = Config.JOURNAL_SECTION_BOOSTS.get(section_type, 0.0)
        
        # Combined boost
        final_boost = base_boost * tier_multiplier + section_boost
        
        return final_boost
    
    def _calculate_quality_tier_boost(self, tier: str, doc_metadata: Dict) -> float:
        """Calculate quality tier boost"""
        # Base tier boost
        tier_weight = Config.QUALITY_TIER_WEIGHTS.get(tier, 0.2)
        
        # Higher boost for results and conclusions from high-tier journals
        section_type = doc_metadata.get('section_type', 'other')
        if tier in ['top', 'high'] and section_type in ['results', 'findings', 'conclusion']:
            return tier_weight * 0.3
        elif tier in ['medium'] and section_type in ['results', 'findings']:
            return tier_weight * 0.2
        else:
            return tier_weight * 0.1
    
    def _calculate_chapter_journal_boost(self, chapter_topic: str, tier: str, doc_metadata: Dict) -> float:
        """Calculate chapter-specific journal boost"""
        if tier in ['top', 'high']:
            # Higher boost for high-quality journals in specific chapters
            chapter_boosts = {
                'core_determinants': 0.15,
                'trust_risk_safety': 0.12,
                'affect_emotion': 0.10,
                'contextual_demographic': 0.08
            }
            return chapter_boosts.get(chapter_topic, 0.05)
        else:
            return 0.0
    
    def _apply_content_boosts(self, scored_docs: List[Tuple[float, Document, Dict]], 
                            chapter_topic: str = None) -> List[Tuple[float, Document, Dict]]:
        """Apply content-based boosts (section, statistical, etc.)"""
        enhanced_scored_docs = []
        
        for score, doc, metadata in scored_docs:
            enhanced_score = score
            
            # Apply section boost
            section_boost = self._calculate_section_boost(doc.metadata, chapter_topic)
            enhanced_score += section_boost
            metadata['boosts_applied'].append(f"section: +{section_boost:.3f}")
            
            # Apply statistical content boost
            if doc.metadata.get('has_statistics', False):
                statistical_boost = self._calculate_statistical_boost(doc.metadata)
                enhanced_score += statistical_boost
                metadata['boosts_applied'].append(f"statistical: +{statistical_boost:.3f}")
                self.reranking_stats['statistical_content_boosted'] += 1
            
            # Apply recency boost
            recency_boost = self._calculate_recency_boost(doc.metadata)
            enhanced_score += recency_boost
            metadata['boosts_applied'].append(f"recency: +{recency_boost:.3f}")
            
            # Apply position boost
            position_boost = self._calculate_position_boost(doc.metadata)
            enhanced_score += position_boost
            metadata['boosts_applied'].append(f"position: +{position_boost:.3f}")
            
            # Update final score
            metadata['final_score'] = enhanced_score
            
            enhanced_scored_docs.append((enhanced_score, doc, metadata))
        
        return enhanced_scored_docs
    
    def _calculate_section_boost(self, doc_metadata: Dict, chapter_topic: str = None) -> float:
        """Calculate section-based boost"""
        section_type = doc_metadata.get('section_type', 'other')
        base_boost = Config.SECTION_WEIGHTS.get(section_type, 0.0)
        
        # Chapter-specific section boosts
        if chapter_topic:
            chapter_section_boosts = {
                'core_determinants': {'results': 0.1, 'findings': 0.1, 'analysis': 0.08},
                'trust_risk_safety': {'results': 0.12, 'findings': 0.12, 'discussion': 0.08},
                'affect_emotion': {'results': 0.1, 'discussion': 0.09, 'conclusion': 0.08},
                'contextual_demographic': {'results': 0.11, 'analysis': 0.09, 'methodology': 0.05}
            }
            
            additional_boost = chapter_section_boosts.get(chapter_topic, {}).get(section_type, 0.0)
            base_boost += additional_boost
        
        return base_boost
    
    def _calculate_statistical_boost(self, doc_metadata: Dict) -> float:
        """Calculate statistical content boost with journal tier awareness"""
        if not doc_metadata.get('has_statistics', False):
            return 0.0
        
        # Base statistical boost
        statistical_score = doc_metadata.get('statistical_score', 0.0)
        base_boost = statistical_score * Config.STATISTICAL_CONTENT_WEIGHT
        
        # Tier-specific statistical boost
        journal_tier = doc_metadata.get('journal_quality_tier', 'unknown')
        tier_multiplier = Config.TIER_STATISTICAL_BOOSTS.get(journal_tier, 1.0)
        
        # Section-specific statistical boost
        section_type = doc_metadata.get('section_type', 'other')
        if section_type in ['results', 'findings', 'analysis']:
            section_multiplier = 1.5
        elif section_type in ['discussion', 'conclusion']:
            section_multiplier = 1.2
        else:
            section_multiplier = 1.0
        
        return base_boost * tier_multiplier * section_multiplier
    
    def _calculate_recency_boost(self, doc_metadata: Dict) -> float:
        """Calculate recency boost for recent publications"""
        try:
            year = int(doc_metadata.get('year', '2000'))
            current_year = 2024  # Update as needed
            
            if year >= current_year - 2:  # Last 2 years
                return 0.1
            elif year >= current_year - 5:  # Last 5 years
                return 0.05
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_position_boost(self, doc_metadata: Dict) -> float:
        """Calculate position-based boost for document location"""
        position = doc_metadata.get('position_in_document', 0.5)
        
        # Boost for middle-to-end sections (where results typically are)
        if 0.3 <= position <= 0.8:
            return 0.05
        elif position > 0.8:
            return 0.03  # Conclusions
        else:
            return 0.0
    
    def _enforce_diversity_and_coverage(self, scored_docs: List[Tuple[float, Document, Dict]], 
                                      query: str) -> List[Tuple[float, Document, Dict]]:
        """Enforce paper diversity and quality tier representation"""
        # Group documents by source (paper)
        docs_by_source = defaultdict(list)
        for score, doc, metadata in scored_docs:
            source = metadata['source']
            docs_by_source[source].append((score, doc, metadata))
        
        # Sort each source's documents by score
        for source in docs_by_source:
            docs_by_source[source].sort(key=lambda x: x[0], reverse=True)
        
        # Enforce diversity and coverage
        selected_docs = []
        source_counts = defaultdict(int)
        tier_counts = defaultdict(int)
        
        # Phase 1: Ensure minimum tier representation
        selected_docs.extend(self._ensure_minimum_tier_representation(docs_by_source))
        
        # Update counts
        for _, doc, metadata in selected_docs:
            source_counts[metadata['source']] += 1
            tier_counts[metadata['journal_tier']] += 1
        
        # Phase 2: Add remaining high-scoring documents with diversity constraints
        remaining_docs = []
        for source, source_docs in docs_by_source.items():
            for score, doc, metadata in source_docs:
                # Skip if already selected
                if any(selected_doc[1].metadata.get('chunk_id') == doc.metadata.get('chunk_id') 
                      for selected_doc in selected_docs):
                    continue
                
                # Check diversity constraints
                if (source_counts[source] < Config.MAX_CHUNKS_PER_PAPER and 
                    len(selected_docs) < Config.MAX_PAPERS_PER_RESPONSE):
                    remaining_docs.append((score, doc, metadata))
        
        # Sort remaining docs by score and add top ones
        remaining_docs.sort(key=lambda x: x[0], reverse=True)
        
        for score, doc, metadata in remaining_docs:
            if len(selected_docs) >= Config.MAX_PAPERS_PER_RESPONSE:
                break
            
            source = metadata['source']
            if source_counts[source] < Config.MAX_CHUNKS_PER_PAPER:
                selected_docs.append((score, doc, metadata))
                source_counts[source] += 1
        
        # Phase 3: Ensure minimum total papers
        if len(set(meta['source'] for _, _, meta in selected_docs)) < Config.MIN_PAPERS_PER_RESPONSE:
            selected_docs = self._ensure_minimum_paper_count(docs_by_source, selected_docs)
        
        self.reranking_stats['paper_diversity_enforced'] = len(set(meta['source'] for _, _, meta in selected_docs))
        
        return selected_docs
    
    def _ensure_minimum_tier_representation(self, docs_by_source: Dict) -> List[Tuple[float, Document, Dict]]:
        """Ensure minimum representation from each quality tier"""
        selected_docs = []
        tier_counts = defaultdict(int)
        
        # Sort all documents by score
        all_docs = []
        for source_docs in docs_by_source.values():
            all_docs.extend(source_docs)
        all_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Select top documents from each tier to meet minimums
        for tier, min_count in Config.MIN_PAPERS_PER_TIER.items():
            tier_docs = [
                (score, doc, metadata) for score, doc, metadata in all_docs
                if metadata.get('journal_tier', 'unknown') == tier and 
                tier_counts[tier] < min_count
            ]
            
            # Add top documents from this tier
            for score, doc, metadata in tier_docs[:min_count]:
                if metadata['source'] not in [meta['source'] for _, _, meta in selected_docs]:
                    selected_docs.append((score, doc, metadata))
                    tier_counts[tier] += 1
                    if tier_counts[tier] >= min_count:
                        break
        
        return selected_docs
    
    def _ensure_minimum_paper_count(self, docs_by_source: Dict, 
                                   current_docs: List[Tuple[float, Document, Dict]]) -> List[Tuple[float, Document, Dict]]:
        """Ensure minimum number of different papers are represented"""
        current_sources = set(meta['source'] for _, _, meta in current_docs)
        
        if len(current_sources) >= Config.MIN_PAPERS_PER_RESPONSE:
            return current_docs
        
        # Add top documents from unrepresented sources
        missing_count = Config.MIN_PAPERS_PER_RESPONSE - len(current_sources)
        
        # Get top document from each unrepresented source
        candidates = []
        for source, source_docs in docs_by_source.items():
            if source not in current_sources and source_docs:
                candidates.append(source_docs[0])  # Top document from this source
        
        # Sort candidates by score and add top ones
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, doc, metadata in candidates[:missing_count]:
            current_docs.append((score, doc, metadata))
        
        return current_docs
    
    def _final_ranking_with_coverage(self, scored_docs: List[Tuple[float, Document, Dict]]) -> List[Document]:
        """Final ranking with comprehensive coverage optimization"""
        # Sort by final score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract documents with final ranking
        final_docs = []
        for i, (score, doc, metadata) in enumerate(scored_docs):
            # Add ranking information to metadata
            doc.metadata['final_rank'] = i + 1
            doc.metadata['final_score'] = score
            doc.metadata['reranking_metadata'] = metadata
            
            final_docs.append(doc)
        
        # Ensure we don't exceed the maximum
        final_docs = final_docs[:Config.RERANK_K]
        
        return final_docs
    
    def _update_ranking_stats(self, final_docs: List[Document]):
        """Update ranking statistics"""
        for doc in final_docs:
            tier = doc.metadata.get('journal_quality_tier', 'unknown')
            self.reranking_stats['quality_tier_distribution'][tier] += 1
    
    def get_reranking_stats(self) -> Dict:
        """Get comprehensive reranking statistics"""
        return {
            'reranking_stats': dict(self.reranking_stats),
            'quality_tier_distribution': dict(self.reranking_stats['quality_tier_distribution']),
            'journal_impact_enabled': self.journal_impact_enabled,
            'configuration': {
                'rerank_k': Config.RERANK_K,
                'min_papers_per_response': Config.MIN_PAPERS_PER_RESPONSE,
                'max_papers_per_response': Config.MAX_PAPERS_PER_RESPONSE,
                'max_chunks_per_paper': Config.MAX_CHUNKS_PER_PAPER,
                'journal_impact_weight': Config.JOURNAL_IMPACT_WEIGHT,
                'statistical_content_weight': Config.STATISTICAL_CONTENT_WEIGHT
            }
        }
    
    def explain_ranking(self, query: str, documents: List[Document]) -> Dict:
        """Explain the ranking decision for documents"""
        if not documents:
            return {}
        
        explanations = {}
        
        # Re-rank with explanation
        scored_docs = self._calculate_base_scores(query, documents)
        if self.journal_impact_enabled:
            scored_docs = self._apply_journal_impact_boosts(scored_docs, query)
        scored_docs = self._apply_content_boosts(scored_docs)
        
        # Create explanations
        for score, doc, metadata in scored_docs:
            doc_id = metadata['document_id']
            explanations[doc_id] = {
                'final_score': score,
                'base_relevance': metadata['base_relevance_score'],
                'source': metadata['source'],
                'journal_tier': metadata.get('journal_tier', 'unknown'),
                'journal_quality_score': metadata.get('journal_quality_score', 0.2),
                'section_type': metadata['section_type'],
                'has_statistics': metadata['has_statistics'],
                'boosts_applied': metadata['boosts_applied'],
                'ranking_factors': self._get_ranking_factors(doc, metadata)
            }
        
        return explanations
    
    def _get_ranking_factors(self, doc: Document, metadata: Dict) -> Dict:
        """Get detailed ranking factors for a document"""
        return {
            'relevance_factor': metadata['base_relevance_score'],
            'journal_impact_factor': metadata.get('journal_quality_score', 0.2),
            'section_importance': Config.SECTION_WEIGHTS.get(metadata['section_type'], 0.0),
            'statistical_content': metadata['has_statistics'],
            'quality_tier': metadata.get('journal_tier', 'unknown'),
            'year': doc.metadata.get('year', 'unknown'),
            'position_in_document': doc.metadata.get('position_in_document', 0.5)
        }
    
    def analyze_paper_coverage(self, query: str, documents: List[Document]) -> Dict:
        """Analyze paper coverage in the ranking"""
        if not documents:
            return {}
        
        # Group by source
        papers_analysis = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            papers_analysis[source].append({
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'section_type': doc.metadata.get('section_type', 'other'),
                'has_statistics': doc.metadata.get('has_statistics', False),
                'journal_tier': doc.metadata.get('journal_quality_tier', 'unknown'),
                'final_score': doc.metadata.get('final_score', 0.0),
                'final_rank': doc.metadata.get('final_rank', 0)
            })
        
        # Calculate coverage statistics
        total_papers = len(papers_analysis)
        tier_coverage = defaultdict(int)
        section_coverage = defaultdict(int)
        
        for source, chunks in papers_analysis.items():
            if chunks:
                # Use the highest-ranked chunk's tier for the paper
                best_chunk = min(chunks, key=lambda x: x['final_rank'])
                tier_coverage[best_chunk['journal_tier']] += 1
                
                # Count sections covered by this paper
                sections = set(chunk['section_type'] for chunk in chunks)
                for section in sections:
                    section_coverage[section] += 1
        
        return {
            'total_papers_in_ranking': total_papers,
            'papers_by_tier': dict(tier_coverage),
            'sections_covered': dict(section_coverage),
            'papers_analysis': dict(papers_analysis),
            'coverage_quality': {
                'tier_diversity': len(tier_coverage),
                'section_diversity': len(section_coverage),
                'average_chunks_per_paper': sum(len(chunks) for chunks in papers_analysis.values()) / total_papers if total_papers > 0 else 0
            }
        }
    
    def suggest_missing_papers(self, query: str, all_documents: List[Document], 
                             ranked_documents: List[Document]) -> Dict:
        """Suggest potentially missing important papers"""
        ranked_sources = set(doc.metadata.get('source', 'unknown') for doc in ranked_documents)
        all_sources = set(doc.metadata.get('source', 'unknown') for doc in all_documents)
        
        missing_sources = all_sources - ranked_sources
        
        suggestions = []
        
        for source in missing_sources:
            # Get documents from this source
            source_docs = [doc for doc in all_documents if doc.metadata.get('source') == source]
            
            if source_docs:
                # Analyze why this source might be missing
                best_doc = max(source_docs, key=lambda x: x.metadata.get('statistical_score', 0))
                
                suggestion = {
                    'source': source,
                    'journal_tier': best_doc.metadata.get('journal_quality_tier', 'unknown'),
                    'journal': best_doc.metadata.get('journal', 'unknown'),
                    'best_section': best_doc.metadata.get('section_type', 'other'),
                    'has_statistics': best_doc.metadata.get('has_statistics', False),
                    'statistical_score': best_doc.metadata.get('statistical_score', 0.0),
                    'chunks_available': len(source_docs),
                    'potential_relevance': self._estimate_relevance(query, source_docs)
                }
                
                suggestions.append(suggestion)
        
        # Sort by potential relevance
        suggestions.sort(key=lambda x: x['potential_relevance'], reverse=True)
        
        return {
            'missing_papers_count': len(missing_sources),
            'suggestions': suggestions[:10],  # Top 10 suggestions
            'analysis': {
                'total_papers_available': len(all_sources),
                'papers_in_ranking': len(ranked_sources),
                'coverage_percentage': (len(ranked_sources) / len(all_sources)) * 100 if all_sources else 0
            }
        }
    
    def _estimate_relevance(self, query: str, documents: List[Document]) -> float:
        """Estimate relevance of documents to query"""
        if not documents:
            return 0.0
        
        # Simple relevance estimation based on keyword matching
        query_terms = query.lower().split()
        
        relevance_scores = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            relevance_scores.append(matches / len(query_terms))
        
        return max(relevance_scores) if relevance_scores else 0.0
    
    def reset_stats(self):
        """Reset reranking statistics"""
        self.reranking_stats = {
            'total_documents_processed': 0,
            'documents_reranked': 0,
            'quality_tier_distribution': defaultdict(int),
            'paper_diversity_enforced': 0,
            'statistical_content_boosted': 0
        }
    
    def get_ranking_summary(self, documents: List[Document]) -> Dict:
        """Get a summary of the ranking results"""
        if not documents:
            return {}
        
        # Analyze the ranking
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        tiers = [doc.metadata.get('journal_quality_tier', 'unknown') for doc in documents]
        sections = [doc.metadata.get('section_type', 'other') for doc in documents]
        
        tier_counts = defaultdict(int)
        section_counts = defaultdict(int)
        
        for tier in tiers:
            tier_counts[tier] += 1
        
        for section in sections:
            section_counts[section] += 1
        
        return {
            'total_documents': len(documents),
            'unique_papers': len(sources),
            'tier_distribution': dict(tier_counts),
            'section_distribution': dict(section_counts),
            'quality_metrics': {
                'tier_diversity': len(tier_counts),
                'section_diversity': len(section_counts),
                'high_quality_papers': tier_counts['top'] + tier_counts['high'],
                'statistical_content': sum(1 for doc in documents if doc.metadata.get('has_statistics', False))
            }
        }