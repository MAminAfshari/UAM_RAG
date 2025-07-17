# core/retrieval.py
"""
Enhanced Retrieval Engine Module with Journal Impact Integration
Handles document retrieval with quality awareness, comprehensive coverage,
multi-stage retrieval, and paper diversity enforcement.
"""

import os
import logging
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .embeddings import EmbeddingManager
from config import Config

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Enhanced retrieval engine with journal impact integration and comprehensive coverage"""
    
    def __init__(self, embedding_manager: EmbeddingManager, journal_manager=None):
        """Initialize enhanced retrieval engine"""
        self.embedding_manager = embedding_manager
        self.journal_manager = journal_manager
        self.db = None
        
        # Retrieval statistics
        self.retrieval_stats = {
            'total_queries_processed': 0,
            'documents_retrieved': 0,
            'comprehensive_mode_used': 0,
            'quality_tier_retrievals': defaultdict(int),
            'paper_diversity_enforced': 0,
            'query_expansions_performed': 0
        }
        
        logger.info(f"Enhanced retrieval engine initialized with journal impact: {bool(journal_manager)}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store"""
        try:
            self.db = FAISS.load_local(
                Config.VECTOR_DB_PATH,
                self.embedding_manager.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return self.db
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create new vector store from documents"""
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        self.db = FAISS.from_documents(documents, self.embedding_manager.embedding_model)
        self.db.save_local(Config.VECTOR_DB_PATH)
        
        logger.info("Vector store created and saved")
        return self.db
    
    def retrieve(self, query: str, k: int = None, chapter_topic: str = None) -> List[Document]:
        """
        Standard retrieval method with enhanced capabilities
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            chapter_topic: Optional chapter topic for focused retrieval
            
        Returns:
            List of retrieved documents
        """
        if not self.db:
            raise ValueError("Please ingest papers first")
        
        k = k or Config.RETRIEVAL_K
        self.retrieval_stats['total_queries_processed'] += 1
        
        logger.info(f"Retrieving documents for query: '{query}' (k={k})")
        
        # Enhanced retrieval with multiple strategies
        if Config.ENABLE_COMPREHENSIVE_COVERAGE:
            return self.retrieve_comprehensive(query, k, chapter_topic)
        else:
            return self.retrieve_standard(query, k, chapter_topic)
    
    def retrieve_standard(self, query: str, k: int, chapter_topic: str = None) -> List[Document]:
        """Standard retrieval with enhancements"""
        # Generate query variants
        queries = self._generate_query_variants(query, chapter_topic)
        
        # Retrieve documents using multiple queries
        all_docs = []
        for q in queries:
            docs = self.db.similarity_search(q, k=k//len(queries))
            all_docs.extend(docs)
        
        # Remove duplicates while preserving diversity
        unique_docs = self._remove_duplicates_preserve_diversity(all_docs)
        
        # Apply quality weighting if journal manager available
        if self.journal_manager:
            unique_docs = self._apply_quality_weighting(unique_docs, query)
        
        # Ensure paper diversity
        diverse_docs = self._ensure_paper_diversity(unique_docs)
        
        # Final selection
        final_docs = diverse_docs[:k]
        
        self.retrieval_stats['documents_retrieved'] += len(final_docs)
        
        logger.info(f"Standard retrieval completed: {len(final_docs)} documents from {self._count_unique_papers(final_docs)} papers")
        
        return final_docs
    
    def retrieve_comprehensive(self, query: str, k: int, chapter_topic: str = None) -> List[Document]:
        """
        Comprehensive retrieval with multi-stage approach for maximum coverage
        
        Args:
            query: The search query
            k: Target number of documents
            chapter_topic: Optional chapter topic
            
        Returns:
            List of documents with comprehensive coverage
        """
        logger.info(f"Starting comprehensive retrieval for query: '{query}'")
        
        self.retrieval_stats['comprehensive_mode_used'] += 1
        
        # Stage 1: Broad initial retrieval
        initial_docs = self._broad_initial_retrieval(query, chapter_topic)
        
        # Stage 2: Quality tier-based retrieval
        tier_docs = self._quality_tier_retrieval(query, initial_docs, chapter_topic)
        
        # Stage 3: Ensure paper diversity
        diverse_docs = self._enforce_comprehensive_diversity(tier_docs)
        
        # Stage 4: Quality-weighted final selection
        final_docs = self._quality_weighted_final_selection(diverse_docs, query, k)
        
        self.retrieval_stats['documents_retrieved'] += len(final_docs)
        
        logger.info(f"Comprehensive retrieval completed: {len(final_docs)} documents from {self._count_unique_papers(final_docs)} papers")
        
        return final_docs
    
    def _broad_initial_retrieval(self, query: str, chapter_topic: str = None) -> List[Document]:
        """Stage 1: Broad initial retrieval to cast wide net"""
        # Use larger k for initial retrieval
        broad_k = Config.COMPREHENSIVE_RETRIEVAL_K
        
        # Generate comprehensive query variants
        queries = self._generate_comprehensive_query_variants(query, chapter_topic)
        
        all_docs = []
        for q in queries:
            docs = self.db.similarity_search(q, k=broad_k//len(queries))
            all_docs.extend(docs)
        
        # Remove duplicates but keep more diverse results
        unique_docs = self._remove_duplicates_preserve_diversity(all_docs, similarity_threshold=0.65)
        
        logger.info(f"Broad initial retrieval: {len(unique_docs)} documents")
        
        return unique_docs
    
    def _quality_tier_retrieval(self, query: str, documents: List[Document], chapter_topic: str = None) -> List[Document]:
        """Stage 2: Ensure representation from all quality tiers"""
        if not self.journal_manager:
            return documents
        
        # Group documents by quality tier
        tier_docs = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            tier_docs[tier].append(doc)
        
        # Ensure minimum representation from each tier
        selected_docs = []
        
        for tier, min_count in Config.MIN_PAPERS_PER_TIER.items():
            tier_documents = tier_docs[tier]
            
            if tier_documents:
                # Sort by relevance and quality
                tier_documents = self._sort_by_relevance_and_quality(tier_documents, query)
                
                # Select top documents from this tier
                selected_count = min(min_count, len(tier_documents))
                selected_docs.extend(tier_documents[:selected_count])
                
                self.retrieval_stats['quality_tier_retrievals'][tier] += selected_count
        
        # Add remaining high-quality documents
        remaining_docs = [doc for doc in documents if doc not in selected_docs]
        remaining_docs = self._sort_by_relevance_and_quality(remaining_docs, query)
        
        # Calculate remaining slots
        remaining_slots = Config.COMPREHENSIVE_RETRIEVAL_K - len(selected_docs)
        selected_docs.extend(remaining_docs[:remaining_slots])
        
        logger.info(f"Quality tier retrieval: {len(selected_docs)} documents with tier representation")
        
        return selected_docs
    
    def _enforce_comprehensive_diversity(self, documents: List[Document]) -> List[Document]:
        """Stage 3: Enforce comprehensive paper diversity"""
        if not documents:
            return documents
        
        # Group by source (paper)
        docs_by_source = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            docs_by_source[source].append(doc)
        
        # Sort documents within each source by quality and relevance
        for source in docs_by_source:
            docs_by_source[source] = self._sort_documents_by_quality(docs_by_source[source])
        
        # Select documents with diversity constraints
        selected_docs = []
        source_counts = defaultdict(int)
        
        # Phase 1: Ensure each source has at least one document
        for source, source_docs in docs_by_source.items():
            if source_docs and len(selected_docs) < Config.COMPREHENSIVE_MAX_PAPERS:
                selected_docs.append(source_docs[0])
                source_counts[source] += 1
        
        # Phase 2: Add more documents with diversity constraints
        all_remaining_docs = []
        for source, source_docs in docs_by_source.items():
            for doc in source_docs[1:]:  # Skip first doc (already selected)
                all_remaining_docs.append(doc)
        
        # Sort remaining documents by quality
        all_remaining_docs = self._sort_documents_by_quality(all_remaining_docs)
        
        for doc in all_remaining_docs:
            if len(selected_docs) >= Config.COMPREHENSIVE_MAX_PAPERS:
                break
            
            source = doc.metadata.get('source', 'unknown')
            if source_counts[source] < Config.MAX_CHUNKS_PER_PAPER:
                selected_docs.append(doc)
                source_counts[source] += 1
        
        self.retrieval_stats['paper_diversity_enforced'] += len(set(source_counts.keys()))
        
        logger.info(f"Comprehensive diversity enforced: {len(selected_docs)} documents from {len(source_counts)} papers")
        
        return selected_docs
    
    def _quality_weighted_final_selection(self, documents: List[Document], query: str, k: int) -> List[Document]:
        """Stage 4: Final selection with quality weighting"""
        if not documents:
            return documents
        
        # Calculate comprehensive scores for each document
        scored_docs = []
        for doc in documents:
            score = self._calculate_comprehensive_score(doc, query)
            scored_docs.append((score, doc))
        
        # Sort by comprehensive score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Select top k documents
        final_docs = [doc for _, doc in scored_docs[:k]]
        
        logger.info(f"Quality-weighted final selection: {len(final_docs)} documents")
        
        return final_docs
    
    def _calculate_comprehensive_score(self, doc: Document, query: str) -> float:
        """Calculate comprehensive score combining multiple factors"""
        # Base similarity score (estimated)
        base_score = 0.5  # Default base score
        
        # Journal impact score
        journal_score = 0.0
        if self.journal_manager:
            source = doc.metadata.get('source', 'unknown')
            journal_score = self.journal_manager.get_paper_quality_score(source)
        
        # Section importance score
        section_score = Config.SECTION_WEIGHTS.get(doc.metadata.get('section_type', 'other'), 0.0)
        
        # Statistical content score
        statistical_score = doc.metadata.get('statistical_score', 0.0)
        
        # Combined quality score from ingestion
        combined_quality = doc.metadata.get('combined_quality_score', 0.0)
        
        # Comprehensive score calculation
        comprehensive_score = (
            base_score * Config.RELEVANCE_WEIGHT +
            journal_score * Config.JOURNAL_IMPACT_WEIGHT +
            statistical_score * Config.STATISTICAL_CONTENT_WEIGHT +
            section_score * 0.1 +
            combined_quality * 0.1
        )
        
        return comprehensive_score
    
    def _generate_query_variants(self, query: str, chapter_topic: str = None) -> List[str]:
        """Generate query variants for standard retrieval"""
        queries = [query]
        
        if Config.ENABLE_QUERY_EXPANSION:
            # Basic query expansion
            expanded_queries = self._expand_query(query, chapter_topic)
            queries.extend(expanded_queries)
        
        # Journal-aware query expansion
        if self.journal_manager and chapter_topic:
            journal_queries = self._generate_journal_aware_queries(query, chapter_topic)
            queries.extend(journal_queries)
        
        self.retrieval_stats['query_expansions_performed'] += len(queries) - 1
        
        return queries[:5]  # Limit to 5 queries for performance
    
    def _generate_comprehensive_query_variants(self, query: str, chapter_topic: str = None) -> List[str]:
        """Generate comprehensive query variants for broad retrieval"""
        queries = [query]
        
        # Extended query expansion
        if Config.ENABLE_QUERY_EXPANSION:
            expanded_queries = self._expand_query(query, chapter_topic)
            queries.extend(expanded_queries)
        
        # Chapter-specific expansion
        if chapter_topic and chapter_topic in Config.CHAPTER_KEYWORDS:
            chapter_queries = self._generate_chapter_specific_queries(query, chapter_topic)
            queries.extend(chapter_queries)
        
        # Statistical content queries
        if Config.ENABLE_STATISTICAL_EXTRACTION:
            statistical_queries = self._generate_statistical_queries(query)
            queries.extend(statistical_queries)
        
        # Journal-aware queries
        if self.journal_manager:
            journal_queries = self._generate_journal_aware_queries(query, chapter_topic)
            queries.extend(journal_queries)
        
        self.retrieval_stats['query_expansions_performed'] += len(queries) - 1
        
        return queries[:8]  # More queries for comprehensive retrieval
    
    def _expand_query(self, query: str, chapter_topic: str = None) -> List[str]:
        """Expand query with related terms"""
        expanded_queries = []
        
        # Chapter-specific expansions
        if chapter_topic and chapter_topic in Config.CHAPTER_KEYWORDS:
            keywords = Config.CHAPTER_KEYWORDS[chapter_topic]
            # Add query with top keywords
            expanded_query = f"{query} {' '.join(keywords[:3])}"
            expanded_queries.append(expanded_query)
        
        # Semantic expansions
        expansions = {
            'trust': ['confidence', 'reliability', 'faith', 'belief'],
            'adoption': ['acceptance', 'uptake', 'usage', 'utilization'],
            'intention': ['willingness', 'readiness', 'inclination', 'propensity'],
            'risk': ['danger', 'hazard', 'uncertainty', 'threat'],
            'safety': ['security', 'protection', 'wellbeing', 'harm'],
            'attitude': ['perception', 'opinion', 'evaluation', 'assessment'],
            'behavior': ['behaviour', 'conduct', 'action', 'response'],
            'technology': ['tech', 'innovation', 'system', 'solution']
        }
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Use top 2 synonyms
                    expanded_query = query.replace(term, synonym)
                    expanded_queries.append(expanded_query)
                break
        
        return expanded_queries[:3]  # Limit expansions
    
    def _generate_chapter_specific_queries(self, query: str, chapter_topic: str) -> List[str]:
        """Generate chapter-specific query variants"""
        chapter_queries = []
        
        chapter_terms = {
            'core_determinants': ['TAM', 'TPB', 'UTAUT', 'behavioral intention', 'attitude'],
            'trust_risk_safety': ['trust', 'safety', 'risk perception', 'security'],
            'affect_emotion': ['emotion', 'anxiety', 'hedonic', 'affective'],
            'contextual_demographic': ['demographic', 'cultural', 'age', 'gender']
        }
        
        if chapter_topic in chapter_terms:
            terms = chapter_terms[chapter_topic]
            for term in terms[:3]:  # Use top 3 terms
                chapter_queries.append(f"{query} {term}")
        
        return chapter_queries
    
    def _generate_statistical_queries(self, query: str) -> List[str]:
        """Generate queries focused on statistical content"""
        statistical_queries = []
        
        # Add statistical terms to query
        statistical_terms = ['results', 'findings', 'significant', 'correlation', 'regression']
        
        for term in statistical_terms[:2]:  # Use top 2 terms
            statistical_queries.append(f"{query} {term}")
        
        return statistical_queries
    
    def _generate_journal_aware_queries(self, query: str, chapter_topic: str = None) -> List[str]:
        """Generate journal-aware query variants"""
        if not self.journal_manager:
            return []
        
        journal_queries = []
        
        # Add quality-focused terms
        quality_terms = ['empirical', 'statistical', 'significant', 'validated']
        
        for term in quality_terms[:2]:  # Use top 2 terms
            journal_queries.append(f"{query} {term}")
        
        return journal_queries
    
    def _remove_duplicates_preserve_diversity(self, documents: List[Document], 
                                            similarity_threshold: float = 0.70) -> List[Document]:
        """Remove duplicates while preserving source diversity"""
        if not documents:
            return documents
        
        unique_docs = []
        seen_content = set()
        source_counts = defaultdict(int)
        
        # Sort documents by quality if journal manager available
        if self.journal_manager:
            documents = self._sort_documents_by_quality(documents)
        
        for doc in documents:
            # Check content similarity
            content_hash = hash(doc.page_content[:200])
            
            # Check if content is too similar to existing documents
            too_similar = False
            for seen_hash in seen_content:
                if self._calculate_content_similarity(content_hash, seen_hash) > similarity_threshold:
                    too_similar = True
                    break
            
            if not too_similar:
                source = doc.metadata.get('source', 'unknown')
                
                # Add document if it doesn't violate diversity constraints
                if source_counts[source] < Config.MAX_CHUNKS_PER_PAPER:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
                    source_counts[source] += 1
        
        return unique_docs
    
    def _calculate_content_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate content similarity between two content hashes"""
        # Simple hash-based similarity (can be improved)
        return 1.0 if hash1 == hash2 else 0.0
    
    def _sort_documents_by_quality(self, documents: List[Document]) -> List[Document]:
        """Sort documents by quality score"""
        if not self.journal_manager:
            return documents
        
        def get_quality_score(doc):
            combined_score = doc.metadata.get('combined_quality_score', 0.0)
            journal_score = doc.metadata.get('journal_quality_score', 0.0)
            statistical_score = doc.metadata.get('statistical_score', 0.0)
            
            return combined_score + journal_score + statistical_score
        
        return sorted(documents, key=get_quality_score, reverse=True)
    
    def _sort_by_relevance_and_quality(self, documents: List[Document], query: str) -> List[Document]:
        """Sort documents by combined relevance and quality"""
        def get_combined_score(doc):
            quality_score = self._calculate_comprehensive_score(doc, query)
            return quality_score
        
        return sorted(documents, key=get_combined_score, reverse=True)
    
    def _apply_quality_weighting(self, documents: List[Document], query: str) -> List[Document]:
        """Apply quality weighting to documents"""
        if not self.journal_manager:
            return documents
        
        # Calculate weighted scores
        weighted_docs = []
        for doc in documents:
            score = self._calculate_comprehensive_score(doc, query)
            weighted_docs.append((score, doc))
        
        # Sort by weighted score
        weighted_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in weighted_docs]
    
    def _ensure_paper_diversity(self, documents: List[Document]) -> List[Document]:
        """Ensure paper diversity in retrieved documents"""
        if not documents:
            return documents
        
        # Group by source
        docs_by_source = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            docs_by_source[source].append(doc)
        
        # Select documents with diversity constraints
        diverse_docs = []
        source_counts = defaultdict(int)
        
        # Sort all documents by quality
        all_docs = self._sort_documents_by_quality(documents)
        
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            
            # Add document if it doesn't violate diversity constraints
            if (source_counts[source] < Config.MAX_CHUNKS_PER_PAPER and 
                len(set(source_counts.keys())) < Config.MAX_PAPERS_PER_RESPONSE):
                diverse_docs.append(doc)
                source_counts[source] += 1
        
        return diverse_docs
    
    def _count_unique_papers(self, documents: List[Document]) -> int:
        """Count unique papers in document list"""
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        return len(sources)
    
    def retrieve_by_quality_tiers(self, query: str, tier_requirements: Dict[str, int]) -> List[Document]:
        """
        Retrieve documents with specific quality tier requirements
        
        Args:
            query: The search query
            tier_requirements: Dict mapping tier names to required counts
            
        Returns:
            List of documents meeting tier requirements
        """
        if not self.db or not self.journal_manager:
            return self.retrieve(query)
        
        logger.info(f"Retrieving documents by quality tiers: {tier_requirements}")
        
        # Broad initial retrieval
        initial_docs = self.db.similarity_search(query, k=Config.COMPREHENSIVE_RETRIEVAL_K)
        
        # Group by quality tier
        tier_docs = defaultdict(list)
        for doc in initial_docs:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            tier_docs[tier].append(doc)
        
        # Select documents to meet tier requirements
        selected_docs = []
        for tier, required_count in tier_requirements.items():
            tier_documents = tier_docs[tier]
            
            if tier_documents:
                # Sort by quality within tier
                tier_documents = self._sort_documents_by_quality(tier_documents)
                
                # Select required number of documents
                selected_count = min(required_count, len(tier_documents))
                selected_docs.extend(tier_documents[:selected_count])
        
        logger.info(f"Quality tier retrieval completed: {len(selected_docs)} documents")
        
        return selected_docs
    
    def find_missing_papers(self, query: str, expected_papers: List[str]) -> Dict:
        """
        Find why expected papers are missing from retrieval
        
        Args:
            query: The search query
            expected_papers: List of expected paper sources
            
        Returns:
            Analysis of missing papers
        """
        if not self.db:
            return {'error': 'Vector store not available'}
        
        logger.info(f"Analyzing missing papers for query: '{query}'")
        
        # Retrieve documents
        retrieved_docs = self.retrieve_comprehensive(query, k=Config.COMPREHENSIVE_RETRIEVAL_K)
        retrieved_sources = set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)
        
        # Analyze missing papers
        missing_papers = []
        for paper_source in expected_papers:
            if paper_source not in retrieved_sources:
                # Try to find this paper in the vector store
                paper_analysis = self._analyze_missing_paper(paper_source, query)
                missing_papers.append(paper_analysis)
        
        return {
            'query': query,
            'expected_papers': expected_papers,
            'retrieved_papers': list(retrieved_sources),
            'missing_papers': missing_papers,
            'coverage_percentage': (len(retrieved_sources) / len(expected_papers)) * 100 if expected_papers else 0
        }
    
    def _analyze_missing_paper(self, paper_source: str, query: str) -> Dict:
        """Analyze why a specific paper is missing"""
        # Search for documents from this paper
        all_docs = self.db.similarity_search(query, k=Config.COMPREHENSIVE_RETRIEVAL_K * 2)
        
        paper_docs = [doc for doc in all_docs if doc.metadata.get('source') == paper_source]
        
        if not paper_docs:
            return {
                'source': paper_source,
                'status': 'not_in_corpus',
                'reason': 'Paper not found in vector store'
            }
        
        # Calculate best similarity score for this paper
        best_doc = paper_docs[0]  # Assuming sorted by similarity
        
        # Get journal information
        journal_info = {}
        if self.journal_manager:
            journal_data = self.journal_manager.match_paper_to_journal(paper_source)
            if journal_data:
                journal_info = {
                    'journal': journal_data['journal'],
                    'tier': self.journal_manager.get_quality_tier(paper_source),
                    'quality_score': self.journal_manager.get_paper_quality_score(paper_source)
                }
        
        return {
            'source': paper_source,
            'status': 'in_corpus_not_retrieved',
            'reason': 'Low similarity score or filtered out',
            'documents_available': len(paper_docs),
            'best_section': best_doc.metadata.get('section_type', 'unknown'),
            'has_statistics': best_doc.metadata.get('has_statistics', False),
            'journal_info': journal_info,
            'suggestions': self._generate_retrieval_suggestions(paper_source, query)
        }
    
    def _generate_retrieval_suggestions(self, paper_source: str, query: str) -> List[str]:
        """Generate suggestions for retrieving a missing paper"""
        suggestions = []
        
        # Check if paper has journal info
        if self.journal_manager:
            journal_info = self.journal_manager.match_paper_to_journal(paper_source)
            if not journal_info:
                suggestions.append("Add journal information to improve quality scoring")
            elif journal_info.get('quartile') in ['Q3', 'Q4']:
                suggestions.append("Paper from lower-tier journal may be filtered out")
        
        # Check for query expansion opportunities
        if len(query.split()) < 3:
            suggestions.append("Try more specific or longer queries")
        
        suggestions.append("Consider using comprehensive retrieval mode")
        suggestions.append("Check if paper has relevant statistical content")
        
        return suggestions
    
    def analyze_retrieval_coverage(self, query: str, chapter_topic: str = None) -> Dict:
        """
        Analyze retrieval coverage and quality
        
        Args:
            query: The search query
            chapter_topic: Optional chapter topic
            
        Returns:
            Comprehensive analysis of retrieval coverage
        """
        if not self.db:
            return {'error': 'Vector store not available'}
        
        logger.info(f"Analyzing retrieval coverage for query: '{query}'")
        
        # Retrieve with different methods
        standard_docs = self.retrieve_standard(query, Config.RETRIEVAL_K, chapter_topic)
        comprehensive_docs = self.retrieve_comprehensive(query, Config.RETRIEVAL_K, chapter_topic)
        
        # Analyze coverage
        analysis = {
            'query': query,
            'chapter_topic': chapter_topic,
            'standard_retrieval': self._analyze_document_set(standard_docs),
            'comprehensive_retrieval': self._analyze_document_set(comprehensive_docs),
            'coverage_comparison': self._compare_coverage(standard_docs, comprehensive_docs),
            'quality_analysis': self._analyze_quality_distribution(comprehensive_docs),
            'recommendations': self._generate_coverage_recommendations(standard_docs, comprehensive_docs)
        }
        
        return analysis
    
    def _analyze_document_set(self, documents: List[Document]) -> Dict:
        """Analyze a set of documents"""
        if not documents:
            return {'total_documents': 0, 'unique_papers': 0}
        
        sources = [doc.metadata.get('source', 'unknown') for doc in documents]
        sections = [doc.metadata.get('section_type', 'other') for doc in documents]
        
        # Count statistics
        unique_sources = set(sources)
        section_counts = defaultdict(int)
        for section in sections:
            section_counts[section] += 1
        
        # Quality analysis
        quality_analysis = {}
        if self.journal_manager:
            tier_counts = defaultdict(int)
            for source in sources:
                tier = self.journal_manager.get_quality_tier(source)
                tier_counts[tier] += 1
            quality_analysis['tier_distribution'] = dict(tier_counts)
        
        return {
            'total_documents': len(documents),
            'unique_papers': len(unique_sources),
            'section_distribution': dict(section_counts),
            'quality_analysis': quality_analysis,
            'papers_list': list(unique_sources)
        }
    
    def _compare_coverage(self, standard_docs: List[Document], comprehensive_docs: List[Document]) -> Dict:
        """Compare coverage between standard and comprehensive retrieval"""
        standard_sources = set(doc.metadata.get('source', 'unknown') for doc in standard_docs)
        comprehensive_sources = set(doc.metadata.get('source', 'unknown') for doc in comprehensive_docs)
        
        return {
            'standard_papers': len(standard_sources),
            'comprehensive_papers': len(comprehensive_sources),
            'additional_papers': len(comprehensive_sources - standard_sources),
            'coverage_improvement': len(comprehensive_sources - standard_sources),
            'overlap': len(standard_sources & comprehensive_sources)
        }
    
    def _analyze_quality_distribution(self, documents: List[Document]) -> Dict:
        """Analyze quality distribution of retrieved documents"""
        if not self.journal_manager or not documents:
            return {}
        
        tier_counts = defaultdict(int)
        quality_scores = []
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            quality_score = self.journal_manager.get_paper_quality_score(source)
            
            tier_counts[tier] += 1
            quality_scores.append(quality_score)
        
        return {
            'tier_distribution': dict(tier_counts),
            'quality_score_stats': {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0
            },
            'high_quality_papers': tier_counts['top'] + tier_counts['high'],
            'total_papers': sum(tier_counts.values())
        }
    
    def _generate_coverage_recommendations(self, standard_docs: List[Document], 
                                        comprehensive_docs: List[Document]) -> List[str]:
        """Generate recommendations for improving coverage"""
        recommendations = []
        
        standard_papers = len(set(doc.metadata.get('source', 'unknown') for doc in standard_docs))
        comprehensive_papers = len(set(doc.metadata.get('source', 'unknown') for doc in comprehensive_docs))
        
        if comprehensive_papers > standard_papers:
            recommendations.append("Use comprehensive retrieval mode for better coverage")
        
        if standard_papers < Config.MIN_PAPERS_PER_RESPONSE:
            recommendations.append("Consider expanding query terms for broader coverage")
        
        if self.journal_manager:
            tier_counts = defaultdict(int)
            for doc in comprehensive_docs:
                source = doc.metadata.get('source', 'unknown')
                tier = self.journal_manager.get_quality_tier(source)
                tier_counts[tier] += 1
            
            if tier_counts['top'] < 2:
                recommendations.append("Few top-tier papers found - consider query refinement")
            
            if tier_counts['unknown'] > len(comprehensive_docs) * 0.3:
                recommendations.append("Many papers lack journal information - update metadata")
        
        return recommendations
    
    def get_retrieval_statistics(self) -> Dict:
        """Get comprehensive retrieval statistics"""
        return {
            'retrieval_stats': dict(self.retrieval_stats),
            'quality_tier_distribution': dict(self.retrieval_stats['quality_tier_retrievals']),
            'configuration': {
                'retrieval_k': Config.RETRIEVAL_K,
                'comprehensive_retrieval_k': Config.COMPREHENSIVE_RETRIEVAL_K,
                'max_papers_per_response': Config.MAX_PAPERS_PER_RESPONSE,
                'max_chunks_per_paper': Config.MAX_CHUNKS_PER_PAPER,
                'enable_comprehensive_coverage': Config.ENABLE_COMPREHENSIVE_COVERAGE,
                'enable_query_expansion': Config.ENABLE_QUERY_EXPANSION
            },
            'journal_impact_enabled': bool(self.journal_manager)
        }
    
    def reset_stats(self):
        """Reset retrieval statistics"""
        self.retrieval_stats = {
            'total_queries_processed': 0,
            'documents_retrieved': 0,
            'comprehensive_mode_used': 0,
            'quality_tier_retrievals': defaultdict(int),
            'paper_diversity_enforced': 0,
            'query_expansions_performed': 0
        }
    
    def clear_cache(self):
        """Clear any cached data"""
        logger.info("Clearing retrieval cache")
        # Add any cache clearing logic here if needed