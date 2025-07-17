# core/rag_system.py
"""
Enhanced Main RAG System Orchestrator with Journal Impact Integration
Coordinates all pipeline components for comprehensive literature review generation
with quality-weighted ranking, paper diversity enforcement, and journal impact awareness.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict

from .embeddings import EmbeddingManager
from .retrieval import RetrievalEngine
from .reranking import ReRankingEngine
from .generation import ResponseGenerator
from .ingestion import DocumentIngester
from .journal_impact import JournalImpactManager

from config import Config

logger = logging.getLogger(__name__)


class UAMRAGSystem:
    """Enhanced UAM Literature Review RAG System with Journal Impact Integration"""
    
    def __init__(self):
        """Initialize the enhanced RAG system with journal impact capabilities"""
        logger.info("Initializing Enhanced UAM RAG System with Journal Impact Integration")
        
        # Validate configuration
        try:
            Config.validate_config()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize journal impact manager first
        self.journal_manager = None
        if Config.ENABLE_JOURNAL_RANKING:
            self._initialize_journal_manager()
        
        # Initialize core components with journal impact support
        self.embedding_manager = EmbeddingManager()
        self.retrieval_engine = RetrievalEngine(self.embedding_manager, self.journal_manager)
        self.reranking_engine = ReRankingEngine(self.journal_manager)
        self.response_generator = ResponseGenerator()
        self.document_ingester = DocumentIngester(self.embedding_manager)
        
        # Load existing vector store if available
        self.db = self.retrieval_engine.load_vector_store()
        
        # System state
        self.is_ready = self._check_system_ready()
        
        # Enhanced system statistics
        self.system_stats = {
            'queries_processed': 0,
            'comprehensive_queries': 0,
            'papers_analyzed': 0,
            'journal_impact_enabled': bool(self.journal_manager),
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"Enhanced UAM RAG System initialized successfully (Journal Impact: {bool(self.journal_manager)})")
    
    def _initialize_journal_manager(self):
        """Initialize journal impact manager with error handling"""
        try:
            # Try to load from cache first
            if os.path.exists(Config.JOURNAL_METADATA_CACHE_PATH):
                self.journal_manager = JournalImpactManager()
                if self.journal_manager.load_metadata_cache(Config.JOURNAL_METADATA_CACHE_PATH):
                    logger.info("Journal impact manager loaded from cache")
                    return
            
            # Load from Excel file
            if os.path.exists(Config.JOURNAL_METADATA_PATH):
                self.journal_manager = JournalImpactManager(Config.JOURNAL_METADATA_PATH)
                
                # Save to cache for faster future loads
                self.journal_manager.save_metadata_cache(Config.JOURNAL_METADATA_CACHE_PATH)
                logger.info("Journal impact manager loaded from Excel and cached")
            else:
                logger.warning(f"Journal metadata file not found: {Config.JOURNAL_METADATA_PATH}")
                
        except Exception as e:
            logger.error(f"Failed to initialize journal impact manager: {e}")
            self.journal_manager = None
    
    def _check_system_ready(self) -> bool:
        """Check if system is ready for queries with enhanced validation"""
        try:
            if self.db is None:
                logger.warning("No vector store available - system not ready for queries")
                return False
            
            # Check if we have papers
            stats = self.get_corpus_statistics()
            if not stats or stats.get('total_papers', 0) == 0:
                logger.warning("No papers in corpus - system not ready for queries")
                return False
            
            # Check journal impact readiness
            if Config.ENABLE_JOURNAL_RANKING and not self.journal_manager:
                logger.warning("Journal ranking enabled but manager not available")
            
            return True
        except Exception as e:
            logger.error(f"Error checking system readiness: {e}")
            return False
    
    def ingest_papers(self, pdf_directory: str, force_text_only: bool = False) -> Dict:
        """
        Enhanced paper ingestion with journal impact integration
        
        Args:
            pdf_directory: Path to directory containing PDF files
            force_text_only: Force text-only processing even if multimodal is available
            
        Returns:
            Dictionary with enhanced ingestion statistics including journal impact
        """
        logger.info(f"Starting enhanced paper ingestion with journal impact from {pdf_directory}")
        
        if not os.path.exists(pdf_directory):
            raise ValueError(f"Directory does not exist: {pdf_directory}")
        
        # Check for PDF files
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Use the enhanced ingestion method with journal impact
        stats = self.document_ingester.ingest_papers(pdf_directory, force_text_only)
        
        # Reload vector store and components
        self.db = self.retrieval_engine.load_vector_store()
        
        # Update retrieval engine with new data
        self.retrieval_engine.db = self.db
        
        # Update system readiness
        self.is_ready = self._check_system_ready()
        
        # Update system statistics
        self.system_stats['papers_analyzed'] += stats.get('successful_papers', 0)
        self.system_stats['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Enhanced ingestion complete with journal impact: {stats}")
        return stats
    
    def enhanced_retrieval(self, query: str, k: int = None, chapter_topic: str = None) -> List:
        """
        Enhanced retrieval with journal impact integration and comprehensive coverage
        
        Args:
            query: The research query
            k: Number of documents to retrieve
            chapter_topic: Optional chapter topic for focused retrieval
            
        Returns:
            List of retrieved documents with quality weighting
        """
        if not self.is_ready:
            raise ValueError("Please ingest papers first before running queries")
        
        k = k or Config.RETRIEVAL_K
        
        logger.info(f"Starting enhanced retrieval with journal impact (k={k})")
        
        # Preprocess query with enhancements
        processed_query = self._preprocess_query(query)
        
        # Use comprehensive retrieval if enabled
        if Config.ENABLE_COMPREHENSIVE_COVERAGE:
            docs = self.retrieval_engine.retrieve_comprehensive(processed_query, k, chapter_topic)
        else:
            docs = self.retrieval_engine.retrieve(processed_query, k, chapter_topic)
        
        # Additional query variants if enabled
        if Config.ENABLE_QUERY_EXPANSION:
            expanded_queries = self._expand_query(processed_query, chapter_topic)
            
            for expanded_query in expanded_queries[:2]:  # Limit to 2 additional queries
                additional_docs = self.retrieval_engine.retrieve(expanded_query, k//3, chapter_topic)
                docs.extend(additional_docs)
        
        # Remove duplicates with quality preservation
        unique_docs = self._remove_duplicates_preserve_quality(docs)
        
        # Apply final quality filtering
        final_docs = self._apply_quality_filtering(unique_docs, k)
        
        logger.info(f"Enhanced retrieval completed: {len(final_docs)} documents from {self._count_unique_papers(final_docs)} papers")
        
        return final_docs
    
    def answer_literature_query(self, 
                               query: str, 
                               chapter_topic: str = None,
                               include_figures: bool = True,
                               include_tables: bool = True,
                               comprehensive_mode: bool = False) -> Tuple[str, List[str]]:
        """
        Answer a literature review query with enhanced quality-weighted generation
        
        Args:
            query: The research question
            chapter_topic: Optional chapter focus for specialized retrieval
            include_figures: Whether to include figure content in retrieval
            include_tables: Whether to include table content in retrieval
            comprehensive_mode: Whether to use comprehensive coverage mode
            
        Returns:
            Tuple of (response_text, source_list, quality_metadata)
        """
        logger.info(f"Processing enhanced literature query: {query}")
        
        if not self.is_ready:
            raise ValueError("Please ingest papers first before generating literature reviews")
        
        # Update system statistics
        self.system_stats['queries_processed'] += 1
        if comprehensive_mode:
            self.system_stats['comprehensive_queries'] += 1
        
        # Enhanced retrieval with quality weighting
        if comprehensive_mode:
            docs = self.retrieval_engine.retrieve_comprehensive(query, Config.RETRIEVAL_K, chapter_topic)
        else:
            docs = self.enhanced_retrieval(query, chapter_topic=chapter_topic)
        
        # Filter by content type if multimodal was used
        if hasattr(self.document_ingester, 'multimodal_available') and self.document_ingester.multimodal_available:
            filtered_docs = self._filter_multimodal_content(docs, include_figures, include_tables)
            docs = filtered_docs
        
        logger.info(f"Retrieved {len(docs)} documents for enhanced processing")
        
        # Enhanced re-ranking with journal impact
        top_docs = self.reranking_engine.rerank(query, docs, chapter_topic)
        logger.info(f"Enhanced re-ranking completed: {len(top_docs)} top documents")
        
        # Generate enhanced response with quality metadata
        response = self.response_generator.generate_response(query, top_docs, chapter_topic)
        
        # Extract sources with quality information
        sources = self._extract_sources_with_quality(top_docs)
        
        # Generate quality metadata
        quality_metadata = self._generate_quality_metadata(top_docs)
        
        logger.info(f"Enhanced response generated with {len(sources)} unique sources")
        
        return response, sources, quality_metadata
    
    def answer_literature_query_comprehensive(self, 
                                            query: str, 
                                            chapter_topic: str = None,
                                            min_papers_per_tier: Dict[str, int] = None,
                                            ensure_statistical_content: bool = True) -> Tuple[str, List[str], Dict]:
        """
        Answer literature query with comprehensive coverage and quality requirements
        
        Args:
            query: The research question
            chapter_topic: Optional chapter focus
            min_papers_per_tier: Minimum papers required per quality tier
            ensure_statistical_content: Whether to ensure statistical content inclusion
            
        Returns:
            Tuple of (response_text, source_list, comprehensive_metadata)
        """
        logger.info(f"Processing comprehensive literature query: {query}")
        
        if not self.is_ready:
            raise ValueError("Please ingest papers first before generating comprehensive reviews")
        
        # Set tier requirements
        tier_requirements = min_papers_per_tier or Config.MIN_PAPERS_PER_TIER
        
        # Comprehensive retrieval with tier requirements
        docs = self.retrieval_engine.retrieve_by_quality_tiers(query, tier_requirements)
        
        # Ensure statistical content if required
        if ensure_statistical_content:
            docs = self._ensure_statistical_content(docs, query)
        
        # Enhanced re-ranking with comprehensive coverage
        top_docs = self.reranking_engine.rerank(query, docs, chapter_topic)
        
        # Generate comprehensive response
        response = self.response_generator.generate_response(query, top_docs, chapter_topic)
        
        # Extract sources with comprehensive metadata
        sources = self._extract_sources_with_quality(top_docs)
        
        # Generate comprehensive metadata
        comprehensive_metadata = self._generate_comprehensive_metadata(top_docs, tier_requirements)
        
        # Update statistics
        self.system_stats['comprehensive_queries'] += 1
        
        logger.info(f"Comprehensive response generated: {len(sources)} sources, {len(top_docs)} documents")
        
        return response, sources, comprehensive_metadata
    
    def analyze_paper_coverage(self, query: str, expected_papers: List[str]) -> Dict:
        """
        Analyze paper coverage for a given query against expected papers
        
        Args:
            query: The research query
            expected_papers: List of expected paper sources
            
        Returns:
            Comprehensive coverage analysis
        """
        logger.info(f"Analyzing paper coverage for query: '{query}'")
        
        if not self.is_ready:
            raise ValueError("Please ingest papers first")
        
        # Use retrieval engine's missing paper analysis
        missing_analysis = self.retrieval_engine.find_missing_papers(query, expected_papers)
        
        # Get comprehensive coverage analysis
        coverage_analysis = self.retrieval_engine.analyze_retrieval_coverage(query)
        
        # Generate quality distribution analysis
        quality_analysis = self._analyze_quality_distribution_for_query(query)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_coverage_improvement_suggestions(
            missing_analysis, coverage_analysis, quality_analysis
        )
        
        comprehensive_analysis = {
            'query': query,
            'expected_papers': expected_papers,
            'missing_paper_analysis': missing_analysis,
            'coverage_analysis': coverage_analysis,
            'quality_distribution': quality_analysis,
            'improvement_suggestions': improvement_suggestions,
            'system_capabilities': {
                'journal_impact_enabled': bool(self.journal_manager),
                'comprehensive_mode_available': Config.ENABLE_COMPREHENSIVE_COVERAGE,
                'quality_tiers_available': bool(self.journal_manager)
            }
        }
        
        return comprehensive_analysis
    
    def find_missing_important_papers(self, query: str, important_papers: List[str]) -> Dict:
        """
        Find missing important papers and suggest retrieval improvements
        
        Args:
            query: The research query
            important_papers: List of important paper sources that should be retrieved
            
        Returns:
            Analysis of missing papers with suggestions
        """
        logger.info(f"Finding missing important papers for query: '{query}'")
        
        if not self.is_ready:
            raise ValueError("Please ingest papers first")
        
        # Analyze missing papers
        missing_analysis = self.retrieval_engine.find_missing_papers(query, important_papers)
        
        # Get ranking explanations for available papers
        retrieved_docs = self.enhanced_retrieval(query, k=Config.RETRIEVAL_K)
        ranking_explanations = self.reranking_engine.explain_ranking(query, retrieved_docs)
        
        # Suggest improvements for missing papers
        improvement_strategies = self._suggest_missing_paper_improvements(
            missing_analysis, ranking_explanations, important_papers
        )
        
        # Generate alternative queries that might retrieve missing papers
        alternative_queries = self._generate_alternative_queries_for_missing_papers(
            query, missing_analysis['missing_papers']
        )
        
        return {
            'query': query,
            'important_papers': important_papers,
            'missing_analysis': missing_analysis,
            'ranking_explanations': ranking_explanations,
            'improvement_strategies': improvement_strategies,
            'alternative_queries': alternative_queries,
            'system_recommendations': self._generate_system_recommendations(missing_analysis)
        }
    
    def get_comprehensive_coverage_stats(self) -> Dict:
        """
        Get comprehensive coverage statistics for the system
        
        Returns:
            Detailed statistics about system coverage capabilities
        """
        logger.info("Generating comprehensive coverage statistics")
        
        # Get corpus statistics
        corpus_stats = self.get_corpus_statistics()
        
        # Get retrieval statistics
        retrieval_stats = self.retrieval_engine.get_retrieval_statistics()
        
        # Get reranking statistics
        reranking_stats = self.reranking_engine.get_reranking_stats()
        
        # Get journal impact statistics
        journal_stats = {}
        if self.journal_manager:
            journal_stats = self.journal_manager.get_journal_statistics()
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(corpus_stats, journal_stats)
        
        comprehensive_stats = {
            'timestamp': datetime.now().isoformat(),
            'system_stats': self.system_stats,
            'corpus_stats': corpus_stats,
            'retrieval_stats': retrieval_stats,
            'reranking_stats': reranking_stats,
            'journal_impact_stats': journal_stats,
            'coverage_metrics': coverage_metrics,
            'configuration': {
                'journal_impact_enabled': bool(self.journal_manager),
                'comprehensive_mode_enabled': Config.ENABLE_COMPREHENSIVE_COVERAGE,
                'quality_tier_balancing': Config.ENABLE_TIER_BALANCING,
                'paper_diversity_enforcement': Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT,
                'min_papers_per_response': Config.MIN_PAPERS_PER_RESPONSE,
                'max_papers_per_response': Config.MAX_PAPERS_PER_RESPONSE,
                'retrieval_k': Config.RETRIEVAL_K,
                'rerank_k': Config.RERANK_K
            }
        }
        
        return comprehensive_stats
    
    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing with journal impact awareness"""
        # Basic preprocessing
        query = query.strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Add UAM-specific context if not present
        uam_terms = ['uam', 'urban air mobility', 'air taxi', 'evtol', 'vtol']
        if not any(term in query.lower() for term in uam_terms):
            query = f"UAM {query}"
        
        # Add quality-focused terms if journal impact is enabled
        if self.journal_manager and Config.ENABLE_JOURNAL_RANKING:
            quality_terms = ['empirical', 'statistical', 'significant']
            if not any(term in query.lower() for term in quality_terms):
                query = f"{query} empirical"
        
        return query
    
    def _expand_query(self, query: str, chapter_topic: str = None) -> List[str]:
        """Enhanced query expansion with journal impact awareness"""
        expanded_queries = []
        
        # Chapter-specific expansions
        if chapter_topic and chapter_topic in Config.CHAPTER_KEYWORDS:
            keywords = Config.CHAPTER_KEYWORDS[chapter_topic]
            # Add query with top keywords
            expanded_query = f"{query} {' '.join(keywords[:3])}"
            expanded_queries.append(expanded_query)
        
        # Statistical content expansions
        if Config.ENABLE_STATISTICAL_EXTRACTION:
            statistical_terms = ['results', 'findings', 'significant', 'correlation']
            for term in statistical_terms[:2]:
                expanded_queries.append(f"{query} {term}")
        
        # Journal quality expansions
        if self.journal_manager:
            quality_terms = ['peer-reviewed', 'validated', 'robust']
            for term in quality_terms[:1]:
                expanded_queries.append(f"{query} {term}")
        
        return expanded_queries[:3]  # Limit expansions
    
    def _remove_duplicates_preserve_quality(self, documents: List) -> List:
        """Remove duplicates while preserving highest quality documents"""
        if not documents:
            return documents
        
        # Group by content hash
        content_groups = defaultdict(list)
        for doc in documents:
            content_hash = hash(doc.page_content[:200])
            content_groups[content_hash].append(doc)
        
        # Select best document from each group
        unique_docs = []
        for content_hash, group_docs in content_groups.items():
            if len(group_docs) == 1:
                unique_docs.append(group_docs[0])
            else:
                # Select highest quality document
                best_doc = self._select_highest_quality_doc(group_docs)
                unique_docs.append(best_doc)
        
        return unique_docs
    
    def _select_highest_quality_doc(self, documents: List) -> object:
        """Select the highest quality document from a group"""
        if not self.journal_manager:
            return documents[0]  # Return first if no quality info
        
        best_doc = documents[0]
        best_score = 0.0
        
        for doc in documents:
            # Calculate quality score
            quality_score = doc.metadata.get('journal_quality_score', 0.0)
            statistical_score = doc.metadata.get('statistical_score', 0.0)
            section_score = Config.SECTION_WEIGHTS.get(doc.metadata.get('section_type', 'other'), 0.0)
            
            combined_score = quality_score + statistical_score + section_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_doc = doc
        
        return best_doc
    
    def _apply_quality_filtering(self, documents: List, k: int) -> List:
        """Apply quality filtering to documents"""
        if not documents:
            return documents
        
        # Sort by quality if journal manager available
        if self.journal_manager:
            documents = sorted(documents, key=self._calculate_document_quality_score, reverse=True)
        
        # Apply tier balancing if enabled
        if Config.ENABLE_TIER_BALANCING and self.journal_manager:
            documents = self._apply_tier_balancing(documents)
        
        return documents[:k]
    
    def _calculate_document_quality_score(self, doc) -> float:
        """Calculate overall quality score for a document"""
        quality_score = doc.metadata.get('journal_quality_score', 0.0)
        statistical_score = doc.metadata.get('statistical_score', 0.0)
        section_score = Config.SECTION_WEIGHTS.get(doc.metadata.get('section_type', 'other'), 0.0)
        
        return quality_score + statistical_score + section_score
    
    def _apply_tier_balancing(self, documents: List) -> List:
        """Apply tier balancing to ensure diverse quality representation"""
        if not self.journal_manager:
            return documents
        
        # Group by tier
        tier_docs = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            tier_docs[tier].append(doc)
        
        # Select documents with tier balancing
        balanced_docs = []
        
        # Ensure minimum representation from each tier
        for tier, min_count in Config.MIN_PAPERS_PER_TIER.items():
            tier_documents = tier_docs[tier]
            if tier_documents:
                selected_count = min(min_count, len(tier_documents))
                balanced_docs.extend(tier_documents[:selected_count])
        
        # Add remaining documents by quality
        remaining_docs = [doc for doc in documents if doc not in balanced_docs]
        remaining_docs.sort(key=self._calculate_document_quality_score, reverse=True)
        
        # Calculate remaining slots
        remaining_slots = Config.MAX_PAPERS_PER_RESPONSE - len(balanced_docs)
        balanced_docs.extend(remaining_docs[:remaining_slots])
        
        return balanced_docs
    
    def _filter_multimodal_content(self, documents: List, include_figures: bool, include_tables: bool) -> List:
        """Filter multimodal content based on preferences"""
        filtered_docs = []
        
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'text')
            
            if content_type == 'figure' and not include_figures:
                continue
            elif content_type == 'table' and not include_tables:
                continue
            
            filtered_docs.append(doc)
        
        return filtered_docs
    
    def _extract_sources_with_quality(self, documents: List) -> List[str]:
        """Extract sources with quality information"""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in seen_sources:
                # Add quality information if available
                if self.journal_manager:
                    journal_info = self.journal_manager.match_paper_to_journal(source)
                    if journal_info:
                        quality_info = f" ({journal_info.get('journal', 'unknown')} - {journal_info.get('quartile', 'unknown')})"
                        sources.append(f"{source}{quality_info}")
                    else:
                        sources.append(source)
                else:
                    sources.append(source)
                
                seen_sources.add(source)
        
        return sources
    
    def _generate_quality_metadata(self, documents: List) -> Dict:
        """Generate quality metadata for documents"""
        if not documents:
            return {}
        
        # Calculate quality distribution
        tier_counts = defaultdict(int)
        section_counts = defaultdict(int)
        statistical_count = 0
        
        for doc in documents:
            # Count tiers
            if self.journal_manager:
                source = doc.metadata.get('source', 'unknown')
                tier = self.journal_manager.get_quality_tier(source)
                tier_counts[tier] += 1
            
            # Count sections
            section = doc.metadata.get('section_type', 'other')
            section_counts[section] += 1
            
            # Count statistical content
            if doc.metadata.get('has_statistics', False):
                statistical_count += 1
        
        return {
            'total_documents': len(documents),
            'unique_papers': len(set(doc.metadata.get('source', 'unknown') for doc in documents)),
            'tier_distribution': dict(tier_counts),
            'section_distribution': dict(section_counts),
            'statistical_content_count': statistical_count,
            'quality_score': self._calculate_overall_quality_score(documents),
            'coverage_metrics': self._calculate_document_coverage_metrics(documents)
        }
    
    def _generate_comprehensive_metadata(self, documents: List, tier_requirements: Dict) -> Dict:
        """Generate comprehensive metadata for documents"""
        basic_metadata = self._generate_quality_metadata(documents)
        
        # Add comprehensive-specific metadata
        comprehensive_metadata = {
            **basic_metadata,
            'tier_requirements': tier_requirements,
            'tier_fulfillment': self._analyze_tier_fulfillment(documents, tier_requirements),
            'coverage_completeness': self._calculate_coverage_completeness(documents),
            'diversity_metrics': self._calculate_diversity_metrics(documents),
            'quality_validation': self._validate_quality_requirements(documents)
        }
        
        return comprehensive_metadata
    
    def _ensure_statistical_content(self, documents: List, query: str) -> List:
        """Ensure statistical content is included in document selection"""
        statistical_docs = [doc for doc in documents if doc.metadata.get('has_statistics', False)]
        non_statistical_docs = [doc for doc in documents if not doc.metadata.get('has_statistics', False)]
        
        # Ensure at least 30% statistical content
        min_statistical = max(1, int(len(documents) * 0.3))
        
        if len(statistical_docs) < min_statistical:
            # Need more statistical content - retrieve specifically
            statistical_query = f"{query} statistical significant results"
            additional_docs = self.retrieval_engine.retrieve(statistical_query, k=min_statistical)
            
            # Filter for statistical content
            additional_statistical = [doc for doc in additional_docs if doc.metadata.get('has_statistics', False)]
            statistical_docs.extend(additional_statistical)
        
        # Combine with preference for statistical content
        combined_docs = statistical_docs + non_statistical_docs
        
        return combined_docs
    
    def _analyze_quality_distribution_for_query(self, query: str) -> Dict:
        """Analyze quality distribution for a specific query"""
        if not self.journal_manager:
            return {}
        
        # Retrieve documents for analysis
        docs = self.enhanced_retrieval(query, k=Config.COMPREHENSIVE_RETRIEVAL_K)
        
        # Analyze quality distribution
        tier_analysis = defaultdict(lambda: {'count': 0, 'papers': set()})
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            tier_analysis[tier]['count'] += 1
            tier_analysis[tier]['papers'].add(source)
        
        # Convert to regular dict
        analysis = {}
        for tier, data in tier_analysis.items():
            analysis[tier] = {
                'document_count': data['count'],
                'unique_papers': len(data['papers']),
                'paper_list': list(data['papers'])
            }
        
        return analysis
    
    def _generate_coverage_improvement_suggestions(self, missing_analysis: Dict, 
                                                 coverage_analysis: Dict, 
                                                 quality_analysis: Dict) -> List[str]:
        """Generate suggestions for improving coverage"""
        suggestions = []
        
        # Analyze missing papers
        missing_count = missing_analysis.get('missing_papers', 0)
        if isinstance(missing_count, list):
            missing_count = len(missing_count)
        
        if missing_count > 0:
            suggestions.append(f"Try alternative query formulations - {missing_count} expected papers not retrieved")
        
        # Analyze quality distribution
        if quality_analysis:
            top_tier_count = quality_analysis.get('top', {}).get('unique_papers', 0)
            if top_tier_count < 2:
                suggestions.append("Consider expanding search to include more top-tier publications")
        
        # Analyze coverage
        if coverage_analysis:
            standard_papers = coverage_analysis.get('standard_retrieval', {}).get('unique_papers', 0)
            comprehensive_papers = coverage_analysis.get('comprehensive_retrieval', {}).get('unique_papers', 0)
            
            if comprehensive_papers > standard_papers:
                suggestions.append("Use comprehensive retrieval mode for better coverage")
        
        # System-specific suggestions
        if not self.journal_manager:
            suggestions.append("Enable journal impact ranking for quality-weighted results")
        
        if not Config.ENABLE_COMPREHENSIVE_COVERAGE:
            suggestions.append("Enable comprehensive coverage mode for maximum paper inclusion")
        
        return suggestions
    
    def _suggest_missing_paper_improvements(self, missing_analysis: Dict, 
                                          ranking_explanations: Dict, 
                                          important_papers: List[str]) -> List[str]:
        """Suggest improvements for retrieving missing papers"""
        suggestions = []
        
        # Analyze missing papers
        missing_papers = missing_analysis.get('missing_papers', [])
        
        for paper_info in missing_papers:
            if isinstance(paper_info, dict):
                source = paper_info.get('source', 'unknown')
                status = paper_info.get('status', 'unknown')
                
                if status == 'not_in_corpus':
                    suggestions.append(f"Paper '{source}' not in corpus - check ingestion")
                elif status == 'in_corpus_not_retrieved':
                    suggestions.append(f"Paper '{source}' in corpus but not retrieved - try expanded queries")
        
        # System recommendations
        if len(missing_papers) > len(important_papers) * 0.5:
            suggestions.append("High number of missing papers - consider query expansion or comprehensive mode")
        
        return suggestions
    
    def _generate_alternative_queries_for_missing_papers(self, original_query: str, 
                                                       missing_papers: List) -> List[str]:
        """Generate alternative queries that might retrieve missing papers"""
        alternative_queries = []
        
        # Add broader terms
        broad_query = f"UAM behavioral research {original_query}"
        alternative_queries.append(broad_query)
        
        # Add specific terms
        specific_terms = ['empirical', 'statistical', 'significant', 'findings', 'results']
        for term in specific_terms[:3]:
            alternative_queries.append(f"{original_query} {term}")
        
        # Add methodological terms
        method_terms = ['survey', 'experiment', 'SEM', 'regression', 'analysis']
        for term in method_terms[:2]:
            alternative_queries.append(f"{original_query} {term}")
        
        return alternative_queries
    
    def _generate_system_recommendations(self, missing_analysis: Dict) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Check if journal impact would help
        if not self.journal_manager:
            recommendations.append("Enable journal impact ranking to prioritize high-quality papers")
        
        # Check if comprehensive mode would help
        if not Config.ENABLE_COMPREHENSIVE_COVERAGE:
            recommendations.append("Enable comprehensive coverage mode for maximum paper inclusion")
        
        # Check configuration
        if Config.RETRIEVAL_K < 50:
            recommendations.append("Increase retrieval K for broader initial coverage")
        
        return recommendations
    
    def _calculate_coverage_metrics(self, corpus_stats: Dict, journal_stats: Dict) -> Dict:
        """Calculate comprehensive coverage metrics"""
        metrics = {}
        
        # Basic coverage metrics
        total_papers = corpus_stats.get('total_papers', 0)
        total_chunks = corpus_stats.get('total_chunks', 0)
        
        metrics['papers_per_chunk_ratio'] = total_chunks / total_papers if total_papers > 0 else 0
        metrics['average_chunks_per_paper'] = total_chunks / total_papers if total_papers > 0 else 0
        
        # Quality coverage metrics
        if journal_stats:
            quality_distribution = journal_stats.get('tier_distribution', {})
            total_quality_papers = sum(quality_distribution.values())
            
            metrics['quality_coverage'] = {
                'total_papers_with_quality_info': total_quality_papers,
                'quality_coverage_percentage': (total_quality_papers / total_papers) * 100 if total_papers > 0 else 0,
                'tier_distribution': quality_distribution
            }
        
        return metrics
    
    def _calculate_overall_quality_score(self, documents: List) -> float:
        """Calculate overall quality score for a set of documents"""
        if not documents or not self.journal_manager:
            return 0.0
        
        quality_scores = []
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            quality_score = self.journal_manager.get_paper_quality_score(source)
            quality_scores.append(quality_score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    def _calculate_document_coverage_metrics(self, documents: List) -> Dict:
        """Calculate coverage metrics for a set of documents"""
        if not documents:
            return {}
        
        # Source diversity
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        
        # Section diversity
        sections = set(doc.metadata.get('section_type', 'other') for doc in documents)
        
        # Quality tier diversity
        tiers = set()
        if self.journal_manager:
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                tier = self.journal_manager.get_quality_tier(source)
                tiers.add(tier)
        
        return {
            'source_diversity': len(sources),
            'section_diversity': len(sections),
            'tier_diversity': len(tiers),
            'documents_per_source': len(documents) / len(sources) if sources else 0,
            'statistical_content_ratio': sum(1 for doc in documents if doc.metadata.get('has_statistics', False)) / len(documents)
        }
    
    def _analyze_tier_fulfillment(self, documents: List, tier_requirements: Dict) -> Dict:
        """Analyze how well tier requirements are fulfilled"""
        if not self.journal_manager:
            return {}
        
        # Count documents by tier
        tier_counts = defaultdict(int)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            tier = self.journal_manager.get_quality_tier(source)
            tier_counts[tier] += 1
        
        # Check fulfillment
        fulfillment = {}
        for tier, required in tier_requirements.items():
            actual = tier_counts[tier]
            fulfillment[tier] = {
                'required': required,
                'actual': actual,
                'fulfilled': actual >= required,
                'percentage': (actual / required) * 100 if required > 0 else 100
            }
        
        return fulfillment
    
    def _calculate_coverage_completeness(self, documents: List) -> Dict:
        """Calculate coverage completeness metrics"""
        if not documents:
            return {}
        
        # Section coverage
        sections_covered = set(doc.metadata.get('section_type', 'other') for doc in documents)
        important_sections = {'results', 'findings', 'conclusion', 'discussion'}
        section_completeness = len(sections_covered & important_sections) / len(important_sections)
        
        # Statistical content coverage
        statistical_docs = sum(1 for doc in documents if doc.metadata.get('has_statistics', False))
        statistical_completeness = statistical_docs / len(documents)
        
        return {
            'section_completeness': section_completeness,
            'statistical_completeness': statistical_completeness,
            'overall_completeness': (section_completeness + statistical_completeness) / 2
        }
    
    def _calculate_diversity_metrics(self, documents: List) -> Dict:
        """Calculate diversity metrics for documents"""
        if not documents:
            return {}
        
        # Source diversity (Shannon entropy)
        source_counts = defaultdict(int)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            source_counts[source] += 1
        
        # Calculate Shannon entropy for source diversity
        total_docs = len(documents)
        source_entropy = 0
        for count in source_counts.values():
            if count > 0:
                p = count / total_docs
                source_entropy -= p * (p.log() if hasattr(p, 'log') else 0)
        
        return {
            'source_diversity_entropy': source_entropy,
            'unique_sources': len(source_counts),
            'max_chunks_per_source': max(source_counts.values()) if source_counts else 0,
            'min_chunks_per_source': min(source_counts.values()) if source_counts else 0,
            'diversity_score': len(source_counts) / total_docs if total_docs > 0 else 0
        }
    
    def _validate_quality_requirements(self, documents: List) -> Dict:
        """Validate that quality requirements are met"""
        validation = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        if not documents:
            validation['passed'] = False
            validation['issues'].append("No documents provided")
            return validation
        
        # Check minimum papers
        unique_papers = len(set(doc.metadata.get('source', 'unknown') for doc in documents))
        if unique_papers < Config.MIN_PAPERS_PER_RESPONSE:
            validation['passed'] = False
            validation['issues'].append(f"Only {unique_papers} unique papers (minimum: {Config.MIN_PAPERS_PER_RESPONSE})")
        
        # Check statistical content
        statistical_docs = sum(1 for doc in documents if doc.metadata.get('has_statistics', False))
        statistical_ratio = statistical_docs / len(documents)
        if statistical_ratio < 0.2:  # Expect at least 20% statistical content
            validation['issues'].append(f"Low statistical content ratio: {statistical_ratio:.2f}")
        
        # Check quality tier distribution
        if self.journal_manager:
            tier_counts = defaultdict(int)
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                tier = self.journal_manager.get_quality_tier(source)
                tier_counts[tier] += 1
            
            if tier_counts['top'] + tier_counts['high'] < unique_papers * 0.3:
                validation['issues'].append("Low high-quality paper ratio")
        
        validation['metrics'] = {
            'unique_papers': unique_papers,
            'statistical_ratio': statistical_ratio,
            'total_documents': len(documents)
        }
        
        return validation
    
    def _count_unique_papers(self, documents: List) -> int:
        """Count unique papers in document list"""
        if not documents:
            return 0
        
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        return len(sources)
    
    def get_corpus_statistics(self) -> Dict:
        """Get enhanced corpus statistics with journal impact"""
        try:
            return self.document_ingester.get_corpus_statistics()
        except Exception as e:
            logger.error(f"Error getting corpus statistics: {e}")
            return {}
    
    def diagnose_extraction(self, pdf_path: str) -> Dict:
        """
        Diagnose extraction for a single PDF with journal impact information
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with enhanced extraction diagnosis results
        """
        return self.document_ingester.diagnose_extraction(pdf_path)
    
    def get_system_status(self) -> Dict:
        """Get enhanced system status with journal impact information"""
        # Get multimodal availability safely
        multimodal_available = getattr(self.document_ingester, 'multimodal_available', False)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_ready': self.is_ready,
            'components': {
                'embedding_manager': bool(self.embedding_manager),
                'retrieval_engine': bool(self.retrieval_engine),
                'reranking_engine': bool(self.reranking_engine),
                'response_generator': bool(self.response_generator),
                'document_ingester': bool(self.document_ingester),
                'journal_manager': bool(self.journal_manager),
                'multimodal_available': multimodal_available
            },
            'configuration': {
                'journal_impact_enabled': bool(self.journal_manager),
                'comprehensive_coverage_enabled': Config.ENABLE_COMPREHENSIVE_COVERAGE,
                'tier_balancing_enabled': Config.ENABLE_TIER_BALANCING,
                'paper_diversity_enforcement': Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT,
                'multimodal_enabled': Config.ENABLE_MULTIMODAL,
                'hyde_enabled': Config.ENABLE_HYDE,
                'query_expansion_enabled': Config.ENABLE_QUERY_EXPANSION,
                'chunk_size': Config.CHUNK_SIZE,
                'retrieval_k': Config.RETRIEVAL_K,
                'rerank_k': Config.RERANK_K,
                'min_papers_per_response': Config.MIN_PAPERS_PER_RESPONSE,
                'max_papers_per_response': Config.MAX_PAPERS_PER_RESPONSE
            },
            'corpus_stats': self.get_corpus_statistics(),
            'system_stats': self.system_stats
        }
        
        # Add journal impact status
        if self.journal_manager:
            status['journal_impact_status'] = {
                'metadata_loaded': True,
                'validation_results': self.journal_manager.validate_journal_data(),
                'journal_statistics': self.journal_manager.get_journal_statistics()
            }
        else:
            status['journal_impact_status'] = {
                'metadata_loaded': False,
                'reason': 'Journal impact ranking disabled or metadata not available'
            }
        
        return status
    
    def reload_system(self):
        """Reload the enhanced system components"""
        logger.info("Reloading Enhanced UAM RAG System")
        
        # Reload journal manager if enabled
        if Config.ENABLE_JOURNAL_RANKING:
            self._initialize_journal_manager()
        
        # Reload vector store
        self.db = self.retrieval_engine.load_vector_store()
        
        # Update retrieval engine with new journal manager
        self.retrieval_engine.journal_manager = self.journal_manager
        
        # Update reranking engine with new journal manager
        self.reranking_engine.journal_manager = self.journal_manager
        
        # Update readiness status
        self.is_ready = self._check_system_ready()
        
        logger.info(f"Enhanced system reloaded. Ready: {self.is_ready}, Journal Impact: {bool(self.journal_manager)}")
    
    def clear_cache(self):
        """Clear enhanced system cache"""
        logger.info("Clearing enhanced system cache")
        
        # Clear component caches
        if hasattr(self.retrieval_engine, 'clear_cache'):
            self.retrieval_engine.clear_cache()
        
        if hasattr(self.response_generator, 'clear_cache'):
            self.response_generator.clear_cache()
        
        # Reset statistics
        if hasattr(self.retrieval_engine, 'reset_stats'):
            self.retrieval_engine.reset_stats()
        
        if hasattr(self.reranking_engine, 'reset_stats'):
            self.reranking_engine.reset_stats()
        
        # Reset system statistics
        self.system_stats = {
            'queries_processed': 0,
            'comprehensive_queries': 0,
            'papers_analyzed': 0,
            'journal_impact_enabled': bool(self.journal_manager),
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info("Enhanced system cache cleared")
    
    def __str__(self) -> str:
        """String representation of the enhanced system"""
        multimodal_available = getattr(self.document_ingester, 'multimodal_available', False)
        return f"UAMRAGSystem(ready={self.is_ready}, journal_impact={bool(self.journal_manager)}, multimodal={multimodal_available})"
    
    def __repr__(self) -> str:
        """Detailed representation of the enhanced system"""
        status = self.get_system_status()
        return f"UAMRAGSystem(components={status['components']}, ready={self.is_ready}, journal_impact={bool(self.journal_manager)})"