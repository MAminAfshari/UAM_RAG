# core/ingestion.py
"""
Enhanced Document Ingestion Module with Excel-only filtering
Only ingests PDF files that have corresponding entries in the Excel journal metadata file.
Includes intelligent filename matching and comprehensive reporting.
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .embeddings import EmbeddingManager
from .journal_impact import JournalImpactManager
from config import Config

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Enhanced document ingester with Excel-only filtering and fuzzy matching"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize enhanced document ingester with Excel filtering"""
        self.embedding_manager = embedding_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.OVERLAP_SIZE,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize journal impact manager
        self.journal_manager = None
        if Config.ENABLE_JOURNAL_RANKING:
            self._initialize_journal_manager()
        
        # Check multimodal capabilities
        self.multimodal_available = self._check_multimodal_dependencies()
        
        # Compile regex patterns for better performance
        self._compile_statistical_patterns()
        
        if self.multimodal_available:
            logger.info("Multimodal processing available")
        else:
            logger.warning("Multimodal processing not available - using text-only mode")
        
        logger.info(f"Enhanced document ingester initialized with Excel filtering: {Config.ONLY_INGEST_EXCEL_PAPERS}")
    
    def _initialize_journal_manager(self):
        """Initialize journal impact manager with enhanced error handling"""
        try:
            # Try to load from cache first
            if os.path.exists(Config.JOURNAL_METADATA_CACHE_PATH):
                self.journal_manager = JournalImpactManager()
                if self.journal_manager.load_metadata_cache(Config.JOURNAL_METADATA_CACHE_PATH):
                    logger.info("Journal metadata loaded from cache")
                    return
            
            # Load from Excel file
            if os.path.exists(Config.JOURNAL_METADATA_PATH):
                self.journal_manager = JournalImpactManager(Config.JOURNAL_METADATA_PATH)
                
                # Save to cache for faster future loads
                self.journal_manager.save_metadata_cache(Config.JOURNAL_METADATA_CACHE_PATH)
                logger.info("Journal metadata loaded from Excel and cached")
            else:
                logger.warning(f"Journal metadata file not found: {Config.JOURNAL_METADATA_PATH}")
                
        except Exception as e:
            logger.error(f"Failed to initialize journal manager: {e}")
            self.journal_manager = None
    
    def _check_multimodal_dependencies(self) -> bool:
        """Check if multimodal dependencies are available"""
        try:
            import fitz  # PyMuPDF
            import pdfplumber
            from PIL import Image
            return True
        except ImportError as e:
            logger.warning(f"Multimodal dependencies not available: {e}")
            return False
    
    def _compile_statistical_patterns(self):
        """Compile statistical regex patterns for better performance"""
        self.statistical_patterns = {}
        
        for category, patterns in Config.STATISTICAL_PATTERNS.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in {category}: {pattern} - {e}")
            self.statistical_patterns[category] = compiled_patterns
    
    def validate_pdf_directory(self, pdf_directory: str) -> Dict:
        """
        Validate PDF directory against Excel metadata
        
        Args:
            pdf_directory: Path to directory containing PDF files
            
        Returns:
            Validation results with detailed matching information
        """
        logger.info(f"Validating PDF directory: {pdf_directory}")
        
        if not os.path.exists(pdf_directory):
            return {'error': f'Directory does not exist: {pdf_directory}'}
        
        if not self.journal_manager:
            return {'error': 'Journal manager not initialized - Excel file may be missing'}
        
        # Validate directory with journal manager
        validation_results = self.journal_manager.validate_pdf_directory(pdf_directory)
        
        # Add ingestion-specific information
        if 'error' not in validation_results:
            ready_files = self.journal_manager.get_ingestion_ready_files(pdf_directory)
            validation_results['ingestion_ready'] = len(ready_files)
            validation_results['ready_files'] = ready_files
            
            # Check if ready for ingestion
            if Config.ONLY_INGEST_EXCEL_PAPERS and len(ready_files) == 0:
                validation_results['ready_for_ingestion'] = False
                validation_results['ingestion_error'] = 'No PDF files match Excel entries'
            else:
                validation_results['ready_for_ingestion'] = len(ready_files) > 0
        
        return validation_results
    
    def generate_pre_ingestion_report(self, pdf_directory: str, output_path: str = None) -> str:
        """
        Generate comprehensive pre-ingestion report
        
        Args:
            pdf_directory: Path to directory containing PDF files
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        logger.info(f"Generating pre-ingestion report for {pdf_directory}")
        
        if not self.journal_manager:
            return "Error: Journal manager not initialized"
        
        # Generate detailed report
        report = self.journal_manager.generate_matching_report(pdf_directory, output_path)
        
        # Add ingestion-specific information
        validation_results = self.validate_pdf_directory(pdf_directory)
        
        if 'error' not in validation_results:
            report += f"\nINGESTION READINESS\n"
            report += f"==================\n"
            report += f"Files ready for ingestion: {validation_results['ingestion_ready']}\n"
            report += f"Ready for ingestion: {'Yes' if validation_results['ready_for_ingestion'] else 'No'}\n"
            
            if Config.ONLY_INGEST_EXCEL_PAPERS:
                report += f"Excel-only mode: Enabled (will ignore unmatched PDFs)\n"
            else:
                report += f"Excel-only mode: Disabled (will process all PDFs)\n"
            
            if 'ingestion_error' in validation_results:
                report += f"Ingestion error: {validation_results['ingestion_error']}\n"
        
        return report
    
    def ingest_papers(self, pdf_directory: str, force_text_only: bool = False) -> Dict:
        """
        Enhanced ingestion with Excel-only filtering
        
        Args:
            pdf_directory: Path to directory containing PDF files
            force_text_only: Force text-only processing even if multimodal is available
            
        Returns:
            Dictionary with enhanced ingestion statistics
        """
        logger.info(f"Starting enhanced paper ingestion with Excel filtering from {pdf_directory}")
        
        # Validate directory
        if not os.path.exists(pdf_directory):
            raise ValueError(f"Directory does not exist: {pdf_directory}")
        
        # Validate with journal manager
        validation_results = self.validate_pdf_directory(pdf_directory)
        
        if 'error' in validation_results:
            raise ValueError(f"Validation failed: {validation_results['error']}")
        
        # Get files to process
        if Config.ONLY_INGEST_EXCEL_PAPERS:
            if not validation_results['ready_for_ingestion']:
                raise ValueError(f"No PDF files ready for ingestion: {validation_results.get('ingestion_error', 'Unknown error')}")
            
            pdf_files = validation_results['ready_files']
            logger.info(f"Excel-only mode: Processing {len(pdf_files)} matched files")
        else:
            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
            logger.info(f"Processing all {len(pdf_files)} PDF files")
        
        if not pdf_files:
            raise ValueError(f"No PDF files found for processing")
        
        # Pre-analyze journal coverage
        journal_coverage = self._analyze_journal_coverage(pdf_files)
        
        # Choose processing method
        use_multimodal = (
            Config.ENABLE_MULTIMODAL and 
            self.multimodal_available and 
            not force_text_only
        )
        
        if use_multimodal:
            logger.info("Using multimodal processing with Excel filtering")
            return self._ingest_multimodal(pdf_directory, pdf_files, journal_coverage, validation_results)
        else:
            logger.info("Using text-only processing with Excel filtering")
            return self._ingest_text_only(pdf_directory, pdf_files, journal_coverage, validation_results)
    
    def _analyze_journal_coverage(self, pdf_files: List[str]) -> Dict:
        """Analyze journal coverage for the filtered PDF files"""
        coverage = {
            'total_papers': len(pdf_files),
            'papers_with_journal_info': 0,
            'papers_without_journal_info': 0,
            'tier_distribution': {'top': 0, 'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
            'journal_distribution': {},
            'missing_papers': [],
            'matched_papers': []
        }
        
        if not self.journal_manager:
            coverage['papers_without_journal_info'] = len(pdf_files)
            coverage['missing_papers'] = pdf_files
            return coverage
        
        for filename in pdf_files:
            journal_info = self.journal_manager.match_paper_to_journal(filename)
            if journal_info:
                coverage['papers_with_journal_info'] += 1
                coverage['matched_papers'].append(filename)
                
                # Update tier distribution
                tier = self.journal_manager.get_quality_tier(filename)
                coverage['tier_distribution'][tier] += 1
                
                # Update journal distribution
                journal = journal_info['journal']
                coverage['journal_distribution'][journal] = coverage['journal_distribution'].get(journal, 0) + 1
            else:
                coverage['papers_without_journal_info'] += 1
                coverage['missing_papers'].append(filename)
        
        logger.info(f"Journal coverage analysis: {coverage['papers_with_journal_info']}/{len(pdf_files)} papers matched")
        return coverage
    
    def _ingest_text_only(self, pdf_directory: str, pdf_files: List[str], 
                         journal_coverage: Dict, validation_results: Dict) -> Dict:
        """Process papers using enhanced text-only extraction with Excel filtering"""
        all_documents = []
        stats = {
            'processing_mode': 'text_only_excel_filtered',
            'total_papers': len(pdf_files),
            'successful_papers': 0,
            'failed_papers': 0,
            'total_chunks': 0,
            'text_chunks': 0,
            'papers_processed': [],
            'papers_skipped': [],
            'errors': [],
            'section_classification_stats': {},
            'statistical_content_stats': {},
            'journal_impact_stats': {
                'papers_with_journal_info': journal_coverage['papers_with_journal_info'],
                'papers_without_journal_info': journal_coverage['papers_without_journal_info'],
                'tier_distribution': journal_coverage['tier_distribution'].copy(),
                'journal_distribution': journal_coverage['journal_distribution'].copy(),
                'quality_score_distribution': {'top': 0, 'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
                'missing_papers': journal_coverage['missing_papers'].copy()
            },
            'excel_filtering_stats': {
                'total_pdfs_in_directory': len([f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]),
                'pdfs_matched_to_excel': validation_results.get('matched_count', 0),
                'pdfs_processed': 0,
                'pdfs_skipped': 0,
                'match_rate': validation_results.get('match_rate', 0)
            }
        }
        
        for i, filename in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {filename}")
            
            # Check if file should be processed
            if Config.ONLY_INGEST_EXCEL_PAPERS and not self._should_process_file(filename):
                logger.info(f"Skipping {filename} - not in Excel or no match")
                stats['papers_skipped'].append(filename)
                stats['excel_filtering_stats']['pdfs_skipped'] += 1
                continue
            
            try:
                filepath = os.path.join(pdf_directory, filename)
                
                # Load PDF with enhanced text extraction
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                
                if not pages:
                    error_msg = f"No pages loaded from {filename}"
                    logger.warning(error_msg)
                    stats['errors'].append({'filename': filename, 'error': error_msg})
                    stats['failed_papers'] += 1
                    continue
                
                # Enhanced text quality check
                total_text = "\n".join([page.page_content for page in pages])
                if len(total_text.strip()) < 100:
                    error_msg = f"Very little text extracted from {filename} ({len(total_text)} chars)"
                    logger.warning(error_msg)
                    stats['errors'].append({'filename': filename, 'error': error_msg})
                    stats['failed_papers'] += 1
                    continue
                
                # Create enhanced metadata with journal impact
                metadata = self._create_enhanced_metadata(filename)
                
                # Enhanced chunking with journal impact integration
                chunks = self._create_enhanced_text_chunks(pages, metadata, total_text)
                
                if not chunks:
                    error_msg = f"No chunks created from {filename}"
                    logger.warning(error_msg)
                    stats['errors'].append({'filename': filename, 'error': error_msg})
                    stats['failed_papers'] += 1
                    continue
                
                # Update statistics
                all_documents.extend(chunks)
                stats['successful_papers'] += 1
                stats['total_chunks'] += len(chunks)
                stats['text_chunks'] += len(chunks)
                stats['papers_processed'].append(filename)
                stats['excel_filtering_stats']['pdfs_processed'] += 1
                
                # Track section classification and journal impact
                self._update_enhanced_stats(chunks, stats, metadata)
                
                logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg)
                stats['failed_papers'] += 1
                stats['errors'].append({'filename': filename, 'error': str(e)})
        
        # Create vector store if we have documents
        if all_documents:
            self._create_and_save_vector_store(all_documents, stats)
        else:
            raise ValueError("No documents were successfully processed")
        
        # Log final statistics
        logger.info(f"Excel filtering results: {stats['excel_filtering_stats']}")
        logger.info(f"Processed {stats['successful_papers']}/{stats['total_papers']} papers")
        
        return stats
    
    def _should_process_file(self, filename: str) -> bool:
        """Check if file should be processed based on Excel matching"""
        if not Config.ONLY_INGEST_EXCEL_PAPERS:
            return True
        
        if not self.journal_manager:
            return True  # Process all if no journal manager
        
        # Check if file has journal info
        journal_info = self.journal_manager.match_paper_to_journal(filename)
        return journal_info is not None
    
    def _create_enhanced_metadata(self, filename: str) -> Dict:
        """Create enhanced metadata with journal impact information"""
        import re
        
        # Extract year from filename
        year_match = re.search(r'(\d{4})', filename)
        year = year_match.group(1) if year_match else "unknown"
        
        # Create citation key
        base_name = filename.replace('.pdf', '').lower()
        citation_key = re.sub(r'^(paper|article|study|research)[-_]?', '', base_name)
        
        # Base metadata
        metadata = {
            'source': citation_key,
            'title': f"Paper: {filename}",
            'year': year,
            'filename': filename,
            'processing_timestamp': datetime.now().isoformat(),
            'excel_matched': Config.ONLY_INGEST_EXCEL_PAPERS
        }
        
        # Add journal impact information
        if self.journal_manager:
            journal_info = self.journal_manager.match_paper_to_journal(filename)
            if journal_info:
                # Get Excel name if different from filename
                excel_name = None
                if filename in self.journal_manager.pdf_to_excel_map:
                    excel_name = self.journal_manager.pdf_to_excel_map[filename]
                
                metadata.update({
                    'journal': journal_info['journal'],
                    'journal_quartile': journal_info['quartile'],
                    'journal_impact_score': journal_info['impact_score'],
                    'journal_normalized_impact': journal_info['normalized_impact_score'],
                    'journal_quality_tier': self.journal_manager.get_quality_tier(filename),
                    'journal_quality_score': self.journal_manager.get_paper_quality_score(filename),
                    'journal_boost_multiplier': self.journal_manager.get_journal_boost_multiplier(filename),
                    'has_journal_info': True,
                    'excel_paper_name': excel_name
                })
            else:
                metadata.update({
                    'journal': 'unknown',
                    'journal_quartile': 'unknown',
                    'journal_impact_score': 0.0,
                    'journal_normalized_impact': 0.0,
                    'journal_quality_tier': 'unknown',
                    'journal_quality_score': 0.2,
                    'journal_boost_multiplier': 0.2,
                    'has_journal_info': False,
                    'excel_paper_name': None
                })
        else:
            metadata.update({
                'journal': 'unknown',
                'journal_quartile': 'unknown',
                'journal_impact_score': 0.0,
                'journal_normalized_impact': 0.0,
                'journal_quality_tier': 'unknown',
                'journal_quality_score': 0.2,
                'journal_boost_multiplier': 0.2,
                'has_journal_info': False,
                'excel_paper_name': None
            })
        
        return metadata
    
    def _create_enhanced_text_chunks(self, pages: List[Document], metadata: Dict, full_text: str) -> List[Document]:
        """Create enhanced text chunks with journal impact integration"""
        chunks = []
        
        # Analyze document structure
        doc_structure = self._analyze_document_structure(full_text)
        
        # Get journal impact boost
        journal_boost = metadata.get('journal_boost_multiplier', 0.2)
        quality_tier = metadata.get('journal_quality_tier', 'unknown')
        
        # Process each page with context
        for page_num, page in enumerate(pages):
            page_text = page.page_content
            
            if len(page_text.strip()) < 50:
                continue
                
            # Split page into chunks
            page_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                if len(chunk_text.strip()) < 50:
                    continue
                
                # Enhanced section detection
                section_info = self._detect_section_type_enhanced(
                    chunk_text, 
                    page_num, 
                    len(pages), 
                    doc_structure
                )
                
                # Statistical content detection
                statistical_info = self._detect_statistical_content(chunk_text)
                
                # Calculate journal-aware boost
                journal_section_boost = self._calculate_journal_section_boost(
                    section_info['section'], 
                    statistical_info['has_statistics'], 
                    journal_boost,
                    quality_tier
                )
                
                # Create enhanced chunk
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_type': 'text',
                        'content_type': 'text',
                        'chunk_id': f"{metadata['source']}_p{page_num + 1}_c{chunk_idx}",
                        'page_number': page_num + 1,
                        'section_type': section_info['section'],
                        'section_confidence': section_info['confidence'],
                        'has_statistics': statistical_info['has_statistics'],
                        'statistical_patterns': statistical_info['patterns'],
                        'statistical_score': statistical_info['score'],
                        'position_in_document': (page_num + 1) / len(pages),
                        'journal_section_boost': journal_section_boost,
                        'combined_quality_score': self._calculate_combined_quality_score(
                            section_info, statistical_info, metadata
                        )
                    }
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _calculate_journal_section_boost(self, section_type: str, has_statistics: bool, 
                                       journal_boost: float, quality_tier: str) -> float:
        """Calculate journal-aware section boost"""
        # Base section boost
        base_boost = Config.JOURNAL_SECTION_BOOSTS.get(section_type, 0.0)
        
        # Statistical content boost
        if has_statistics:
            statistical_boost = Config.TIER_STATISTICAL_BOOSTS.get(quality_tier, 1.0)
            base_boost *= statistical_boost
        
        # Journal quality boost
        final_boost = base_boost * journal_boost
        
        return final_boost
    
    def _calculate_combined_quality_score(self, section_info: Dict, statistical_info: Dict, 
                                        metadata: Dict) -> float:
        """Calculate combined quality score for chunk"""
        # Section quality
        section_weight = Config.SECTION_WEIGHTS.get(section_info['section'], 0.0)
        section_score = section_weight * section_info['confidence']
        
        # Statistical quality
        statistical_score = statistical_info['score'] * Config.STATISTICAL_CONTENT_WEIGHT
        
        # Journal quality
        journal_score = metadata.get('journal_quality_score', 0.2) * Config.JOURNAL_IMPACT_WEIGHT
        
        # Combined score
        combined_score = section_score + statistical_score + journal_score
        
        return min(combined_score, 1.0)  # Cap at 1.0
    
    def _analyze_document_structure(self, full_text: str) -> Dict:
        """Analyze overall document structure for better section detection"""
        structure = {
            'length': len(full_text),
            'has_abstract': False,
            'has_introduction': False,
            'has_methodology': False,
            'has_results': False,
            'has_discussion': False,
            'has_conclusion': False,
            'section_markers': [],
            'statistical_density': 0,
            'journal_quality_indicators': []
        }
        
        text_lower = full_text.lower()
        
        # Check for major sections
        structure['has_abstract'] = any(keyword in text_lower for keyword in ['abstract', 'summary'])
        structure['has_introduction'] = any(keyword in text_lower for keyword in ['introduction', 'background'])
        structure['has_methodology'] = any(keyword in text_lower for keyword in ['methodology', 'methods', 'procedure'])
        structure['has_results'] = any(keyword in text_lower for keyword in ['results', 'findings', 'analysis'])
        structure['has_discussion'] = any(keyword in text_lower for keyword in ['discussion', 'interpretation'])
        structure['has_conclusion'] = any(keyword in text_lower for keyword in ['conclusion', 'conclusions'])
        
        # Find section markers
        section_patterns = [
            r'\n\s*\d+\.?\s+(introduction|background|literature review|methodology|methods|results|findings|discussion|conclusion)',
            r'\n\s*[A-Z\s]{3,20}\n',  # ALL CAPS headers
            r'\n\s*\d+\.\d+\s+[A-Z][a-z\s]+\n'  # Numbered subsections
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            structure['section_markers'].extend([match.group() for match in matches])
        
        # Calculate statistical density
        statistical_matches = 0
        for patterns in self.statistical_patterns.values():
            for pattern in patterns:
                statistical_matches += len(pattern.findall(full_text))
        
        structure['statistical_density'] = statistical_matches / max(len(full_text), 1) * 1000  # per 1000 chars
        
        return structure
    
    def _detect_section_type_enhanced(self, text: str, page_num: int, total_pages: int, doc_structure: Dict) -> Dict:
        """Enhanced section type detection with journal impact awareness"""
        text_lower = text.lower()
        text_clean = re.sub(r'\s+', ' ', text_lower).strip()
        
        # Position-based heuristics
        position_ratio = (page_num + 1) / total_pages
        is_early = position_ratio < 0.3
        is_middle = 0.3 <= position_ratio <= 0.7
        is_late = position_ratio > 0.7
        
        section_scores = {}
        
        # Score each section type
        for section_type, keywords in Config.SECTION_KEYWORDS.items():
            score = 0
            
            # Keyword matching with proximity weighting
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Exact matches at start of text (higher weight)
                if text_clean.startswith(keyword_lower):
                    score += 15
                
                # Exact matches anywhere (medium weight)
                if keyword_lower in text_clean:
                    score += 8
                
                # Partial matches (lower weight)
                if any(word in text_clean for word in keyword_lower.split()):
                    score += 3
            
            # Position-based adjustments
            if section_type in ['results', 'findings', 'analysis']:
                if is_middle or is_late:
                    score *= 1.8
                else:
                    score *= 0.6
            
            elif section_type in ['conclusion', 'conclusions', 'discussion']:
                if is_late:
                    score *= 2.0
                elif is_middle:
                    score *= 1.4
                else:
                    score *= 0.4
            
            elif section_type in ['introduction', 'background', 'abstract']:
                if is_early:
                    score *= 1.6
                else:
                    score *= 0.7
            
            elif section_type in ['methodology', 'methods']:
                if is_early or is_middle:
                    score *= 1.4
                else:
                    score *= 0.8
            
            section_scores[section_type] = score
        
        # Statistical content boost for results sections
        statistical_info = self._detect_statistical_content(text)
        if statistical_info['has_statistics']:
            section_scores['results'] = section_scores.get('results', 0) + 20
            section_scores['findings'] = section_scores.get('findings', 0) + 20
            section_scores['analysis'] = section_scores.get('analysis', 0) + 15
        
        # Header detection boost
        header_patterns = [
            r'^\s*\d+\.?\s*(introduction|background|literature|methodology|methods|results|findings|discussion|conclusion)',
            r'^\s*[A-Z\s]{3,20}$',
            r'^\s*\d+\.\d+\s+[A-Z][a-z\s]+$'
        ]
        
        for pattern in header_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                header_text = match.group().lower()
                for section_type in section_scores:
                    if any(keyword in header_text for keyword in Config.SECTION_KEYWORDS.get(section_type, [])):
                        section_scores[section_type] += 25
        
        # Determine best section
        if section_scores:
            best_section = max(section_scores, key=section_scores.get)
            confidence = section_scores[best_section] / max(sum(section_scores.values()), 1)
            
            # Minimum confidence threshold
            if confidence < 0.08:
                best_section = 'other'
                confidence = 0.08
        else:
            best_section = 'other'
            confidence = 0.08
        
        return {
            'section': best_section,
            'confidence': confidence,
            'all_scores': section_scores
        }
    
    def _detect_statistical_content(self, text: str) -> Dict:
        """Detect statistical content in text with enhanced patterns"""
        statistical_info = {
            'has_statistics': False,
            'patterns': [],
            'score': 0,
            'categories': {}
        }
        
        total_matches = 0
        
        for category, patterns in self.statistical_patterns.items():
            matches = 0
            found_patterns = []
            
            for pattern in patterns:
                pattern_matches = pattern.findall(text)
                if pattern_matches:
                    matches += len(pattern_matches)
                    found_patterns.extend(pattern_matches)
            
            if matches > 0:
                statistical_info['categories'][category] = matches
                statistical_info['patterns'].extend(found_patterns)
                total_matches += matches
        
        statistical_info['has_statistics'] = total_matches > 0
        statistical_info['score'] = min(total_matches * 0.15, 1.0)
        
        return statistical_info
    
    def _update_enhanced_stats(self, chunks: List[Document], stats: Dict, metadata: Dict):
        """Update enhanced statistics with journal impact information"""
        section_counts = {}
        statistical_counts = {}
        
        quality_tier = metadata.get('journal_quality_tier', 'unknown')
        
        for chunk in chunks:
            section = chunk.metadata.get('section_type', 'other')
            section_counts[section] = section_counts.get(section, 0) + 1
            
            if chunk.metadata.get('has_statistics', False):
                statistical_counts[section] = statistical_counts.get(section, 0) + 1
        
        stats['section_classification_stats'] = section_counts
        stats['statistical_content_stats'] = statistical_counts
        
        # Update journal impact stats
        stats['journal_impact_stats']['quality_score_distribution'][quality_tier] += 1
    
    def _create_and_save_vector_store(self, documents: List[Document], stats: Dict):
        """Create and save vector store with enhanced journal impact metadata"""
        from .retrieval import RetrievalEngine
        
        logger.info(f"Creating vector store with {len(documents)} enhanced documents with journal impact")
        
        retrieval_engine = RetrievalEngine(self.embedding_manager)
        retrieval_engine.create_vector_store(documents)
        
        # Save enhanced metadata
        self._save_enhanced_ingestion_metadata(stats, documents)
        
        logger.info("Vector store created and enhanced metadata with journal impact saved")
    
    def _save_enhanced_ingestion_metadata(self, stats: Dict, documents: List[Document]):
        """Save enhanced ingestion metadata with journal impact and Excel filtering information"""
        # Calculate enhanced statistics
        section_distribution = {}
        keyword_distribution = {}
        statistical_distribution = {}
        confidence_distribution = {}
        journal_impact_distribution = {}
        quality_tier_distribution = {}
        
        for doc in documents:
            # Section distribution
            section = doc.metadata.get('section_type', 'other')
            section_distribution[section] = section_distribution.get(section, 0) + 1
            
            # Statistical content distribution
            if doc.metadata.get('has_statistics', False):
                statistical_distribution[section] = statistical_distribution.get(section, 0) + 1
            
            # Confidence distribution
            confidence = doc.metadata.get('section_confidence', 0)
            confidence_range = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 9}%"
            confidence_distribution[confidence_range] = confidence_distribution.get(confidence_range, 0) + 1
            
            # Journal impact distribution
            journal = doc.metadata.get('journal', 'unknown')
            journal_impact_distribution[journal] = journal_impact_distribution.get(journal, 0) + 1
            
            # Quality tier distribution
            quality_tier = doc.metadata.get('journal_quality_tier', 'unknown')
            quality_tier_distribution[quality_tier] = quality_tier_distribution.get(quality_tier, 0) + 1
            
            # Keyword distribution (basic)
            content_lower = doc.page_content.lower()
            for keyword_category, keywords in Config.UAM_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        keyword_distribution[keyword] = keyword_distribution.get(keyword, 0) + 1
        
        # Enhanced metadata
        metadata = {
            'ingestion_timestamp': datetime.now().isoformat(),
            'ingestion_stats': stats,
            'total_documents': len(documents),
            'total_papers': stats.get('successful_papers', 0),
            'total_chunks': len(documents),
            'section_distribution': section_distribution,
            'statistical_content_distribution': statistical_distribution,
            'section_confidence_distribution': confidence_distribution,
            'journal_impact_distribution': journal_impact_distribution,
            'quality_tier_distribution': quality_tier_distribution,
            'keyword_distribution': dict(sorted(keyword_distribution.items(), key=lambda x: x[1], reverse=True)[:50]),
            'papers_processed': stats.get('papers_processed', []),
            'papers_skipped': stats.get('papers_skipped', []),
            'configuration': {
                'chunk_size': Config.CHUNK_SIZE,
                'overlap_size': Config.OVERLAP_SIZE,
                'embedding_model': Config.EMBEDDING_MODEL,
                'processing_mode': stats.get('processing_mode', 'unknown'),
                'section_weights': Config.SECTION_WEIGHTS,
                'statistical_patterns_count': sum(len(patterns) for patterns in Config.STATISTICAL_PATTERNS.values()),
                'journal_impact_enabled': bool(self.journal_manager),
                'journal_metadata_path': Config.JOURNAL_METADATA_PATH if Config.ENABLE_JOURNAL_RANKING else None,
                'excel_only_mode': Config.ONLY_INGEST_EXCEL_PAPERS,
                'fuzzy_matching_threshold': Config.FUZZY_MATCHING_THRESHOLD
            }
        }
        
        # Add journal statistics if available
        if self.journal_manager:
            metadata['journal_statistics'] = self.journal_manager.get_journal_statistics()
        
        with open(Config.METADATA_DB_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Enhanced metadata saved with journal impact and Excel filtering: {len(section_distribution)} section types, {len(journal_impact_distribution)} journals")
    
    def get_corpus_statistics(self) -> Dict:
        """Get enhanced corpus statistics with journal impact and Excel filtering"""
        try:
            with open(Config.METADATA_DB_PATH, 'r') as f:
                stats = json.load(f)
            
            # Add real-time journal statistics if manager is available
            if self.journal_manager:
                stats['real_time_journal_stats'] = self.journal_manager.get_journal_statistics()
            
            return stats
        except FileNotFoundError:
            return {}
    
    def diagnose_extraction(self, pdf_path: str) -> Dict:
        """Enhanced diagnosis with journal impact and Excel matching information"""
        filename = os.path.basename(pdf_path)
        
        try:
            # Test enhanced text extraction
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            total_text = "\n".join([page.page_content for page in pages])
            
            # Analyze document structure
            doc_structure = self._analyze_document_structure(total_text)
            
            # Get journal impact information
            journal_info = None
            excel_match_info = None
            
            if self.journal_manager:
                journal_info = self.journal_manager.match_paper_to_journal(filename)
                
                # Check Excel matching
                if filename in self.journal_manager.pdf_to_excel_map:
                    excel_match_info = {
                        'matched': True,
                        'excel_name': self.journal_manager.pdf_to_excel_map[filename],
                        'match_type': 'direct'
                    }
                else:
                    # Try fuzzy matching
                    excel_papers = list(self.journal_manager.paper_to_journal_map.keys())
                    best_match = self.journal_manager._find_best_excel_match(filename, excel_papers, Config.FUZZY_MATCHING_THRESHOLD)
                    if best_match:
                        excel_match_info = {
                            'matched': True,
                            'excel_name': best_match[0],
                            'match_type': 'fuzzy',
                            'match_score': best_match[1]
                        }
                    else:
                        excel_match_info = {
                            'matched': False,
                            'reason': 'No Excel match found'
                        }
            
            # Test section detection on sample chunks
            sample_chunks = self.text_splitter.split_text(total_text)[:5]
            section_analysis = []
            
            for i, chunk in enumerate(sample_chunks):
                section_info = self._detect_section_type_enhanced(
                    chunk, i, len(sample_chunks), doc_structure
                )
                statistical_info = self._detect_statistical_content(chunk)
                
                section_analysis.append({
                    'chunk_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'section_type': section_info['section'],
                    'confidence': section_info['confidence'],
                    'has_statistics': statistical_info['has_statistics'],
                    'statistical_score': statistical_info['score']
                })
            
            result = {
                'filename': filename,
                'success': True,
                'text_extraction': {
                    'pages': len(pages),
                    'total_text_length': len(total_text),
                    'average_page_length': len(total_text) / len(pages) if pages else 0
                },
                'document_structure': doc_structure,
                'section_analysis': section_analysis,
                'journal_impact_info': {
                    'has_journal_info': bool(journal_info),
                    'journal_info': journal_info,
                    'quality_tier': self.journal_manager.get_quality_tier(filename) if self.journal_manager else 'unknown',
                    'quality_score': self.journal_manager.get_paper_quality_score(filename) if self.journal_manager else 0.2
                },
                'excel_matching_info': excel_match_info,
                'ingestion_eligibility': {
                    'will_be_processed': self._should_process_file(filename),
                    'excel_only_mode': Config.ONLY_INGEST_EXCEL_PAPERS,
                    'reason': 'In Excel' if excel_match_info and excel_match_info['matched'] else 'Not in Excel'
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'filename': filename,
                'success': False,
                'error': str(e)
            }