# core/journal_impact.py
"""
Enhanced Journal Impact Metadata Handler with Fuzzy PDF Matching
Handles Excel-based journal metadata with intelligent PDF filename matching
to ensure all papers in Excel are included even with different filenames.
"""

import os
import json
import logging
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)


class JournalImpactManager:
    """Enhanced journal impact manager with intelligent PDF filename matching"""
    
    def __init__(self, excel_path: str = None):
        """Initialize enhanced journal impact manager"""
        self.excel_path = excel_path
        self.journal_metadata = {}
        self.paper_to_journal_map = {}
        self.pdf_to_excel_map = {}  # NEW: Maps PDF filenames to Excel entries
        self.excel_to_pdf_map = {}  # NEW: Maps Excel entries to PDF filenames
        self.unmatched_excel_papers = []  # NEW: Papers in Excel but no PDF found
        self.unmatched_pdf_files = []     # NEW: PDF files not in Excel
        self.journal_statistics = {}
        self.quality_tiers = {}
        
        # Enhanced column mappings
        self.column_mappings = {
            'paper_name': ['paper_name', 'paper', 'filename', 'title', 'paper_title', 'file_name'],
            'journal': ['journal', 'journal_name', 'publication', 'venue', 'publisher'],
            'quartile': ['quartile', 'q', 'tier', 'quality_tier', 'journal_quartile'],
            'impact_score': ['impact_score', 'impact_factor', 'if', 'score', 'rating', 'impact']
        }
        
        if excel_path and os.path.exists(excel_path):
            self.load_excel_metadata(excel_path)
        
        logger.info("Enhanced journal impact manager initialized with fuzzy matching")
    
    def load_excel_metadata(self, excel_path: str) -> Dict:
        """Load enhanced journal metadata from Excel file"""
        try:
            logger.info(f"Loading enhanced journal metadata from {excel_path}")
            
            # Try to read Excel file
            try:
                df = pd.read_excel(excel_path)
                logger.info(f"Excel file loaded successfully. Shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
            except Exception as e:
                logger.error(f"Failed to read Excel file: {e}")
                return {'error': f'Failed to read Excel file: {e}'}
            
            # Detect column structure
            detected_columns = self._detect_column_structure(df)
            if not detected_columns:
                return {'error': 'Could not detect required columns in Excel file'}
            
            logger.info(f"Detected columns: {detected_columns}")
            
            # Process the data
            processed_data = self._process_excel_data(df, detected_columns)
            
            # Create mappings
            self._create_paper_mappings(processed_data)
            
            # Calculate statistics
            self._calculate_journal_statistics()
            
            # Create quality tiers
            self._create_quality_tiers()
            
            logger.info(f"Loaded metadata for {len(self.journal_metadata)} journals")
            logger.info(f"Mapped {len(self.paper_to_journal_map)} papers to journals")
            
            return {
                'success': True,
                'journals_loaded': len(self.journal_metadata),
                'papers_mapped': len(self.paper_to_journal_map),
                'detected_columns': detected_columns,
                'quality_tiers': self.quality_tiers,
                'excel_papers': list(self.paper_to_journal_map.keys())
            }
            
        except Exception as e:
            logger.error(f"Error loading journal metadata: {e}")
            return {'error': str(e)}
    
    def match_pdf_files_to_excel(self, pdf_directory: str, fuzzy_threshold: int = 80) -> Dict:
        """
        Match PDF files in directory to Excel entries with fuzzy matching
        
        Args:
            pdf_directory: Directory containing PDF files
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            
        Returns:
            Dictionary with matching results
        """
        logger.info(f"Matching PDF files in {pdf_directory} to Excel entries")
        
        if not os.path.exists(pdf_directory):
            return {'error': f'Directory does not exist: {pdf_directory}'}
        
        # Get PDF files
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Get Excel paper names
        excel_papers = list(self.paper_to_journal_map.keys())
        logger.info(f"Found {len(excel_papers)} papers in Excel")
        
        # Clear previous mappings
        self.pdf_to_excel_map = {}
        self.excel_to_pdf_map = {}
        self.unmatched_excel_papers = []
        self.unmatched_pdf_files = []
        
        # Match each PDF file to Excel entries
        matched_count = 0
        
        for pdf_file in pdf_files:
            best_match = self._find_best_excel_match(pdf_file, excel_papers, fuzzy_threshold)
            
            if best_match:
                excel_name, score = best_match
                self.pdf_to_excel_map[pdf_file] = excel_name
                self.excel_to_pdf_map[excel_name] = pdf_file
                matched_count += 1
                logger.info(f"Matched: '{pdf_file}' -> '{excel_name}' (score: {score})")
            else:
                self.unmatched_pdf_files.append(pdf_file)
                logger.warning(f"No match found for PDF: {pdf_file}")
        
        # Find unmatched Excel papers
        for excel_paper in excel_papers:
            if excel_paper not in self.excel_to_pdf_map:
                self.unmatched_excel_papers.append(excel_paper)
        
        # Results
        results = {
            'total_pdfs': len(pdf_files),
            'total_excel_papers': len(excel_papers),
            'matched_count': matched_count,
            'unmatched_pdfs': len(self.unmatched_pdf_files),
            'unmatched_excel_papers': len(self.unmatched_excel_papers),
            'match_rate': (matched_count / len(excel_papers)) * 100 if excel_papers else 0,
            'pdf_to_excel_map': self.pdf_to_excel_map,
            'excel_to_pdf_map': self.excel_to_pdf_map,
            'unmatched_pdf_files': self.unmatched_pdf_files,
            'unmatched_excel_papers': self.unmatched_excel_papers
        }
        
        logger.info(f"Matching complete: {matched_count}/{len(excel_papers)} papers matched ({results['match_rate']:.1f}%)")
        
        return results
    
    def _find_best_excel_match(self, pdf_filename: str, excel_papers: List[str], threshold: int) -> Optional[Tuple[str, int]]:
        """Find the best Excel match for a PDF filename"""
        # Clean PDF filename
        clean_pdf = self._clean_filename(pdf_filename)
        
        # Try exact match first
        for excel_paper in excel_papers:
            clean_excel = self._clean_filename(excel_paper)
            if clean_pdf == clean_excel:
                return excel_paper, 100
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        for excel_paper in excel_papers:
            # Test multiple comparison strategies
            scores = [
                fuzz.ratio(clean_pdf, self._clean_filename(excel_paper)),
                fuzz.partial_ratio(clean_pdf, self._clean_filename(excel_paper)),
                fuzz.token_sort_ratio(clean_pdf, self._clean_filename(excel_paper)),
                fuzz.token_set_ratio(clean_pdf, self._clean_filename(excel_paper))
            ]
            
            # Use the highest score
            max_score = max(scores)
            
            if max_score >= threshold and max_score > best_score:
                best_score = max_score
                best_match = excel_paper
        
        return (best_match, best_score) if best_match else None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for better matching"""
        # Remove file extension
        clean = filename.replace('.pdf', '').replace('.PDF', '')
        
        # Remove common prefixes
        prefixes = ['paper', 'article', 'study', 'research', 'doc', 'document']
        for prefix in prefixes:
            clean = re.sub(rf'^{prefix}[-_\s]*', '', clean, flags=re.IGNORECASE)
        
        # Remove special characters and normalize
        clean = re.sub(r'[^\w\s]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip().lower()
        
        return clean
    
    def get_matched_pdf_files(self) -> List[str]:
        """Get list of PDF files that have Excel matches"""
        return list(self.pdf_to_excel_map.keys())
    
    def get_unmatched_pdf_files(self) -> List[str]:
        """Get list of PDF files that don't have Excel matches"""
        return self.unmatched_pdf_files.copy()
    
    def get_missing_excel_papers(self) -> List[str]:
        """Get list of Excel papers that don't have PDF matches"""
        return self.unmatched_excel_papers.copy()
    
    def validate_pdf_directory(self, pdf_directory: str) -> Dict:
        """
        Validate PDF directory against Excel entries
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            Validation results with recommendations
        """
        logger.info(f"Validating PDF directory: {pdf_directory}")
        
        # Match files
        match_results = self.match_pdf_files_to_excel(pdf_directory)
        
        if 'error' in match_results:
            return match_results
        
        # Generate recommendations
        recommendations = []
        
        if match_results['unmatched_excel_papers']:
            recommendations.append(f"Missing PDFs: {len(match_results['unmatched_excel_papers'])} papers in Excel have no corresponding PDF files")
        
        if match_results['unmatched_pdfs']:
            recommendations.append(f"Extra PDFs: {len(match_results['unmatched_pdfs'])} PDF files are not in Excel and will be ignored")
        
        if match_results['match_rate'] < 90:
            recommendations.append(f"Low match rate: {match_results['match_rate']:.1f}% - consider renaming files or updating Excel")
        
        if match_results['match_rate'] == 100:
            recommendations.append("Perfect match! All Excel papers have corresponding PDFs")
        
        validation_results = {
            **match_results,
            'validation_passed': match_results['match_rate'] >= 80,
            'recommendations': recommendations,
            'ready_for_ingestion': match_results['matched_count'] > 0
        }
        
        return validation_results
    
    def generate_matching_report(self, pdf_directory: str, output_path: str = None) -> str:
        """
        Generate detailed matching report
        
        Args:
            pdf_directory: Directory containing PDF files
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        validation_results = self.validate_pdf_directory(pdf_directory)
        
        report = f"""
PDF-Excel Matching Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Directory: {pdf_directory}

OVERVIEW
--------
Total PDF files: {validation_results['total_pdfs']}
Total Excel papers: {validation_results['total_excel_papers']}
Successfully matched: {validation_results['matched_count']}
Match rate: {validation_results['match_rate']:.1f}%

MATCHED FILES ({len(validation_results['pdf_to_excel_map'])})
-------------
"""
        
        for pdf_file, excel_name in validation_results['pdf_to_excel_map'].items():
            journal_info = self.paper_to_journal_map.get(excel_name, {})
            journal = journal_info.get('journal', 'Unknown')
            quartile = journal_info.get('quartile', 'Unknown')
            report += f"✅ {pdf_file} -> {excel_name} ({journal}, {quartile})\n"
        
        if validation_results['unmatched_excel_papers']:
            report += f"\nMISSING PDF FILES ({len(validation_results['unmatched_excel_papers'])})\n"
            report += "-------------------\n"
            for paper in validation_results['unmatched_excel_papers']:
                journal_info = self.paper_to_journal_map.get(paper, {})
                journal = journal_info.get('journal', 'Unknown')
                quartile = journal_info.get('quartile', 'Unknown')
                report += f"❌ {paper} ({journal}, {quartile})\n"
        
        if validation_results['unmatched_pdf_files']:
            report += f"\nUNMATCHED PDF FILES ({len(validation_results['unmatched_pdf_files'])})\n"
            report += "-------------------\n"
            for pdf_file in validation_results['unmatched_pdf_files']:
                report += f"⚠️  {pdf_file} (not in Excel - will be ignored)\n"
        
        report += "\nRECOMMENDATIONS\n"
        report += "---------------\n"
        for rec in validation_results['recommendations']:
            report += f"• {rec}\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Matching report saved to {output_path}")
        
        return report
    
    def match_paper_to_journal(self, paper_identifier: str) -> Optional[Dict]:
        """
        Enhanced paper to journal matching that handles both PDF filenames and Excel names
        
        Args:
            paper_identifier: Could be PDF filename or Excel paper name
            
        Returns:
            Journal information if found
        """
        # First try direct lookup (Excel name)
        if paper_identifier in self.paper_to_journal_map:
            return self.paper_to_journal_map[paper_identifier]
        
        # Try PDF filename lookup
        if paper_identifier in self.pdf_to_excel_map:
            excel_name = self.pdf_to_excel_map[paper_identifier]
            return self.paper_to_journal_map.get(excel_name)
        
        # Try fuzzy matching as fallback
        return self._fuzzy_match_paper(paper_identifier)
    
    def get_quality_tier(self, paper_identifier: str) -> str:
        """Enhanced quality tier lookup"""
        journal_info = self.match_paper_to_journal(paper_identifier)
        if not journal_info:
            return 'unknown'
        
        quartile = journal_info.get('quartile', 'Q4')
        impact_score = journal_info.get('impact_score', 0.0) or 0.0
        
        # Determine tier
        for tier, criteria in self.quality_tiers.items():
            if (quartile in criteria['quartiles'] and 
                impact_score >= criteria['min_impact']):
                return tier
        
        return 'unknown'
    
    def get_paper_quality_score(self, paper_identifier: str) -> float:
        """Enhanced quality score calculation"""
        journal_info = self.match_paper_to_journal(paper_identifier)
        if not journal_info:
            return 0.2  # Default low score for unknown papers
        
        # Calculate quality score based on quartile and impact
        quartile = journal_info.get('quartile', 'Q4')
        impact_score = journal_info.get('normalized_impact_score', 0.0)
        
        # Quartile-based scoring
        quartile_scores = {'Q1': 1.0, 'Q2': 0.8, 'Q3': 0.6, 'Q4': 0.4}
        quartile_score = quartile_scores.get(quartile, 0.2)
        
        # Combined score (70% quartile, 30% impact)
        combined_score = (quartile_score * 0.7) + (impact_score * 0.3)
        
        return combined_score
    
    def get_journal_boost_multiplier(self, paper_identifier: str) -> float:
        """Enhanced boost multiplier calculation"""
        tier = self.get_quality_tier(paper_identifier)
        return self.quality_tiers[tier]['weight']
    
    def _detect_column_structure(self, df: pd.DataFrame) -> Optional[Dict]:
        """Enhanced column structure detection"""
        detected = {}
        columns = [col.lower().strip() for col in df.columns]
        
        logger.info(f"Detecting columns from: {columns}")
        
        # Find columns with enhanced matching
        for col_type, possible_names in self.column_mappings.items():
            for col in columns:
                if any(possible in col for possible in possible_names):
                    detected[col_type] = df.columns[columns.index(col)]
                    break
        
        # Validate required columns
        required_cols = ['paper_name', 'journal']
        missing_cols = [col for col in required_cols if col not in detected]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            # Try fuzzy matching for missing columns
            for col_type in missing_cols:
                possible_names = self.column_mappings[col_type]
                best_match = process.extractOne(col_type, columns)
                if best_match and best_match[1] > 60:  # 60% similarity threshold
                    detected[col_type] = df.columns[columns.index(best_match[0])]
                    logger.info(f"Fuzzy matched '{col_type}' to '{best_match[0]}'")
        
        # Final validation
        if not all(col in detected for col in required_cols):
            logger.error(f"Still missing required columns after fuzzy matching: {detected}")
            return None
        
        logger.info(f"Successfully detected columns: {detected}")
        return detected
    
    def _process_excel_data(self, df: pd.DataFrame, detected_columns: Dict) -> List[Dict]:
        """Enhanced Excel data processing"""
        processed_data = []
        
        logger.info(f"Processing {len(df)} rows from Excel")
        
        for idx, row in df.iterrows():
            try:
                # Extract paper name
                paper_name = str(row[detected_columns['paper_name']]).strip()
                if pd.isna(paper_name) or paper_name == 'nan' or not paper_name:
                    logger.warning(f"Row {idx}: Empty paper name, skipping")
                    continue
                
                # Extract journal
                journal = str(row[detected_columns['journal']]).strip()
                if pd.isna(journal) or journal == 'nan' or not journal:
                    logger.warning(f"Row {idx}: Empty journal for paper '{paper_name}', skipping")
                    continue
                
                # Extract quartile (optional)
                quartile = None
                if 'quartile' in detected_columns:
                    quartile_val = row[detected_columns['quartile']]
                    if not pd.isna(quartile_val):
                        quartile = self._normalize_quartile(str(quartile_val))
                
                # Extract impact score (optional)
                impact_score = None
                if 'impact_score' in detected_columns:
                    impact_val = row[detected_columns['impact_score']]
                    if not pd.isna(impact_val):
                        try:
                            impact_score = float(impact_val)
                        except (ValueError, TypeError):
                            logger.warning(f"Row {idx}: Invalid impact score '{impact_val}' for paper '{paper_name}'")
                
                # Create paper entry
                paper_entry = {
                    'paper_name': paper_name,
                    'journal': journal,
                    'quartile': quartile,
                    'impact_score': impact_score,
                    'original_paper_name': paper_name,
                    'row_index': idx
                }
                
                processed_data.append(paper_entry)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} papers from Excel")
        return processed_data
    
    def _normalize_quartile(self, quartile_str: str) -> str:
        """Enhanced quartile normalization"""
        quartile_str = quartile_str.strip().upper()
        
        # Handle various quartile formats
        if quartile_str in ['Q1', '1', 'FIRST', 'TOP', 'QUARTILE 1']:
            return 'Q1'
        elif quartile_str in ['Q2', '2', 'SECOND', 'QUARTILE 2']:
            return 'Q2'
        elif quartile_str in ['Q3', '3', 'THIRD', 'QUARTILE 3']:
            return 'Q3'
        elif quartile_str in ['Q4', '4', 'FOURTH', 'BOTTOM', 'QUARTILE 4']:
            return 'Q4'
        else:
            # Try to extract number
            match = re.search(r'(\d)', quartile_str)
            if match:
                return f'Q{match.group(1)}'
        
        logger.warning(f"Could not normalize quartile: {quartile_str}")
        return 'Q4'  # Default to Q4 if unclear
    
    def _create_paper_mappings(self, processed_data: List[Dict]):
        """Create enhanced paper-to-journal mappings"""
        journal_papers = {}
        
        for paper_data in processed_data:
            paper_name = paper_data['paper_name']
            journal = paper_data['journal']
            
            # Add to journal metadata
            if journal not in self.journal_metadata:
                self.journal_metadata[journal] = {
                    'name': journal,
                    'papers': [],
                    'quartile': paper_data['quartile'],
                    'impact_score': paper_data['impact_score'],
                    'paper_count': 0
                }
            
            # Add paper to journal
            self.journal_metadata[journal]['papers'].append(paper_data)
            self.journal_metadata[journal]['paper_count'] += 1
            
            # Create paper mapping
            self.paper_to_journal_map[paper_name] = {
                'journal': journal,
                'quartile': paper_data['quartile'],
                'impact_score': paper_data['impact_score'],
                'normalized_impact_score': self._normalize_impact_score(paper_data['impact_score'])
            }
            
            # Create normalized mappings for fuzzy matching
            normalized_name = self._normalize_paper_name(paper_name)
            if normalized_name != paper_name:
                self.paper_to_journal_map[normalized_name] = self.paper_to_journal_map[paper_name]
        
        logger.info(f"Created mappings for {len(self.paper_to_journal_map)} papers")
    
    def _normalize_paper_name(self, paper_name: str) -> str:
        """Enhanced paper name normalization"""
        # Remove common prefixes and suffixes
        normalized = paper_name.lower()
        normalized = re.sub(r'^(paper|article|study|research|doc|document)[-_\s]*', '', normalized)
        normalized = re.sub(r'[-_\s]*\.pdf$', '', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove special characters
        normalized = re.sub(r'\s+', '_', normalized.strip())  # Replace spaces with underscores
        return normalized
    
    def _normalize_impact_score(self, impact_score: float) -> float:
        """Enhanced impact score normalization"""
        if impact_score is None:
            return 0.0
        
        # Assume impact scores are typically 0-50 range
        # Adjust this based on your actual data
        max_expected_impact = 50.0
        normalized = min(impact_score / max_expected_impact, 1.0)
        return normalized
    
    def _calculate_journal_statistics(self):
        """Enhanced journal statistics calculation"""
        if not self.journal_metadata:
            return
        
        # Calculate quartile distribution
        quartile_dist = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Unknown': 0}
        impact_scores = []
        
        for journal_data in self.journal_metadata.values():
            quartile = journal_data['quartile']
            if quartile and quartile in quartile_dist:
                quartile_dist[quartile] += journal_data['paper_count']
            else:
                quartile_dist['Unknown'] += journal_data['paper_count']
            
            if journal_data['impact_score']:
                impact_scores.append(journal_data['impact_score'])
        
        # Calculate impact score statistics
        if impact_scores:
            import numpy as np
            self.journal_statistics = {
                'quartile_distribution': quartile_dist,
                'impact_score_stats': {
                    'min': float(np.min(impact_scores)),
                    'max': float(np.max(impact_scores)),
                    'mean': float(np.mean(impact_scores)),
                    'median': float(np.median(impact_scores)),
                    'std': float(np.std(impact_scores))
                },
                'total_papers': sum(quartile_dist.values()),
                'journals_with_impact_scores': len(impact_scores),
                'total_journals': len(self.journal_metadata)
            }
        else:
            self.journal_statistics = {
                'quartile_distribution': quartile_dist,
                'total_papers': sum(quartile_dist.values()),
                'total_journals': len(self.journal_metadata)
            }
    
    def _create_quality_tiers(self):
        """Enhanced quality tier creation"""
        self.quality_tiers = {
            'top': {'quartiles': ['Q1'], 'min_impact': 10.0, 'weight': 1.8},
            'high': {'quartiles': ['Q1', 'Q2'], 'min_impact': 5.0, 'weight': 1.4},
            'medium': {'quartiles': ['Q2', 'Q3'], 'min_impact': 2.0, 'weight': 1.0},
            'low': {'quartiles': ['Q3', 'Q4'], 'min_impact': 0.0, 'weight': 0.7},
            'unknown': {'quartiles': ['Unknown'], 'min_impact': 0.0, 'weight': 0.3}
        }
    
    def _fuzzy_match_paper(self, paper_filename: str, threshold: int = 70) -> Optional[Dict]:
        """Enhanced fuzzy matching with better normalization"""
        normalized_filename = self._normalize_paper_name(paper_filename)
        
        # Get all paper names for fuzzy matching
        paper_names = list(self.paper_to_journal_map.keys())
        
        # Try fuzzy matching
        best_match = process.extractOne(normalized_filename, paper_names)
        
        if best_match and best_match[1] >= threshold:
            matched_paper = best_match[0]
            logger.info(f"Fuzzy matched '{paper_filename}' to '{matched_paper}' (score: {best_match[1]})")
            return self.paper_to_journal_map[matched_paper]
        
        return None
    
    def get_ingestion_ready_files(self, pdf_directory: str) -> List[str]:
        """
        Get list of PDF files ready for ingestion (only those with Excel matches)
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            List of PDF filenames ready for ingestion
        """
        validation_results = self.validate_pdf_directory(pdf_directory)
        
        if 'error' in validation_results:
            logger.error(f"Cannot get ingestion files: {validation_results['error']}")
            return []
        
        ready_files = list(validation_results['pdf_to_excel_map'].keys())
        logger.info(f"Found {len(ready_files)} files ready for ingestion")
        
        return ready_files
    
    def save_metadata_cache(self, cache_path: str):
        """Enhanced metadata cache saving"""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'journal_metadata': self.journal_metadata,
            'paper_to_journal_map': self.paper_to_journal_map,
            'pdf_to_excel_map': self.pdf_to_excel_map,
            'excel_to_pdf_map': self.excel_to_pdf_map,
            'unmatched_excel_papers': self.unmatched_excel_papers,
            'unmatched_pdf_files': self.unmatched_pdf_files,
            'journal_statistics': self.journal_statistics,
            'quality_tiers': self.quality_tiers
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Enhanced journal metadata cache saved to {cache_path}")
    
    def load_metadata_cache(self, cache_path: str) -> bool:
        """Enhanced metadata cache loading"""
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            self.journal_metadata = cache_data['journal_metadata']
            self.paper_to_journal_map = cache_data['paper_to_journal_map']
            self.pdf_to_excel_map = cache_data.get('pdf_to_excel_map', {})
            self.excel_to_pdf_map = cache_data.get('excel_to_pdf_map', {})
            self.unmatched_excel_papers = cache_data.get('unmatched_excel_papers', [])
            self.unmatched_pdf_files = cache_data.get('unmatched_pdf_files', [])
            self.journal_statistics = cache_data['journal_statistics']
            self.quality_tiers = cache_data['quality_tiers']
            
            logger.info(f"Enhanced journal metadata loaded from cache: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load enhanced metadata cache: {e}")
            return False


# Enhanced utility functions
def validate_pdf_directory_with_excel(pdf_directory: str, excel_path: str) -> Dict:
    """
    Validate PDF directory against Excel file
    
    Args:
        pdf_directory: Directory containing PDF files
        excel_path: Path to Excel file with paper metadata
        
    Returns:
        Validation results
    """
    manager = JournalImpactManager(excel_path)
    return manager.validate_pdf_directory(pdf_directory)

def generate_pdf_excel_report(pdf_directory: str, excel_path: str, output_path: str = None) -> str:
    """
    Generate comprehensive PDF-Excel matching report
    
    Args:
        pdf_directory: Directory containing PDF files
        excel_path: Path to Excel file with paper metadata
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    manager = JournalImpactManager(excel_path)
    return manager.generate_matching_report(pdf_directory, output_path)

def get_ingestion_ready_files(pdf_directory: str, excel_path: str) -> List[str]:
    """
    Get list of PDF files ready for ingestion
    
    Args:
        pdf_directory: Directory containing PDF files
        excel_path: Path to Excel file with paper metadata
        
    Returns:
        List of PDF filenames ready for ingestion
    """
    manager = JournalImpactManager(excel_path)
    return manager.get_ingestion_ready_files(pdf_directory)