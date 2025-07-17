# core/multimodal_processor.py
"""
Multimodal Processing Module
Handles extraction of figures, tables, and other multimodal content.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import Config, MultimodalConfig

logger = logging.getLogger(__name__)


@dataclass
class FigureData:
    """Structure for figure information"""
    figure_id: str
    page_number: int
    caption: str
    bbox: Tuple[float, float, float, float]
    image_data: Optional[bytes] = None
    image_base64: Optional[str] = None
    ocr_text: Optional[str] = None
    figure_type: str = "figure"


@dataclass
class TableData:
    """Structure for table information"""
    table_id: str
    page_number: int
    caption: str
    bbox: Tuple[float, float, float, float]
    data: List[List[str]]
    df: Optional[object] = None
    text_representation: str = ""
    column_headers: List[str] = None
    table_type: str = "table"


class MultimodalProcessor:
    """Processes multimodal content from PDFs"""
    
    def __init__(self):
        """Initialize multimodal processor"""
        self._check_dependencies()
        self.setup_extraction_patterns()
        logger.info("Multimodal processor initialized")
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import fitz
            import pdfplumber
            from PIL import Image
            self.dependencies_available = True
        except ImportError as e:
            logger.error(f"Multimodal dependencies not available: {e}")
            self.dependencies_available = False
            raise
    
    def setup_extraction_patterns(self):
        """Setup patterns for identifying figures and tables"""
        self.figure_patterns = [
            r'(?i)fig(?:ure)?\s*\.?\s*(\d+)',
            r'(?i)figure\s+(\d+)',
            r'(?i)chart\s+(\d+)',
            r'(?i)diagram\s+(\d+)'
        ]
        
        self.table_patterns = [
            r'(?i)table\s*\.?\s*(\d+)',
            r'(?i)tab\s*\.?\s*(\d+)',
        ]
    
    def extract_multimodal_content(self, pdf_path: str) -> Dict:
        """Extract multimodal content from PDF"""
        if not self.dependencies_available:
            raise RuntimeError("Multimodal dependencies not available")
        
        logger.info(f"Extracting multimodal content from {pdf_path}")
        
        # Import here to avoid dependency issues
        import fitz
        import pdfplumber
        
        extraction_results = {
            'text_content': "",
            'figures': [],
            'tables': [],
            'captions': [],
            'stats': {
                'total_pages': 0,
                'figures_found': 0,
                'tables_found': 0,
                'captions_found': 0
            }
        }
        
        try:
            # Process with PyMuPDF for figures
            pdf_document = fitz.open(pdf_path)
            all_text = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                all_text.append(page_text)
                
                # Extract figures
                page_figures = self._extract_figures_from_page(page, page_num)
                extraction_results['figures'].extend(page_figures)
            
            # Process with pdfplumber for tables
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(len(pdf.pages)):
                    page_tables = self._extract_tables_from_page(pdf, page_num)
                    extraction_results['tables'].extend(page_tables)
            
            # Combine text
            extraction_results['text_content'] = '\n'.join(all_text)
            extraction_results['stats']['total_pages'] = len(pdf_document)
            extraction_results['stats']['figures_found'] = len(extraction_results['figures'])
            extraction_results['stats']['tables_found'] = len(extraction_results['tables'])
            
            pdf_document.close()
            
            logger.info(f"Extraction complete: {extraction_results['stats']}")
            return extraction_results
            
        except Exception as e:
            logger.error(f"Error in multimodal extraction: {e}")
            # Return basic text extraction as fallback
            return self._fallback_text_extraction(pdf_path)
    
    def _extract_figures_from_page(self, page, page_num: int) -> List[FigureData]:
        """Extract figures from a PDF page"""
        figures = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract basic image info
                    figure = FigureData(
                        figure_id=f"fig_p{page_num + 1}_{img_index + 1}",
                        page_number=page_num + 1,
                        caption="",
                        bbox=(0, 0, 100, 100),  # Placeholder
                        figure_type="figure"
                    )
                    
                    figures.append(figure)
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error extracting figures from page {page_num}: {e}")
        
        return figures
    
    def _extract_tables_from_page(self, pdf, page_num: int) -> List[TableData]:
        """Extract tables from a PDF page"""
        tables = []
        
        try:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                page_tables = page.extract_tables()
                
                for table_index, table_data in enumerate(page_tables or []):
                    if table_data and len(table_data) > 1:
                        # Clean table data
                        cleaned_data = []
                        for row in table_data:
                            if row and any(cell and str(cell).strip() for cell in row):
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                cleaned_data.append(cleaned_row)
                        
                        if len(cleaned_data) >= 2:
                            table = TableData(
                                table_id=f"tab_p{page_num + 1}_{table_index + 1}",
                                page_number=page_num + 1,
                                caption="",
                                bbox=(0, 0, 100, 100),  # Placeholder
                                data=cleaned_data,
                                text_representation=self._table_to_text(cleaned_data),
                                column_headers=cleaned_data[0] if cleaned_data else [],
                                table_type="table"
                            )
                            
                            tables.append(table)
        
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation"""
        if not table_data:
            return ""
        
        try:
            text_parts = []
            
            # Headers
            if table_data:
                headers = table_data[0]
                text_parts.append("Headers: " + " | ".join(headers))
            
            # Data rows (limit to prevent enormous text)
            max_rows = min(5, len(table_data) - 1)
            for i in range(1, max_rows + 1):
                if i < len(table_data):
                    row = table_data[i]
                    row_text = " | ".join(str(cell) for cell in row)
                    text_parts.append(f"Row {i}: {row_text}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.warning(f"Error converting table to text: {e}")
            return str(table_data)
    
    def _fallback_text_extraction(self, pdf_path: str) -> Dict:
        """Fallback text extraction when multimodal fails"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            return {
                'text_content': "\n".join([page.page_content for page in pages]),
                'figures': [],
                'tables': [],
                'captions': [],
                'stats': {
                    'total_pages': len(pages),
                    'figures_found': 0,
                    'tables_found': 0,
                    'captions_found': 0
                }
            }
        except Exception as e:
            logger.error(f"Fallback text extraction failed: {e}")
            return {
                'text_content': "",
                'figures': [],
                'tables': [],
                'captions': [],
                'stats': {
                    'total_pages': 0,
                    'figures_found': 0,
                    'tables_found': 0,
                    'captions_found': 0
                }
            }
    
    def create_multimodal_chunks(self, extraction_results: Dict, metadata: Dict) -> List:
        """Create document chunks from multimodal content"""
        from langchain.schema import Document
        
        chunks = []
        
        # Create text chunks
        if extraction_results['text_content']:
            text_chunks = self._create_text_chunks(extraction_results['text_content'], metadata)
            chunks.extend(text_chunks)
        
        # Create figure chunks
        for figure in extraction_results['figures']:
            doc = Document(
                page_content=f"Figure {figure.figure_id}: {figure.caption or 'No caption'}",
                metadata={
                    **metadata,
                    'chunk_type': 'figure',
                    'chunk_id': figure.figure_id,
                    'content_type': 'figure',
                    'page_number': figure.page_number
                }
            )
            chunks.append(doc)
        
        # Create table chunks
        for table in extraction_results['tables']:
            doc = Document(
                page_content=f"Table {table.table_id}: {table.text_representation}",
                metadata={
                    **metadata,
                    'chunk_type': 'table',
                    'chunk_id': table.table_id,
                    'content_type': 'table',
                    'page_number': table.page_number
                }
            )
            chunks.append(doc)
        
        return chunks
    
    def _create_text_chunks(self, text: str, metadata: Dict) -> List:
        """Create text chunks from extracted text"""
        from langchain.schema import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.OVERLAP_SIZE,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_chunks = splitter.split_text(text)
        documents = []
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) > 50:
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_type': 'text',
                        'chunk_id': f"{metadata['source']}_text_{i}",
                        'content_type': 'text'
                    }
                )
                documents.append(doc)
        
        return documents
    
    def diagnose_extraction(self, pdf_path: str) -> Dict:
        """Diagnose extraction for a single PDF"""
        try:
            extraction_results = self.extract_multimodal_content(pdf_path)
            
            return {
                'filename': os.path.basename(pdf_path),
                'success': True,
                'stats': extraction_results['stats'],
                'text_length': len(extraction_results['text_content']),
                'figures': [
                    {
                        'id': fig.figure_id,
                        'page': fig.page_number,
                        'type': fig.figure_type,
                        'has_caption': bool(fig.caption)
                    }
                    for fig in extraction_results['figures']
                ],
                'tables': [
                    {
                        'id': tab.table_id,
                        'page': tab.page_number,
                        'type': tab.table_type,
                        'rows': len(tab.data) if tab.data else 0,
                        'columns': len(tab.column_headers) if tab.column_headers else 0
                    }
                    for tab in extraction_results['tables']
                ]
            }
            
        except Exception as e:
            return {
                'filename': os.path.basename(pdf_path),
                'success': False,
                'error': str(e)
            }