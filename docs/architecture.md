# UAM Literature Review RAG System - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Module Architecture](#module-architecture)
4. [Data Flow](#data-flow)
5. [Component Interactions](#component-interactions)
6. [Extension Guidelines](#extension-guidelines)
7. [Maintenance Guide](#maintenance-guide)

---

## System Overview

The UAM Literature Review RAG System is a comprehensive Retrieval-Augmented Generation system designed specifically for synthesizing Urban Air Mobility (UAM) behavioral research literature. The system combines state-of-the-art NLP techniques with domain-specific knowledge to generate high-quality literature reviews.

### Key Features
- **Multimodal Processing**: Extracts text, figures, and tables from PDFs
- **Domain-Specific Optimization**: Tailored for UAM behavioral research
- **Comprehensive Evaluation**: Multi-dimensional evaluation framework
- **Modular Design**: Clean separation of concerns for maintainability
- **Extensible Architecture**: Easy to add new features and capabilities

### Technology Stack
- **Backend**: Python 3.8+
- **ML Framework**: LangChain, Sentence Transformers
- **Vector Store**: FAISS
- **LLM**: OpenRouter API (DeepSeek)
- **UI**: Tkinter (Desktop)
- **PDF Processing**: PyMuPDF, pdfplumber
- **Evaluation**: Custom metrics framework

---

## Project Structure

```
uam_rag_system/
├── config.py                 # Main configuration
├── ui.py                     # Main UI entry point
├── eval.py                   # Evaluation entry point
├── test.py                   # Testing entry point
├── query_template.py         # Query template system
├── 
├── core/                     # Core RAG pipeline modules
│   ├── __init__.py
│   ├── rag_system.py        # Main system orchestrator
│   ├── embeddings.py        # Embedding management
│   ├── retrieval.py         # Document retrieval
│   ├── reranking.py         # Document re-ranking
│   ├── generation.py        # Response generation
│   ├── ingestion.py         # Document ingestion
│   ├── llm_client.py        # LLM API client
│   └── multimodal_processor.py # Multimodal content processing
├── 
├── evaluation/              # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py         # Main evaluation orchestrator
│   ├── metrics.py           # Core evaluation metrics
│   └── emergency_eval.py    # Quick evaluation without ground truth
├── 
├── docs/                    # Documentation
│   ├── architecture.md      # This file
│   ├── user_guide.md        # User guide
│   └── api_reference.md     # API documentation
├── 
├── tests/                   # Test files
│   ├── test_core.py
│   ├── test_evaluation.py
│   └── test_integration.py
├── 
├── data/                    # Data directories
│   ├── pdfs/               # Input PDFs
│   ├── processed/          # Processed documents
│   └── results/            # Output results
├── 
└── requirements.txt         # Dependencies
```

---

## Module Architecture

### 1. Entry Point Modules

#### `config.py`
**Purpose**: Central configuration management
- Contains all system configuration parameters
- Manages API keys and model settings
- Provides chapter-specific configurations
- Handles multimodal processing settings

**Key Components**:
- `Config` class: Main configuration
- `MultimodalConfig` class: Multimodal settings
- `EvaluationConfig` class: Evaluation settings
- System prompts and templates

#### `ui.py`
**Purpose**: Main user interface
- Provides desktop GUI for system interaction
- Handles user inputs and displays results
- Manages system status and corpus operations

**Key Components**:
- `UAMLiteratureReviewUI`: Main UI class
- Modern styling and responsive design
- Integrated corpus management
- Real-time system status monitoring

#### `eval.py`
**Purpose**: Evaluation orchestration
- Coordinates comprehensive system evaluation
- Supports multiple evaluation modes
- Generates evaluation reports

**Key Components**:
- `EvaluationOrchestrator`: Main evaluation class
- Command-line interface for different evaluation modes
- Automated report generation

#### `test.py`
**Purpose**: System testing
- Comprehensive testing framework
- Dependency verification
- Performance benchmarking

**Key Components**:
- `SystemTester`: Main testing class
- Automated dependency installation
- Performance metrics collection

#### `query_template.py`
**Purpose**: Query template system
- Provides structured query generation
- Supports domain-specific query templates
- Enables systematic evaluation

**Key Components**:
- `UAMQueryTemplateLibrary`: Template management
- `QueryTemplate` dataclass: Template structure
- `ResearchDomain` enum: Domain categorization

### 2. Core Pipeline Modules

#### `core/rag_system.py`
**Purpose**: Main system orchestrator
- Coordinates all pipeline components
- Provides high-level API for literature queries
- Manages system lifecycle

**Key Responsibilities**:
- Component initialization and coordination
- Query processing workflow
- Error handling and logging
- System state management

#### `core/embeddings.py`
**Purpose**: Embedding management
- Handles document and query embeddings
- Manages embedding model lifecycle
- Provides embedding dimension information

**Key Responsibilities**:
- Document embedding generation
- Query embedding generation
- Model initialization and configuration
- Embedding normalization

#### `core/retrieval.py`
**Purpose**: Document retrieval
- Manages vector store operations
- Implements similarity search
- Handles query expansion

**Key Responsibilities**:
- Vector store creation and loading
- Similarity search execution
- Query expansion and optimization
- Result deduplication

#### `core/reranking.py`
**Purpose**: Document re-ranking
- Improves retrieval relevance
- Applies domain-specific boosts
- Handles multi-factor scoring

**Key Responsibilities**:
- Cross-encoder re-ranking
- Section-based scoring
- Statistical content boosting
- Chapter-specific optimization

#### `core/generation.py`
**Purpose**: Response generation
- Generates literature review responses
- Manages LLM interactions
- Handles context formatting

**Key Responsibilities**:
- Context preparation from documents
- LLM prompt engineering
- Response post-processing
- Error handling for generation

#### `core/ingestion.py`
**Purpose**: Document ingestion
- Processes PDF documents
- Creates vector embeddings
- Manages corpus metadata

**Key Responsibilities**:
- PDF text extraction
- Document chunking and processing
- Multimodal content integration
- Corpus statistics management

#### `core/multimodal_processor.py`
**Purpose**: Multimodal content processing
- Extracts figures and tables from PDFs
- Performs OCR on images
- Analyzes statistical tables

**Key Responsibilities**:
- Figure extraction and captioning
- Table extraction and analysis
- OCR text processing
- Content type classification

### 3. Evaluation Framework

#### `evaluation/evaluator.py`
**Purpose**: Evaluation orchestration
- Integrates metrics with query templates
- Provides domain-specific evaluation
- Generates comprehensive reports

**Key Responsibilities**:
- Domain benchmark execution
- Progressive complexity testing
- Statistical reporting evaluation
- Result visualization

#### `evaluation/metrics.py`
**Purpose**: Core evaluation metrics
- Implements comprehensive evaluation metrics
- Provides detailed performance analysis
- Supports both automated and manual evaluation

**Key Responsibilities**:
- Retrieval metrics calculation
- Generation quality assessment
- Temporal analysis evaluation
- Synthesis quality measurement

#### `evaluation/emergency_eval.py`
**Purpose**: Quick evaluation
- Provides rapid system health checks
- Works without ground truth data
- Generates immediate feedback

**Key Responsibilities**:
- Basic functionality verification
- Response quality analysis
- Quick recommendation generation
- Error detection and reporting

---

## Data Flow

### 1. Document Ingestion Flow

```
PDF Files → Multimodal Processor → Document Chunks → Embeddings → Vector Store
    ↓              ↓                    ↓              ↓            ↓
Text Extract   Figure/Table      Text Chunking   Embedding     FAISS Index
               Extraction        + Metadata      Generation    + Metadata
```

**Process Details**:
1. **PDF Processing**: Extract text, figures, and tables
2. **Chunking**: Split content into manageable pieces
3. **Embedding**: Generate vector representations
4. **Indexing**: Store in FAISS vector database
5. **Metadata**: Save corpus statistics and metadata

### 2. Query Processing Flow

```
User Query → Query Processing → Retrieval → Re-ranking → Generation → Response
    ↓             ↓                ↓          ↓            ↓          ↓
Preprocessing  UAM Context      Vector     Cross-encoder  Context    Literature
+ Expansion    Addition         Search     Scoring        Formatting Review
```

**Process Details**:
1. **Query Preprocessing**: Add UAM context, expand terms
2. **Retrieval**: Vector similarity search in FAISS
3. **Re-ranking**: Cross-encoder scoring with domain boosts
4. **Context Formatting**: Prepare documents for generation
5. **Response Generation**: LLM-based literature synthesis
6. **Post-processing**: Format and validate response

### 3. Evaluation Flow

```
Test Queries → Query Execution → Metrics Calculation → Analysis → Report
     ↓              ↓                    ↓              ↓         ↓
Template       RAG Pipeline       Retrieval +        Score      Markdown
Generation     Execution          Generation         Analysis   Report
                                  Metrics
```

**Process Details**:
1. **Template Generation**: Create domain-specific test queries
2. **Execution**: Run queries through RAG pipeline
3. **Metrics Calculation**: Evaluate retrieval and generation
4. **Analysis**: Compare results across domains and complexity
5. **Report Generation**: Create comprehensive evaluation report

---

## Component Interactions

### 1. System Initialization

```python
# Main system startup sequence
Config.validate_config()
embedding_manager = EmbeddingManager()
retrieval_engine = RetrievalEngine(embedding_manager)
reranking_engine = ReRankingEngine()
response_generator = ResponseGenerator()
rag_system = UAMRAGSystem()
```

### 2. Document Ingestion

```python
# Ingestion workflow
multimodal_processor = MultimodalProcessor()
document_ingester = DocumentIngester(embedding_manager)
stats = document_ingester.ingest_multimodal(pdf_directory, multimodal_processor)
```

### 3. Query Processing

```python
# Query processing workflow
docs = retrieval_engine.retrieve(query)
top_docs = reranking_engine.rerank(query, docs, chapter_topic)
response = response_generator.generate_response(query, top_docs, chapter_topic)
```

### 4. Evaluation Execution

```python
# Evaluation workflow
evaluator = RAGEvaluationSystem(rag_system)
result = evaluator.evaluate_query(query, expected_papers, expected_findings)
```

---

## Extension Guidelines

### Adding New Evaluation Metrics

1. **Create Metric Class**:
   ```python
   @dataclass
   class CustomMetrics:
       custom_score: float = 0.0
       custom_coverage: float = 0.0
   ```

2. **Implement Evaluation Logic**:
   ```python
   def _evaluate_custom(self, query: str, response: str) -> CustomMetrics:
       # Implementation here
       pass
   ```

3. **Integrate with Main Evaluator**:
   ```python
   # Add to EvaluationResult
   custom_metrics: Optional[CustomMetrics] = None
   ```

### Adding New Query Templates

1. **Define Template**:
   ```python
   QueryTemplate(
       id="new_template_001",
       domain=ResearchDomain.NEW_DOMAIN,
       template="Template with {variables}",
       variables=["variable1", "variable2"],
       expected_sections=["results", "discussion"],
       statistical_focus=True,
       example_query="Example query",
       evaluation_criteria={"criteria": "values"}
   )
   ```

2. **Add to Template Library**:
   ```python
   # Add to _initialize_templates() method
   ```

### Adding New Retrieval Features

1. **Extend Retrieval Engine**:
   ```python
   def new_retrieval_method(self, query: str) -> List[Document]:
       # Implementation
       pass
   ```

2. **Update Configuration**:
   ```python
   # Add to Config class
   ENABLE_NEW_FEATURE = True
   ```

3. **Integrate with Main System**:
   ```python
   # Update enhanced_retrieval() method
   ```

### Adding New Multimodal Features

1. **Extend Multimodal Processor**:
   ```python
   def extract_new_content_type(self, pdf_path: str) -> List[NewContentType]:
       # Implementation
       pass
   ```

2. **Update Data Structures**:
   ```python
   @dataclass
   class NewContentData:
       # Define structure
       pass
   ```

3. **Integrate with Chunking**:
   ```python
   def create_new_content_chunks(self, content: List[NewContentData]) -> List[Document]:
       # Implementation
       pass
   ```

---

## Maintenance Guide

### Regular Maintenance Tasks

#### 1. Dependency Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check for security vulnerabilities
pip audit

# Test after updates
python test.py --mode comprehensive
```

#### 2. Performance Monitoring
```python
# Monitor system performance
python eval.py --mode performance

# Check corpus statistics
python test.py --mode corpus

# Monitor memory usage
python test.py --mode performance
```

#### 3. Configuration Validation
```python
# Validate configuration
python -c "from config import Config; Config.validate_config()"

# Check API connectivity
python test.py --mode quick
```

### Debugging Common Issues

#### 1. Import Errors
- Check dependency installation
- Verify Python environment
- Run dependency test: `python test.py --mode deps`

#### 2. API Failures
- Verify API key configuration
- Check network connectivity
- Test with simple query

#### 3. Performance Issues
- Monitor memory usage
- Check corpus size
- Optimize chunk size and retrieval parameters

#### 4. Evaluation Problems
- Verify ground truth data format
- Check template library consistency
- Test with emergency evaluation

### Code Quality Maintenance

#### 1. Code Style
```bash
# Format code
black *.py core/ evaluation/

# Check style
flake8 *.py core/ evaluation/

# Type checking
mypy *.py
```

#### 2. Testing
```bash
# Run comprehensive tests
python test.py --mode comprehensive

# Run specific test modules
python -m pytest tests/

# Check test coverage
coverage run -m pytest tests/
coverage report
```

#### 3. Documentation Updates
- Update docstrings for new features
- Maintain architecture documentation
- Update user guide with new capabilities

### Backup and Recovery

#### 1. Backup Procedures
```python
# Backup corpus
python ui.py  # Use backup function in UI

# Backup configuration
cp config.py config.py.backup

# Backup evaluation results
cp -r evaluation_results/ evaluation_results.backup/
```

#### 2. Recovery Procedures
```python
# Restore from backup
# 1. Restore configuration
# 2. Restore corpus data
# 3. Rebuild vector store if needed
# 4. Validate system functionality
```

### Performance Optimization

#### 1. Memory Optimization
- Monitor embedding model memory usage
- Optimize chunk sizes
- Implement lazy loading for large corpora

#### 2. Speed Optimization
- Cache frequently used embeddings
- Optimize vector search parameters
- Use batch processing for large ingestions

#### 3. Quality Optimization
- Tune retrieval parameters
- Adjust re-ranking weights
- Optimize generation prompts

---

## Security Considerations

### 1. API Key Management
- Store API keys in environment variables
- Never commit API keys to version control
- Rotate keys regularly
- Use key validation in configuration

### 2. Input Validation
- Validate PDF file types and sizes
- Sanitize user inputs
- Implement query length limits
- Check for malicious content

### 3. Data Privacy
- Implement secure PDF processing
- Avoid storing sensitive information
- Use secure temporary file handling
- Implement data retention policies

---

## Troubleshooting

### Common Error Messages

1. **"Missing required dependencies"**
   - Solution: Run `python test.py --install-deps`

2. **"OPENROUTER_API_KEY not found"**
   - Solution: Set environment variable or update config

3. **"Please ingest papers first"**
   - Solution: Run paper ingestion before querying

4. **"Multimodal dependencies not available"**
   - Solution: Install PyMuPDF, pdfplumber, Pillow

### Performance Issues

1. **Slow query processing**
   - Check corpus size and retrieval parameters
   - Monitor memory usage
   - Consider reducing chunk size

2. **High memory usage**
   - Reduce batch size during ingestion
   - Use lighter embedding models
   - Optimize vector store configuration

3. **Poor response quality**
   - Check corpus quality and size
   - Tune re-ranking parameters
   - Adjust generation prompts

### Getting Help

1. **Check system status**: `python test.py --mode quick`
2. **Run diagnostics**: `python eval.py --mode emergency`
3. **Review logs**: Check console output for detailed error messages
4. **Validate configuration**: `python -c "from config import Config; Config.validate_config()"`

---

This architecture documentation provides a comprehensive guide to understanding, extending, and maintaining the UAM Literature Review RAG System. For specific implementation details, refer to the source code and inline documentation.