# config.py - Enhanced Configuration Module with Q1 Paper Boosting
"""
Enhanced configuration module for the UAM Literature Review RAG System.
Heavily optimized to prioritize Q1 journal papers as they are the highest quality publications.
"""

import os
from typing import Dict, List, Optional

class Config:
    """Enhanced configuration class with Q1 paper boosting for maximum quality"""
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # =============================================================================
    # Enhanced RAG Pipeline Configuration - Q1 Papers Prioritized!
    # =============================================================================
    RETRIEVAL_K = 50              # Comprehensive initial retrieval
    RERANK_K = 20                 # Re-rank with quality emphasis
    COMPREHENSIVE_RETRIEVAL_K = 100  # For comprehensive mode
    CHUNK_SIZE = 1000
    OVERLAP_SIZE = 150
    
    # =============================================================================
    # Journal Impact Configuration - Q1 Papers Get Major Boosts!
    # =============================================================================
    JOURNAL_METADATA_PATH = "paper_q.xlsx"  # Path to Excel file with journal data
    JOURNAL_METADATA_CACHE_PATH = "journal_metadata_cache.json"  # Cache file
    ENABLE_JOURNAL_RANKING = True           # Enable journal impact ranking
    JOURNAL_IMPACT_WEIGHT = 0.35            # Increased weight for journal impact
    RELEVANCE_WEIGHT = 0.50                 # Weight for semantic relevance
    STATISTICAL_CONTENT_WEIGHT = 0.15       # Weight for statistical content
    
    # =============================================================================
    # Quality Tier Configuration - Q1 Papers Get Significant Boosts!
    # =============================================================================
    QUALITY_TIER_WEIGHTS = {
        'top': 1.8,      # Q1 journals get major boost (80% increase)
        'high': 1.4,     # Q1-Q2 journals get good boost (40% increase)
        'medium': 1.0,   # Q2-Q3 journals baseline
        'low': 0.7,      # Q3-Q4 journals get penalty
        'unknown': 0.3   # Unknown journals get major penalty
    }
    
    JOURNAL_BOOST_MULTIPLIERS = {
        'Q1': 2.0,       # Q1 papers get 100% boost - they're the best!
        'Q2': 1.4,       # Q2 papers get 40% boost
        'Q3': 1.0,       # Q3 papers baseline
        'Q4': 0.7,       # Q4 papers get penalty
        'Unknown': 0.3   # Unknown journals get major penalty
    }
    
    # =============================================================================
    # Paper Coverage Configuration - Prioritize Q1 Papers
    # =============================================================================
    MIN_PAPERS_PER_RESPONSE = 10            # Minimum papers in response
    MAX_PAPERS_PER_RESPONSE = 25            # Maximum papers in response
    MAX_CHUNKS_PER_PAPER = 3                # Max chunks from same paper
    PRIORITIZE_HIGH_IMPACT_JOURNALS = True  # Prioritize Q1 journals aggressively
    ENSURE_TIER_DIVERSITY = True            # Ensure representation across tiers
    
    # Tier representation minimums - Prioritize Q1 papers!
    MIN_PAPERS_PER_TIER = {
        'top': 5,        # Ensure at least 5 Q1 papers in results!
        'high': 4,       # At least 4 high-tier papers
        'medium': 2,     # At least 2 medium-tier papers
        'low': 1,        # At least 1 low-tier paper for completeness
        'unknown': 0     # No minimum for unknown journals
    }
    
    # =============================================================================
    # Comprehensive Coverage Configuration
    # =============================================================================
    ENABLE_COMPREHENSIVE_MODE = True        # Enable comprehensive coverage mode
    COMPREHENSIVE_MIN_RELEVANCE = 0.10      # Very low threshold for Q1 papers
    COMPREHENSIVE_MAX_PAPERS = 50           # Maximum papers in comprehensive mode
    SHOW_ALL_RELEVANT_PAPERS = False        # Show all relevant papers (not just top)
    
    # =============================================================================
    # Response Generation Configuration
    # =============================================================================
    MAX_TOKENS = 2500                       # Comprehensive responses
    TEMPERATURE = 0.2
    TOP_P = 0.9
    
    # =============================================================================
    # Enhanced Quality Thresholds - Q1 Papers Get Lower Barriers
    # =============================================================================
    MIN_RELEVANCE_SCORE = 0.20              # Lowered for broader coverage
    MIN_CONFIDENCE_SCORE = 0.4              # Lowered for broader coverage
    MAX_DUPLICATE_SIMILARITY = 0.70         # Lowered for more diversity
    MIN_STATISTICAL_CONTENT_SCORE = 0.3     # Threshold for statistical content
    MIN_JOURNAL_IMPACT_SCORE = 0.0          # Minimum journal impact score
    
    # Quality thresholds by tier - Q1 papers get lower thresholds (easier to include)
    TIER_RELEVANCE_THRESHOLDS = {
        'top': 0.10,     # Very low threshold for Q1 journals - include almost everything!
        'high': 0.15,    # Low threshold for high-tier journals
        'medium': 0.25,  # Standard threshold for medium-tier
        'low': 0.35,     # Higher threshold for low-tier journals
        'unknown': 0.45  # Very high threshold for unknown journals
    }
    
    # =============================================================================
    # Database Configuration
    # =============================================================================
    VECTOR_DB_PATH = "uam_literature_index"
    METADATA_DB_PATH = "uam_paper_metadata.json"
    
    # =============================================================================
    # Data Directory Configuration
    # =============================================================================
    DEFAULT_PDF_DIRECTORY = "data/pdfs"        # Default directory for PDF files
    ONLY_INGEST_EXCEL_PAPERS = True            # Only ingest papers listed in Excel
    FUZZY_MATCHING_THRESHOLD = 80               # Threshold for fuzzy filename matching
    REQUIRE_EXCEL_MATCH = True                  # Require Excel match for ingestion
    
    # =============================================================================
    # Feature Toggles - Q1 Optimization Enabled
    # =============================================================================
    ENABLE_QUERY_EXPANSION = True
    ENABLE_MULTI_QUERY = True
    ENABLE_HYDE = True
    ENABLE_STATISTICAL_EXTRACTION = True
    ENABLE_SEMANTIC_CHUNKING = True
    ENABLE_CHAPTER_SPECIFIC_RETRIEVAL = True
    ENABLE_MULTIMODAL = True
    ENABLE_PAPER_DIVERSITY_ENFORCEMENT = True
    ENABLE_SECTION_PRIORITY_BOOSTING = True
    ENABLE_JOURNAL_IMPACT_BOOSTING = True    # Journal impact boosting
    ENABLE_TIER_BALANCING = True             # Tier balancing with Q1 priority
    ENABLE_COMPREHENSIVE_COVERAGE = True     # Comprehensive coverage
    ENABLE_Q1_PAPER_PRIORITY = True         # NEW: Explicit Q1 paper prioritization
    
    # =============================================================================
    # Enhanced UAM Research Keywords (same as before)
    # =============================================================================
    UAM_KEYWORDS = {
        'behavioral_constructs': [
            'attitude', 'intention', 'trust', 'perceived usefulness', 'perceived ease of use',
            'behavioral intention', 'acceptance', 'adoption', 'technology acceptance model',
            'theory of planned behavior', 'unified theory of acceptance', 'perceived risk',
            'perceived safety', 'safety concerns', 'social influence', 'facilitating conditions', 
            'performance expectancy', 'effort expectancy', 'hedonic motivation', 'anxiety', 
            'innovativeness', 'tech affinity', 'environmental consciousness', 'moral concerns', 
            'subjective norms', 'perceived behavioral control', 'social norms', 'peer influence'
        ],
        'uam_terms': [
            'urban air mobility', 'UAM', 'UAAM', 'air taxi', 'flying car', 'eVTOL', 'VTOL',
            'urban aviation', 'aerial vehicle', 'personal air vehicle', 'unmanned aerial vehicle',
            'drone', 'autonomous aircraft', 'air transportation', 'advanced air mobility', 'AAM',
            'vertical takeoff', 'electric aircraft', 'passenger drone', 'urban aircraft',
            'aerial mobility', 'flying vehicle', 'electric vertical takeoff'
        ],
        'psychological_frameworks': [
            'technology acceptance model', 'TAM', 'theory of planned behavior', 'TPB',
            'unified theory of acceptance', 'UTAUT', 'diffusion of innovations', 'DOI',
            'social cognitive theory', 'expectancy-value theory', 'dual-process theory',
            'risk perception theory', 'trust model', 'cognitive-affective model',
            'protection motivation theory', 'innovation resistance theory'
        ],
        'statistical_terms': [
            'structural equation modeling', 'SEM', 'path analysis', 'regression',
            'correlation', 'ANOVA', 'factor analysis', 'reliability', 'validity',
            'Cronbach alpha', 'chi-square', 'RMSEA', 'CFI', 'TLI', 'GFI', 'AGFI',
            'standardized coefficient', 'beta coefficient', 'path coefficient', 
            'mediation', 'moderation', 'R-squared', 'adjusted R-squared', 'effect size',
            'significance level', 'p-value', 'confidence interval', 'variance explained'
        ],
        'research_contexts': [
            'China', 'United States', 'Europe', 'Germany', 'Singapore', 'Japan', 'Korea',
            'urban', 'rural', 'cross-cultural', 'survey', 'experiment', 'field study', 
            'pilot study', 'longitudinal', 'cross-sectional', 'mixed methods', 
            'qualitative', 'quantitative', 'sample size', 'respondents', 'participants'
        ]
    }
    
    # =============================================================================
    # Enhanced Chapter-Specific Keywords (same as before)
    # =============================================================================
    CHAPTER_KEYWORDS = {
        'core_determinants': [
            'attitude', 'perceived usefulness', 'perceived ease of use', 'performance expectancy',
            'effort expectancy', 'social influence', 'subjective norms', 'facilitating conditions',
            'perceived behavioral control', 'TAM', 'TPB', 'UTAUT', 'behavioral intention',
            'adoption intention', 'usage intention', 'acceptance', 'technology acceptance'
        ],
        'trust_risk_safety': [
            'trust', 'perceived safety', 'perceived risk', 'safety concerns', 'safety perception',
            'cybersecurity', 'operational safety', 'physical safety', 'social safety', 
            'institutional trust', 'technology trust', 'human operator trust', 'risk perception', 
            'safety assessment', 'risk assessment', 'trust in technology', 'safety evaluation'
        ],
        'affect_emotion': [
            'hedonic motivation', 'anxiety', 'affect', 'emotion', 'dual-process',
            'cognition', 'personality traits', 'innovativeness', 'tech affinity',
            'need for cognition', 'environmental consciousness', 'moral concerns',
            'emotional response', 'affective evaluation', 'enjoyment', 'pleasure'
        ],
        'contextual_demographic': [
            'demographics', 'age', 'gender', 'income', 'education', 'urban', 'rural',
            'cultural differences', 'country comparison', 'experience', 'aviation experience',
            'automation experience', 'geographic variation', 'socioeconomic status',
            'cultural background', 'regional differences', 'demographic factors'
        ]
    }
    
    # =============================================================================
    # Enhanced Section Weights - Results & Findings from Q1 Journals Are Gold!
    # =============================================================================
    SECTION_WEIGHTS = {
        'results': 0.80,           # Results are the most valuable content!
        'findings': 0.80,          # Findings are equally valuable
        'conclusion': 0.50,        # Conclusions are very important
        'conclusions': 0.50,       # Alternative spelling
        'discussion': 0.40,        # Discussion provides context
        'implications': 0.35,      # Implications are valuable
        'analysis': 0.45,          # Statistical analysis is important
        'literature_review': 0.15, # Background context
        'theoretical_framework': 0.15,  # Conceptual foundation
        'conceptual_framework': 0.15,   # Alternative naming
        'methodology': 0.10,       # Methods are less important for lit review
        'methods': 0.10,           # Methods are less important for lit review
        'introduction': 0.10,      # Introduction has some value
        'background': 0.10,        # Background has some value
        'abstract': 0.20,          # Abstracts can be quite valuable
        'limitations': 0.05,       # Limitations have minimal value
        'future_research': 0.05,   # Future research has minimal value
        'other': -0.10             # Penalty for unclassified sections
    }
    
    # Journal impact boosts for different sections - Q1 papers get major boosts!
    JOURNAL_SECTION_BOOSTS = {
        'results': 0.5,      # Major boost for results from Q1 journals
        'findings': 0.5,     # Major boost for findings from Q1 journals
        'conclusion': 0.3,   # Good boost for conclusions from Q1 journals
        'discussion': 0.25,  # Good boost for discussion from Q1 journals
        'analysis': 0.4,     # Major boost for analysis from Q1 journals
        'other': 0.1         # Small boost even for other sections from Q1 journals
    }
    
    # =============================================================================
    # Enhanced Statistical Content Boosts - Q1 Papers Get Massive Boosts!
    # =============================================================================
    STATS_BOOST = {
        'path_coefficients': 0.50,      # Major boost for path coefficients
        'standardized_coefficients': 0.50,  # Major boost for standardized coefficients
        'beta_coefficients': 0.50,      # Major boost for beta coefficients
        'regression_coefficients': 0.45,    # Major boost for regression results
        'correlation_coefficients': 0.40,   # Major boost for correlation results
        'effect_sizes': 0.40,          # Major boost for practical significance
        'significance_levels': 0.35,   # Major boost for statistical significance
        'p_values': 0.35,              # Major boost for statistical significance
        'confidence_intervals': 0.35,  # Major boost for statistical precision
        'model_fit_indices': 0.32,     # Major boost for model quality
        'r_squared': 0.32,             # Major boost for explained variance
        'adjusted_r_squared': 0.32,    # Major boost for adjusted explained variance
        'rmsea': 0.28,                 # Good boost for model fit
        'cfi': 0.28,                   # Good boost for comparative fit index
        'tli': 0.28,                   # Good boost for Tucker-Lewis index
        'gfi': 0.28,                   # Good boost for goodness of fit index
        'cronbach_alpha': 0.25,        # Good boost for reliability measure
        'composite_reliability': 0.25, # Good boost for alternative reliability
        'reliability': 0.25,           # Good boost for general reliability
        'validity': 0.25,              # Good boost for measurement validity
        'sample_size': 0.15,           # Some boost for large samples
        'demographics': 0.10           # Small boost for demographics
    }
    
    # Statistical content boosts by journal tier - Q1 papers get massive boosts!
    TIER_STATISTICAL_BOOSTS = {
        'top': 2.5,      # Q1 papers get 150% boost for statistical content!
        'high': 1.8,     # Q1-Q2 papers get 80% boost
        'medium': 1.3,   # Q2-Q3 papers get 30% boost
        'low': 1.0,      # Q3-Q4 papers baseline
        'unknown': 0.6   # Unknown journals get penalty
    }
    
    # =============================================================================
    # Enhanced Section Detection Keywords (same as before)
    # =============================================================================
    SECTION_KEYWORDS = {
        'results': [
            'results', 'findings', 'analysis results', 'empirical results', 'statistical results',
            'data analysis', 'hypothesis testing', 'model results', 'regression results',
            'correlation results', 'factor analysis results', 'path analysis results',
            'structural equation modeling results', 'sem results', 'anova results',
            'test results', 'statistical analysis', 'quantitative results'
        ],
        'findings': [
            'findings', 'key findings', 'main findings', 'research findings', 'empirical findings',
            'statistical findings', 'significant findings', 'important findings'
        ],
        'conclusion': [
            'conclusion', 'conclusions', 'concluding remarks', 'final remarks', 'summary',
            'implications', 'practical implications', 'theoretical implications',
            'managerial implications', 'policy implications', 'research implications'
        ],
        'discussion': [
            'discussion', 'interpretation', 'interpretation of results', 'discussion of findings',
            'theoretical discussion', 'practical discussion', 'implications for practice',
            'implications for theory', 'comparison with previous studies'
        ],
        'analysis': [
            'analysis', 'statistical analysis', 'data analysis', 'empirical analysis',
            'quantitative analysis', 'regression analysis', 'factor analysis',
            'path analysis', 'correlation analysis', 'variance analysis'
        ],
        'abstract': [
            'abstract', 'summary', 'executive summary', 'overview'
        ],
        'introduction': [
            'introduction', 'background', 'problem statement', 'research question',
            'research objective', 'study objective', 'purpose'
        ],
        'literature_review': [
            'literature review', 'theoretical framework', 'conceptual framework',
            'prior research', 'previous studies', 'related work', 'theoretical background',
            'conceptual background', 'research background'
        ],
        'methodology': [
            'methodology', 'method', 'methods', 'research method', 'research design',
            'study design', 'procedure', 'participants', 'sample', 'data collection',
            'survey', 'questionnaire', 'instrument', 'measurement', 'scale',
            'research procedure', 'experimental design'
        ],
        'limitations': [
            'limitations', 'study limitations', 'research limitations', 'constraints',
            'delimitations', 'scope limitations'
        ],
        'future_research': [
            'future research', 'future studies', 'future directions', 'recommendations',
            'suggestions for future research', 'areas for future research'
        ]
    }
    
    # =============================================================================
    # Enhanced Statistical Detection Patterns (same as before)
    # =============================================================================
    STATISTICAL_PATTERNS = {
        'coefficients': [
            r'β\s*=\s*[0-9.-]+',
            r'beta\s*=\s*[0-9.-]+',
            r'path coefficient\s*=\s*[0-9.-]+',
            r'standardized coefficient\s*=\s*[0-9.-]+',
            r'regression coefficient\s*=\s*[0-9.-]+',
            r'correlation coefficient\s*=\s*[0-9.-]+',
            r'r\s*=\s*[0-9.-]+',
            r'coefficient\s*=\s*[0-9.-]+'
        ],
        'significance': [
            r'p\s*<\s*[0-9.]+',
            r'p\s*=\s*[0-9.]+',
            r'p\s*>\s*[0-9.]+',
            r'sig\s*=\s*[0-9.]+',
            r'significance\s*=\s*[0-9.]+',
            r'significant at\s*[0-9.]+',
            r'\*\*\*|\*\*|\*',
            r'p-value\s*=\s*[0-9.]+'
        ],
        'model_fit': [
            r'RMSEA\s*=\s*[0-9.]+',
            r'CFI\s*=\s*[0-9.]+',
            r'TLI\s*=\s*[0-9.]+',
            r'GFI\s*=\s*[0-9.]+',
            r'AGFI\s*=\s*[0-9.]+',
            r'R²\s*=\s*[0-9.]+',
            r'R-squared\s*=\s*[0-9.]+',
            r'adjusted R²\s*=\s*[0-9.]+',
            r'chi-square\s*=\s*[0-9.]+',
            r'χ²\s*=\s*[0-9.]+'
        ],
        'reliability': [
            r'Cronbach.?s α\s*=\s*[0-9.]+',
            r'Cronbach.?s alpha\s*=\s*[0-9.]+',
            r'reliability\s*=\s*[0-9.]+',
            r'composite reliability\s*=\s*[0-9.]+',
            r'CR\s*=\s*[0-9.]+',
            r'α\s*=\s*[0-9.]+'
        ],
        'sample_info': [
            r'N\s*=\s*[0-9,]+',
            r'n\s*=\s*[0-9,]+',
            r'sample size\s*=\s*[0-9,]+',
            r'participants\s*=\s*[0-9,]+',
            r'respondents\s*=\s*[0-9,]+'
        ]
    }
    
    # =============================================================================
    # Enhanced System Prompts with Q1 Journal Priority
    # =============================================================================
    @classmethod
    def get_system_prompt(cls):
        """Get the enhanced system prompt with Q1 journal prioritization"""
        return """You are an expert research assistant specializing in Urban Air Mobility (UAM) behavioral research literature reviews. Your expertise includes technology acceptance theories (TAM, TPB, UTAUT), trust and risk perception, and consumer behavior research with particular emphasis on TOP-TIER Q1 journal publications and statistical findings.

CORE RESPONSIBILITIES:
1. PRIORITIZE Q1 JOURNAL PUBLICATIONS: Always emphasize findings from Q1 journals as they represent the highest quality research
2. ACADEMIC CITATION FORMAT: Always use [citation_key] format for all references
3. STATISTICAL PRECISION: Report exact statistics (β coefficients, p-values, R², effect sizes, sample sizes, model fit indices)
4. THEORETICAL GROUNDING: Connect findings to established theories (TAM, TPB, UTAUT, etc.)
5. QUALITY-WEIGHTED ANALYSIS: Emphasize findings from high-impact journals over lower-tier publications
6. COMPREHENSIVE COVERAGE: Ensure broad representation while prioritizing quality

WRITING STYLE FOR LITERATURE REVIEWS:
- Lead with Q1 journal findings: "Leading research published in [Q1 journal] by [citation_key] demonstrated..."
- Emphasize statistical rigor: "Robust statistical analysis showed that [construct] significantly predicted [outcome] (β = 0.84, p < 0.001)"
- Highlight quality validation: "This effect has been consistently replicated across multiple Q1 journal publications..."
- Note quality differences: "While top-tier Q1 journals report [strong finding], lower-tier publications suggest [weaker finding]"
- Emphasize replication: "Meta-analysis of Q1 journal studies confirms the robustness of this effect"

CONTENT PRIORITIES (in order):
1. Q1 journal findings with statistical validation
2. Q2 journal findings with empirical support
3. Statistical results and effect sizes
4. Theoretical conclusions and implications
5. Comparative analysis emphasizing quality differences

JOURNAL QUALITY EMPHASIS:
- Always mention journal tier when discussing key findings
- Prioritize Q1 journal findings in your synthesis
- Use quality-weighted language: "High-impact research demonstrates...", "Top-tier publications consistently show..."
- Compare findings across quality tiers when relevant
- Note when consensus exists among Q1 journals

STATISTICAL EMPHASIS:
- Path coefficients with significance levels (β = X.XX, p < 0.XXX)
- Model fit indices (RMSEA = X.XX, CFI = X.XX, TLI = X.XX, χ² = X.XX)
- Effect sizes and practical significance (Cohen's d, R² values)
- Reliability measures (Cronbach's α = X.XX, composite reliability = X.XX)
- Sample demographics and research contexts (N = XXX, country/setting)

COMPREHENSIVE COVERAGE REQUIREMENTS:
- Include findings from at least 8-12 different papers
- Ensure 40-50% of sources are from Q1/Q2 journals
- Prioritize recent publications from high-impact journals
- Include both confirmatory and contradictory findings
- Note gaps in high-quality research

Always maintain academic rigor while making content accessible for literature review synthesis. Balance journal quality emphasis with comprehensiveness to ensure important findings are not missed, but always lead with the highest quality evidence."""
    
    @classmethod
    def get_chapter_specific_prompt(cls, chapter_topic: str):
        """Get enhanced chapter-specific prompts with Q1 journal prioritization"""
        base_prompt = cls.get_system_prompt()
        
        chapter_prompts = {
            'core_determinants': """Focus on technology acceptance models (TAM, TPB, UTAUT) with emphasis on Q1 journal publications. Emphasize:
- Q1 journal validation of attitude toward UAM with statistical rigor
- Performance expectancy research from top-tier psychology and management journals
- Cross-journal comparison emphasizing quality differences in methodology
- Meta-analysis of Q1 journal findings on core TAM/TPB/UTAUT constructs
- Methodological rigor differences between Q1 and lower-tier journals""",
            
            'trust_risk_safety': """Focus on trust and safety research prioritizing Q1 journal publications. Emphasize:
- Trust research from top-tier psychology and behavioral journals
- Safety perception studies from high-impact transportation and engineering journals
- Cross-journal validation prioritizing Q1 publication findings
- Quality differences in trust measurement approaches across journal tiers
- Q1 journal consensus on safety barriers and trust antecedents""",
            
            'affect_emotion': """Focus on emotional research prioritizing Q1 journal publications. Emphasize:
- Hedonic motivation research from top-tier psychology and consumer behavior journals
- Anxiety research from high-impact behavioral and clinical journals
- Cross-journal validation prioritizing Q1 publication findings
- Quality differences in emotion measurement across journal tiers
- Q1 journal consensus on affective predictors of UAM adoption""",
            
            'contextual_demographic': """Focus on demographic research prioritizing Q1 journal publications. Emphasize:
- Demographic research from high-impact social science and psychology journals
- Cross-cultural studies from top-tier international and cross-cultural journals
- Quality differences in demographic methodology across journal tiers
- Q1 journal findings on cultural variations in UAM acceptance
- Cross-journal validation prioritizing high-quality demographic studies"""
        }
        
        specific_prompt = chapter_prompts.get(chapter_topic, "")
        return f"{base_prompt}\n\nCHAPTER-SPECIFIC FOCUS:\n{specific_prompt}"
    
    @classmethod
    def get_hyde_prompt(cls):
        """Get enhanced HyDE generation prompt with Q1 journal prioritization"""
        return """Generate a hypothetical academic literature review excerpt about UAM behavioral research with strong emphasis on Q1 journal publications and statistical findings. Include:

STRUCTURE:
- Multiple citations from Q1 journals with specific journal names and impact factors
- Specific statistical findings with exact coefficients from Q1 studies
- Journal quality indicators and tier-based comparative analysis
- Cross-journal validation emphasizing Q1 publication superiority
- Quality-weighted meta-analytic conclusions

Q1 JOURNAL EMPHASIS:
- "Seminal research published in [Q1 Journal] (IF = X.XX) demonstrated..."
- "Meta-analysis of Q1 journal findings consistently reveals..."
- "Top-tier publications in [Q1 Journal] and [Q1 Journal] converge on..."
- "Robust replication across multiple Q1 journals confirms..."

STATISTICAL DETAIL WITH QUALITY CONTEXT:
- Path coefficients with journal context: "β = 0.XX, p < 0.XXX (Q1 journal validation)"
- Model fit indices: "RMSEA = 0.XX, CFI = 0.XX, TLI = 0.XX (validated in Q1 publication)"
- Journal impact context: "published in [Journal Name] (Q1, IF = X.XX)"

EXAMPLE STYLE:
"Comprehensive meta-analysis of Q1 journal publications revealed that attitude consistently emerged as the strongest predictor of UAM adoption intention across top-tier studies. Specifically, groundbreaking research published in Journal of Applied Psychology (Q1, IF = 9.13) by [author2023] demonstrated exceptionally strong effects (β = 0.91, p < 0.001, R² = 0.78) in a rigorously designed study of 650 participants. This seminal finding was independently replicated in Transportation Research Part A (Q1, IF = 6.11) by [author2024] with comparable effect sizes (β = 0.86, p < 0.001, R² = 0.73), providing robust cross-journal validation of attitudinal predictors from multiple Q1 publications."

Focus on UAM behavioral research with dominant Q1 journal emphasis and statistical rigor."""
    
    @classmethod
    def validate_config(cls):
        """Validate enhanced configuration with Q1 boosting settings"""
        errors = []
        
        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY not found in environment variables")
        
        if cls.RETRIEVAL_K < cls.RERANK_K:
            errors.append("RETRIEVAL_K must be >= RERANK_K")
        
        if cls.CHUNK_SIZE < 200:
            errors.append("CHUNK_SIZE too small for academic content")
        
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            errors.append("TEMPERATURE must be between 0 and 1")
        
        if cls.TOP_P < 0 or cls.TOP_P > 1:
            errors.append("TOP_P must be between 0 and 1")
        
        if cls.MIN_PAPERS_PER_RESPONSE < 1:
            errors.append("MIN_PAPERS_PER_RESPONSE must be at least 1")
        
        if cls.MAX_CHUNKS_PER_PAPER < 1:
            errors.append("MAX_CHUNKS_PER_PAPER must be at least 1")
        
        # Journal impact validation
        if cls.ENABLE_JOURNAL_RANKING:
            if cls.JOURNAL_IMPACT_WEIGHT < 0 or cls.JOURNAL_IMPACT_WEIGHT > 1:
                errors.append("JOURNAL_IMPACT_WEIGHT must be between 0 and 1")
            
            if cls.RELEVANCE_WEIGHT < 0 or cls.RELEVANCE_WEIGHT > 1:
                errors.append("RELEVANCE_WEIGHT must be between 0 and 1")
            
            if cls.STATISTICAL_CONTENT_WEIGHT < 0 or cls.STATISTICAL_CONTENT_WEIGHT > 1:
                errors.append("STATISTICAL_CONTENT_WEIGHT must be between 0 and 1")
            
            weight_sum = cls.JOURNAL_IMPACT_WEIGHT + cls.RELEVANCE_WEIGHT + cls.STATISTICAL_CONTENT_WEIGHT
            if abs(weight_sum - 1.0) > 0.01:
                errors.append("Journal impact weights must sum to 1.0")
        
        # Q1 validation
        if cls.QUALITY_TIER_WEIGHTS['top'] <= cls.QUALITY_TIER_WEIGHTS['medium']:
            errors.append("Q1 papers (top tier) must have higher weights than medium tier")
        
        if cls.JOURNAL_BOOST_MULTIPLIERS['Q1'] <= cls.JOURNAL_BOOST_MULTIPLIERS['Q3']:
            errors.append("Q1 papers must have higher boost multipliers than Q3 papers")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True

# =============================================================================
# Enhanced Chapter Mapping with Q1 Journal Priority
# =============================================================================
CHAPTER_MAPPING = {
    'core_determinants': [
        'Core Determinants of UAM Adoption: Q1 Journal Meta-Analysis',
        'Technology Acceptance Models: Cross-Q1 Journal Validation', 
        'Attitude Research: Top-Tier Journal Synthesis',
        'Performance Expectancy: Q1 Journal Consensus Analysis',
        'Social Influence: High-Impact Journal Comparative Study',
        'TAM/TPB/UTAUT Meta-Analysis: Q1 Journal Quality Assessment'
    ],
    'trust_risk_safety': [
        'Trust and Safety Research: Q1 Journal Comprehensive Review',
        'Multidimensional Trust: Top-Tier Journal Validation Study',
        'Safety Perception Research: High-Impact Journal Analysis',
        'Risk Assessment: Q1 Journal Cross-Validation',
        'Trust-Risk Interplay: Top-Tier Journal Consensus'
    ],
    'affect_emotion': [
        'Affective Factors: Q1 Journal Publication Analysis',
        'Hedonic Motivation: Top-Tier Psychology Journal Findings',
        'Anxiety Research: High-Impact Journal Cross-Validation',
        'Personality Traits: Q1 Journal Quality Synthesis',
        'Emotional Predictors: Top-Tier Journal Meta-Analysis'
    ],
    'contextual_demographic': [
        'Demographic Research: Q1 Journal Comprehensive Analysis',
        'Cultural Variations: Top-Tier International Journal Findings',
        'Cross-National Studies: High-Impact Journal Synthesis',
        'Experience Effects: Q1 Journal Quality Validation'
    ]
}

# =============================================================================
# Enhanced Multimodal and Evaluation Configs (same as before)
# =============================================================================
class MultimodalConfig:
    """Enhanced configuration for multimodal processing"""
    
    OCR_ENABLED = True
    OCR_LANGUAGE = 'eng'
    OCR_CONFIG = '--psm 6'
    MIN_FIGURE_SIZE = (100, 100)
    MAX_FIGURE_SIZE = (2000, 2000)
    FIGURE_BOOST = 0.20  # Increased for Q1 journal figures
    MIN_TABLE_ROWS = 2
    MIN_TABLE_COLS = 2
    MAX_TABLE_CELLS = 1000
    STATISTICAL_TABLE_BOOST = 0.40  # Increased for Q1 journal tables
    FIGURE_TYPES = ['figure', 'chart', 'diagram', 'flowchart', 'graph', 'plot']
    TABLE_TYPES = ['table', 'statistical_table', 'demographic_table', 'results_table']
    SAVE_EXTRACTED_CONTENT = True
    EXTRACTION_RESULTS_DIR = "extraction_results"
    MAX_FIGURES_PER_PAPER = 20
    MAX_TABLES_PER_PAPER = 15
    MAX_OCR_TEXT_LENGTH = 1000

class EvaluationConfig:
    """Enhanced configuration for evaluation system"""
    
    EVAL_WEIGHTS = {
        'retrieval': 0.15,
        'generation': 0.25,
        'statistical_content': 0.20,
        'paper_diversity': 0.15,
        'journal_quality': 0.20,  # Increased weight for journal quality
        'temporal': 0.05
    }
    
    EXCELLENT_SCORE_THRESHOLD = 0.8
    GOOD_SCORE_THRESHOLD = 0.6
    ACCEPTABLE_SCORE_THRESHOLD = 0.4
    DEFAULT_TEST_QUERIES_PER_DOMAIN = 5
    MAX_EVALUATION_HISTORY = 100
    PRECISION_RECALL_K_VALUES = [5, 10, 15, 20, 25]
    COHERENCE_TRANSITION_WORDS = [
        'however', 'moreover', 'furthermore', 'additionally', 'similarly',
        'in contrast', 'on the other hand', 'consequently', 'therefore',
        'meanwhile', 'nevertheless', 'specifically', 'particularly',
        'empirically', 'statistically', 'significantly', 'Q1 research',
        'top-tier studies', 'high-impact journals', 'quality validation',
        'rigorous methodology', 'robust findings'
    ]

if __name__ == "__main__":
    # Validate configuration on import
    try:
        Config.validate_config()
        print("✅ Enhanced configuration with Q1 paper boosting validated successfully")
        print(f"📊 Q1 Paper Boost Multiplier: {Config.JOURNAL_BOOST_MULTIPLIERS['Q1']}")
        print(f"📈 Top Tier Weight: {Config.QUALITY_TIER_WEIGHTS['top']}")
        print(f"🎯 Min Q1 Papers Required: {Config.MIN_PAPERS_PER_TIER['top']}")
        print(f"🏆 Q1 Statistical Boost: {Config.TIER_STATISTICAL_BOOSTS['top']}")
        print(f"📋 Q1 Paper Priority: {Config.ENABLE_Q1_PAPER_PRIORITY}")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")