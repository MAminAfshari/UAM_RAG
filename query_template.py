# query_template.py - Query Template System
"""
Query template system for the UAM Literature Review RAG System.
Refactored from uam_query_templates.py for better organization.
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ResearchDomain(Enum):
    """UAM research domains"""
    BEHAVIORAL_DETERMINANTS = "behavioral_determinants"
    TRUST_SAFETY_RISK = "trust_safety_risk"
    AFFECT_EMOTION = "affect_emotion"
    DEMOGRAPHICS = "demographics"
    CULTURAL_COMPARISON = "cultural_comparison"
    METHODOLOGY = "methodology"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    THEORETICAL_FRAMEWORKS = "theoretical_frameworks"


@dataclass
class QueryTemplate:
    """Template for generating research queries"""
    id: str
    domain: ResearchDomain
    template: str
    variables: List[str]
    expected_sections: List[str]
    statistical_focus: bool
    example_query: str
    evaluation_criteria: Dict[str, any]


class UAMQueryTemplateLibrary:
    """Comprehensive library of UAM research query templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.variable_options = self._initialize_variables()
    
    def _initialize_templates(self) -> List[QueryTemplate]:
        """Initialize all query templates"""
        return [
            # =============================================================================
            # Behavioral Determinants Templates
            # =============================================================================
            QueryTemplate(
                id="bd_001",
                domain=ResearchDomain.BEHAVIORAL_DETERMINANTS,
                template="What is the relationship between {construct1} and {construct2} in UAM adoption intention across different studies?",
                variables=["construct1", "construct2"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="What is the relationship between perceived usefulness and behavioral intention in UAM adoption across different studies?",
                evaluation_criteria={
                    "should_include": ["path coefficients", "significance levels", "multiple studies"],
                    "statistical_terms": ["β", "p-value", "R²"]
                }
            ),
            
            QueryTemplate(
                id="bd_002",
                domain=ResearchDomain.BEHAVIORAL_DETERMINANTS,
                template="How do the effects of {tam_construct} on UAM adoption compare between TAM, TPB, and UTAUT models?",
                variables=["tam_construct"],
                expected_sections=["results", "methodology", "discussion"],
                statistical_focus=True,
                example_query="How do the effects of perceived ease of use on UAM adoption compare between TAM, TPB, and UTAUT models?",
                evaluation_criteria={
                    "should_include": ["model comparison", "effect sizes", "theoretical frameworks"],
                    "models_mentioned": ["TAM", "TPB", "UTAUT"]
                }
            ),
            
            QueryTemplate(
                id="bd_003",
                domain=ResearchDomain.BEHAVIORAL_DETERMINANTS,
                template="What are the direct, indirect, and total effects of {predictor} on {outcome} in UAM acceptance models?",
                variables=["predictor", "outcome"],
                expected_sections=["results", "methodology"],
                statistical_focus=True,
                example_query="What are the direct, indirect, and total effects of social influence on adoption intention in UAM acceptance models?",
                evaluation_criteria={
                    "should_include": ["direct effects", "indirect effects", "mediation", "path analysis"],
                    "statistical_terms": ["direct effect", "indirect effect", "total effect", "mediation"]
                }
            ),
            
            # =============================================================================
            # Trust, Safety, and Risk Templates
            # =============================================================================
            QueryTemplate(
                id="tsr_001",
                domain=ResearchDomain.TRUST_SAFETY_RISK,
                template="How does {trust_dimension} influence UAM acceptance compared to {comparison_factor}?",
                variables=["trust_dimension", "comparison_factor"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="How does technology trust influence UAM acceptance compared to perceived safety?",
                evaluation_criteria={
                    "should_include": ["trust measures", "comparative analysis", "effect sizes"],
                    "concepts": ["trust", "safety", "risk"]
                }
            ),
            
            QueryTemplate(
                id="tsr_002",
                domain=ResearchDomain.TRUST_SAFETY_RISK,
                template="What is the mediating role of {mediator} in the relationship between {risk_type} and UAM adoption intention?",
                variables=["mediator", "risk_type"],
                expected_sections=["results", "methodology"],
                statistical_focus=True,
                example_query="What is the mediating role of trust in the relationship between perceived physical risk and UAM adoption intention?",
                evaluation_criteria={
                    "should_include": ["mediation analysis", "indirect effects", "bootstrapping"],
                    "statistical_terms": ["mediation", "indirect effect", "confidence interval"]
                }
            ),
            
            QueryTemplate(
                id="tsr_003",
                domain=ResearchDomain.TRUST_SAFETY_RISK,
                template="How do different dimensions of {safety_construct} (physical, operational, cyber, social) affect UAM acceptance?",
                variables=["safety_construct"],
                expected_sections=["results", "literature_review"],
                statistical_focus=True,
                example_query="How do different dimensions of perceived safety (physical, operational, cyber, social) affect UAM acceptance?",
                evaluation_criteria={
                    "should_include": ["multidimensional analysis", "safety dimensions", "comparative effects"],
                    "dimensions": ["physical", "operational", "cyber", "social"]
                }
            ),
            
            # =============================================================================
            # Affect and Emotion Templates
            # =============================================================================
            QueryTemplate(
                id="ae_001",
                domain=ResearchDomain.AFFECT_EMOTION,
                template="What is the relative importance of {emotional_factor} versus utilitarian factors in predicting UAM adoption?",
                variables=["emotional_factor"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="What is the relative importance of hedonic motivation versus utilitarian factors in predicting UAM adoption?",
                evaluation_criteria={
                    "should_include": ["hedonic vs utilitarian", "comparative analysis", "relative weights"],
                    "concepts": ["hedonic", "utilitarian", "emotional", "practical"]
                }
            ),
            
            QueryTemplate(
                id="ae_002",
                domain=ResearchDomain.AFFECT_EMOTION,
                template="How do personality traits like {trait1} and {trait2} moderate the relationship between attitude and UAM adoption intention?",
                variables=["trait1", "trait2"],
                expected_sections=["results", "methodology"],
                statistical_focus=True,
                example_query="How do personality traits like innovativeness and technology affinity moderate the relationship between attitude and UAM adoption intention?",
                evaluation_criteria={
                    "should_include": ["moderation analysis", "personality traits", "interaction effects"],
                    "statistical_terms": ["moderation", "interaction", "moderator"]
                }
            ),
            
            # =============================================================================
            # Demographics and Context Templates
            # =============================================================================
            QueryTemplate(
                id="dc_001",
                domain=ResearchDomain.DEMOGRAPHICS,
                template="How do demographic factors ({demo_var}) influence the effect of {predictor} on UAM adoption intention?",
                variables=["demo_var", "predictor"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="How do demographic factors (age, gender, income) influence the effect of perceived usefulness on UAM adoption intention?",
                evaluation_criteria={
                    "should_include": ["demographic analysis", "moderation", "group differences"],
                    "demographics": ["age", "gender", "income", "education"]
                }
            ),
            
            QueryTemplate(
                id="dc_002",
                domain=ResearchDomain.DEMOGRAPHICS,
                template="What are the differences in UAM acceptance determinants between {group1} and {group2}?",
                variables=["group1", "group2"],
                expected_sections=["results", "methodology"],
                statistical_focus=True,
                example_query="What are the differences in UAM acceptance determinants between urban and rural populations?",
                evaluation_criteria={
                    "should_include": ["group comparison", "multi-group analysis", "invariance testing"],
                    "statistical_terms": ["group differences", "invariance", "multi-group"]
                }
            ),
            
            # =============================================================================
            # Cultural Comparison Templates
            # =============================================================================
            QueryTemplate(
                id="cc_001",
                domain=ResearchDomain.CULTURAL_COMPARISON,
                template="How do UAM acceptance models differ between {country1} and {country2} in terms of {construct}?",
                variables=["country1", "country2", "construct"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="How do UAM acceptance models differ between China and the United States in terms of social influence?",
                evaluation_criteria={
                    "should_include": ["cross-cultural comparison", "country differences", "cultural factors"],
                    "analysis_type": ["multi-group", "invariance", "cultural dimensions"]
                }
            ),
            
            QueryTemplate(
                id="cc_002",
                domain=ResearchDomain.CULTURAL_COMPARISON,
                template="What cultural factors explain the variance in {construct} effects on UAM adoption across different countries?",
                variables=["construct"],
                expected_sections=["results", "literature_review", "discussion"],
                statistical_focus=True,
                example_query="What cultural factors explain the variance in trust effects on UAM adoption across different countries?",
                evaluation_criteria={
                    "should_include": ["cultural dimensions", "cross-national analysis", "variance explanation"],
                    "concepts": ["Hofstede", "cultural values", "collectivism", "individualism"]
                }
            ),
            
            # =============================================================================
            # Methodology Templates
            # =============================================================================
            QueryTemplate(
                id="mt_001",
                domain=ResearchDomain.METHODOLOGY,
                template="What are the most common {method_aspect} used in UAM acceptance studies and their impact on findings?",
                variables=["method_aspect"],
                expected_sections=["methodology", "results"],
                statistical_focus=False,
                example_query="What are the most common measurement scales used in UAM acceptance studies and their impact on findings?",
                evaluation_criteria={
                    "should_include": ["methodological review", "measurement", "reliability", "validity"],
                    "method_terms": ["scale", "instrument", "reliability", "validity"]
                }
            ),
            
            QueryTemplate(
                id="mt_002",
                domain=ResearchDomain.METHODOLOGY,
                template="How do different research designs ({design_type}) affect the reported relationships between {construct} and UAM adoption?",
                variables=["design_type", "construct"],
                expected_sections=["methodology", "results", "discussion"],
                statistical_focus=True,
                example_query="How do different research designs (experimental vs survey) affect the reported relationships between trust and UAM adoption?",
                evaluation_criteria={
                    "should_include": ["research design comparison", "methodological effects", "validity threats"],
                    "designs": ["experimental", "survey", "longitudinal", "cross-sectional"]
                }
            ),
            
            # =============================================================================
            # Temporal Evolution Templates
            # =============================================================================
            QueryTemplate(
                id="te_001",
                domain=ResearchDomain.TEMPORAL_EVOLUTION,
                template="How has the effect of {construct} on UAM adoption intention evolved from {start_year} to {end_year}?",
                variables=["construct", "start_year", "end_year"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="How has the effect of perceived risk on UAM adoption intention evolved from 2015 to 2024?",
                evaluation_criteria={
                    "should_include": ["temporal analysis", "trend", "longitudinal changes"],
                    "temporal_terms": ["over time", "evolution", "trend", "temporal"]
                }
            ),
            
            QueryTemplate(
                id="te_002",
                domain=ResearchDomain.TEMPORAL_EVOLUTION,
                template="What are the emerging {factor_type} in recent UAM acceptance studies (post-{year}) compared to earlier research?",
                variables=["factor_type", "year"],
                expected_sections=["literature_review", "results", "discussion"],
                statistical_focus=False,
                example_query="What are the emerging psychological factors in recent UAM acceptance studies (post-2020) compared to earlier research?",
                evaluation_criteria={
                    "should_include": ["temporal comparison", "emerging factors", "recent developments"],
                    "time_markers": ["recent", "emerging", "new", "novel"]
                }
            ),
            
            # =============================================================================
            # Theoretical Framework Templates
            # =============================================================================
            QueryTemplate(
                id="tf_001",
                domain=ResearchDomain.THEORETICAL_FRAMEWORKS,
                template="How well do {theory1} and {theory2} explain UAM adoption intention and what are their unique contributions?",
                variables=["theory1", "theory2"],
                expected_sections=["literature_review", "results", "discussion"],
                statistical_focus=True,
                example_query="How well do TAM and UTAUT explain UAM adoption intention and what are their unique contributions?",
                evaluation_criteria={
                    "should_include": ["theory comparison", "explanatory power", "R²", "model fit"],
                    "statistical_terms": ["R²", "explanatory power", "variance explained", "model fit"]
                }
            ),
            
            QueryTemplate(
                id="tf_002",
                domain=ResearchDomain.THEORETICAL_FRAMEWORKS,
                template="What theoretical extensions or modifications to {base_theory} have been proposed specifically for UAM acceptance?",
                variables=["base_theory"],
                expected_sections=["literature_review", "discussion"],
                statistical_focus=False,
                example_query="What theoretical extensions or modifications to TAM have been proposed specifically for UAM acceptance?",
                evaluation_criteria={
                    "should_include": ["theoretical extensions", "model modifications", "new constructs"],
                    "theory_terms": ["extension", "modification", "adapted", "extended model"]
                }
            ),
            
            # =============================================================================
            # Meta-Analytical Templates
            # =============================================================================
            QueryTemplate(
                id="ma_001",
                domain=ResearchDomain.BEHAVIORAL_DETERMINANTS,
                template="What is the pooled effect size of {construct} on UAM adoption intention across all available studies?",
                variables=["construct"],
                expected_sections=["results", "methodology"],
                statistical_focus=True,
                example_query="What is the pooled effect size of perceived usefulness on UAM adoption intention across all available studies?",
                evaluation_criteria={
                    "should_include": ["meta-analysis", "pooled effect", "heterogeneity", "multiple studies"],
                    "meta_terms": ["pooled", "combined", "across studies", "meta-analysis"]
                }
            ),
            
            # =============================================================================
            # Complex Synthesis Templates
            # =============================================================================
            QueryTemplate(
                id="cs_001",
                domain=ResearchDomain.BEHAVIORAL_DETERMINANTS,
                template="What are the conflicting findings regarding {construct}'s effect on UAM adoption and how can they be reconciled?",
                variables=["construct"],
                expected_sections=["results", "discussion"],
                statistical_focus=True,
                example_query="What are the conflicting findings regarding social influence's effect on UAM adoption and how can they be reconciled?",
                evaluation_criteria={
                    "should_include": ["contradictions", "conflicting findings", "reconciliation", "moderators"],
                    "synthesis_terms": ["conflicting", "contradictory", "inconsistent", "reconcile"]
                }
            )
        ]
    
    def _initialize_variables(self) -> Dict[str, List[str]]:
        """Initialize variable options for templates"""
        return {
            "construct1": ["perceived usefulness", "perceived ease of use", "attitude", "social influence", "trust"],
            "construct2": ["behavioral intention", "adoption intention", "willingness to use", "actual use"],
            "tam_construct": ["perceived usefulness", "perceived ease of use", "attitude", "behavioral intention"],
            "predictor": ["trust", "perceived risk", "social influence", "performance expectancy", "attitude"],
            "outcome": ["adoption intention", "willingness to use", "behavioral intention", "actual use"],
            "trust_dimension": ["technology trust", "institutional trust", "interpersonal trust", "dispositional trust"],
            "comparison_factor": ["perceived risk", "perceived safety", "perceived usefulness", "social influence"],
            "mediator": ["trust", "attitude", "perceived safety", "anxiety"],
            "risk_type": ["physical risk", "performance risk", "financial risk", "privacy risk", "social risk"],
            "safety_construct": ["perceived safety", "safety concerns", "safety perception"],
            "emotional_factor": ["hedonic motivation", "anxiety", "enjoyment", "excitement"],
            "trait1": ["innovativeness", "technology affinity", "risk-taking propensity", "openness"],
            "trait2": ["need for cognition", "environmental consciousness", "sensation seeking", "trust propensity"],
            "demo_var": ["age", "gender", "income", "education", "experience"],
            "group1": ["young adults", "urban residents", "early adopters", "high-income individuals"],
            "group2": ["older adults", "rural residents", "late adopters", "low-income individuals"],
            "country1": ["China", "United States", "Germany", "South Korea", "Japan"],
            "country2": ["United Kingdom", "India", "Brazil", "Australia", "Canada"],
            "method_aspect": ["measurement scales", "sampling methods", "analytical techniques", "experimental designs"],
            "design_type": ["experimental", "survey", "longitudinal", "cross-sectional", "mixed methods"],
            "factor_type": ["psychological factors", "contextual factors", "technological factors", "social factors"],
            "theory1": ["TAM", "TPB", "UTAUT", "DOI", "Social Cognitive Theory"],
            "theory2": ["UTAUT2", "TAM3", "C-TAM-TPB", "Risk Perception Theory", "Trust Theory"],
            "base_theory": ["TAM", "TPB", "UTAUT", "DOI"],
            "start_year": ["2015", "2018", "2020"],
            "end_year": ["2023", "2024", "2025"],
            "year": ["2020", "2021", "2022"],
            "construct": ["trust", "perceived usefulness", "social influence", "attitude", "perceived risk"]
        }
    
    # =============================================================================
    # Core Template Operations
    # =============================================================================
    
    def generate_query(self, template_id: str, **kwargs) -> str:
        """Generate a query from a template with specified variables"""
        template = next((t for t in self.templates if t.id == template_id), None)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        query = template.template
        for var in template.variables:
            if var in kwargs:
                query = query.replace(f"{{{var}}}", kwargs[var])
            else:
                # Use random default if not specified
                if var in self.variable_options:
                    default_value = random.choice(self.variable_options[var])
                    query = query.replace(f"{{{var}}}", default_value)
        
        return query
    
    def get_template_by_id(self, template_id: str) -> Optional[QueryTemplate]:
        """Get a specific template by ID"""
        return next((t for t in self.templates if t.id == template_id), None)
    
    def get_template_by_domain(self, domain: ResearchDomain) -> List[QueryTemplate]:
        """Get all templates for a specific domain"""
        return [t for t in self.templates if t.domain == domain]
    
    def get_statistical_templates(self) -> List[QueryTemplate]:
        """Get all templates that focus on statistical analysis"""
        return [t for t in self.templates if t.statistical_focus]
    
    def get_templates_by_keyword(self, keyword: str) -> List[QueryTemplate]:
        """Get templates that contain a specific keyword"""
        keyword_lower = keyword.lower()
        matching_templates = []
        
        for template in self.templates:
            if (keyword_lower in template.template.lower() or 
                keyword_lower in template.example_query.lower() or
                any(keyword_lower in str(criterion).lower() 
                    for criterion in template.evaluation_criteria.values())):
                matching_templates.append(template)
        
        return matching_templates
    
    # =============================================================================
    # Query Generation and Testing
    # =============================================================================
    
    def generate_test_suite(self, n_queries_per_domain: int = 2) -> List[Dict]:
        """Generate a comprehensive test suite of queries"""
        test_suite = []
        
        for domain in ResearchDomain:
            domain_templates = self.get_template_by_domain(domain)
            selected_templates = random.sample(
                domain_templates, 
                min(n_queries_per_domain, len(domain_templates))
            )
            
            for template in selected_templates:
                # Generate query with random variables
                variables = {}
                for var in template.variables:
                    if var in self.variable_options:
                        variables[var] = random.choice(self.variable_options[var])
                
                query = self.generate_query(template.id, **variables)
                
                test_case = {
                    'query': query,
                    'template_id': template.id,
                    'domain': domain.value,
                    'expected_sections': template.expected_sections,
                    'statistical_focus': template.statistical_focus,
                    'evaluation_criteria': template.evaluation_criteria,
                    'variables_used': variables
                }
                
                test_suite.append(test_case)
        
        return test_suite
    
    def get_progressive_queries(self, topic: str) -> List[Tuple[str, str]]:
        """Generate a series of progressively complex queries on a topic"""
        progressive_queries = []
        
        # Level 1: Basic relationship
        query1 = self.generate_query("bd_001", 
                                   construct1=topic, 
                                   construct2="adoption intention")
        progressive_queries.append(("Basic Relationship", query1))
        
        # Level 2: Comparative analysis
        query2 = self.generate_query("bd_002", 
                                   tam_construct=topic)
        progressive_queries.append(("Model Comparison", query2))
        
        # Level 3: Mediation/Moderation
        query3 = self.generate_query("bd_003", 
                                   predictor=topic, 
                                   outcome="adoption intention")
        progressive_queries.append(("Mediation Analysis", query3))
        
        # Level 4: Cultural comparison
        query4 = self.generate_query("cc_001", 
                                   country1="China", 
                                   country2="United States", 
                                   construct=topic)
        progressive_queries.append(("Cultural Comparison", query4))
        
        # Level 5: Temporal evolution
        query5 = self.generate_query("te_001", 
                                   construct=topic, 
                                   start_year="2015", 
                                   end_year="2024")
        progressive_queries.append(("Temporal Evolution", query5))
        
        # Level 6: Contradiction synthesis
        query6 = self.generate_query("cs_001", 
                                   construct=topic)
        progressive_queries.append(("Contradiction Synthesis", query6))
        
        return progressive_queries
    
    def generate_domain_focused_queries(self, domain: ResearchDomain, n_queries: int = 5) -> List[str]:
        """Generate queries focused on a specific domain"""
        domain_templates = self.get_template_by_domain(domain)
        
        if not domain_templates:
            return []
        
        queries = []
        for i in range(n_queries):
            template = random.choice(domain_templates)
            
            # Generate variables
            variables = {}
            for var in template.variables:
                if var in self.variable_options:
                    variables[var] = random.choice(self.variable_options[var])
            
            query = self.generate_query(template.id, **variables)
            queries.append(query)
        
        return queries
    
    # =============================================================================
    # Query Validation and Analysis
    # =============================================================================
    
    def validate_query_response(self, 
                              query: str, 
                              response: str, 
                              template_id: Optional[str] = None) -> Dict[str, any]:
        """Validate if a response adequately addresses a query based on template criteria"""
        validation_result = {
            'query': query,
            'template_id': template_id,
            'criteria_met': {},
            'missing_elements': [],
            'score': 0.0
        }
        
        if template_id:
            template = self.get_template_by_id(template_id)
            if template and template.evaluation_criteria:
                criteria = template.evaluation_criteria
                
                # Check for required inclusions
                if 'should_include' in criteria:
                    for element in criteria['should_include']:
                        if element.lower() in response.lower():
                            validation_result['criteria_met'][element] = True
                        else:
                            validation_result['missing_elements'].append(element)
                
                # Check for statistical terms if required
                if 'statistical_terms' in criteria:
                    found_stats = []
                    for term in criteria['statistical_terms']:
                        if term in response or term.lower() in response.lower():
                            found_stats.append(term)
                    validation_result['criteria_met']['statistical_terms'] = found_stats
                
                # Check for theory mentions
                if 'models_mentioned' in criteria:
                    found_models = []
                    for model in criteria['models_mentioned']:
                        if model in response:
                            found_models.append(model)
                    validation_result['criteria_met']['models'] = found_models
                
                # Calculate score
                total_criteria = len(criteria.get('should_include', [])) + \
                               len(criteria.get('statistical_terms', [])) + \
                               len(criteria.get('models_mentioned', []))
                
                met_criteria = len(validation_result['criteria_met'].get('should_include', [])) + \
                              len(validation_result['criteria_met'].get('statistical_terms', [])) + \
                              len(validation_result['criteria_met'].get('models', []))
                
                validation_result['score'] = met_criteria / total_criteria if total_criteria > 0 else 0
        
        return validation_result
    
    def analyze_query_complexity(self, query: str) -> Dict[str, any]:
        """Analyze the complexity of a query"""
        complexity_analysis = {
            'word_count': len(query.split()),
            'question_words': 0,
            'comparative_terms': 0,
            'statistical_terms': 0,
            'temporal_terms': 0,
            'complexity_score': 0.0
        }
        
        query_lower = query.lower()
        
        # Count question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        complexity_analysis['question_words'] = sum(1 for word in question_words if word in query_lower)
        
        # Count comparative terms
        comparative_terms = ['compare', 'versus', 'difference', 'between', 'contrast', 'relative']
        complexity_analysis['comparative_terms'] = sum(1 for term in comparative_terms if term in query_lower)
        
        # Count statistical terms
        statistical_terms = ['coefficient', 'effect', 'correlation', 'regression', 'analysis', 'model']
        complexity_analysis['statistical_terms'] = sum(1 for term in statistical_terms if term in query_lower)
        
        # Count temporal terms
        temporal_terms = ['evolution', 'change', 'trend', 'over time', 'temporal', 'longitudinal']
        complexity_analysis['temporal_terms'] = sum(1 for term in temporal_terms if term in query_lower)
        
        # Calculate complexity score
        complexity_score = (
            complexity_analysis['word_count'] * 0.1 +
            complexity_analysis['question_words'] * 0.2 +
            complexity_analysis['comparative_terms'] * 0.3 +
            complexity_analysis['statistical_terms'] * 0.2 +
            complexity_analysis['temporal_terms'] * 0.2
        )
        
        complexity_analysis['complexity_score'] = min(complexity_score, 10.0)  # Cap at 10
        
        return complexity_analysis
    
    # =============================================================================
    # Import/Export Functions
    # =============================================================================
    
    def export_templates(self, output_path: str = "uam_query_templates.json"):
        """Export all templates to JSON format"""
        export_data = {
            'templates': [
                {
                    'id': t.id,
                    'domain': t.domain.value,
                    'template': t.template,
                    'variables': t.variables,
                    'expected_sections': t.expected_sections,
                    'statistical_focus': t.statistical_focus,
                    'example_query': t.example_query,
                    'evaluation_criteria': t.evaluation_criteria
                }
                for t in self.templates
            ],
            'variable_options': self.variable_options,
            'domains': [d.value for d in ResearchDomain]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data
    
    def import_templates(self, input_path: str):
        """Import templates from JSON format"""
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        # Update variable options
        if 'variable_options' in import_data:
            self.variable_options.update(import_data['variable_options'])
        
        # Import templates
        if 'templates' in import_data:
            for template_data in import_data['templates']:
                domain = ResearchDomain(template_data['domain'])
                template = QueryTemplate(
                    id=template_data['id'],
                    domain=domain,
                    template=template_data['template'],
                    variables=template_data['variables'],
                    expected_sections=template_data['expected_sections'],
                    statistical_focus=template_data['statistical_focus'],
                    example_query=template_data['example_query'],
                    evaluation_criteria=template_data['evaluation_criteria']
                )
                
                # Replace if exists, otherwise add
                existing_index = next((i for i, t in enumerate(self.templates) if t.id == template.id), None)
                if existing_index is not None:
                    self.templates[existing_index] = template
                else:
                    self.templates.append(template)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the template library"""
        domain_counts = {}
        for domain in ResearchDomain:
            domain_counts[domain.value] = len(self.get_template_by_domain(domain))
        
        statistical_count = len(self.get_statistical_templates())
        
        return {
            'total_templates': len(self.templates),
            'domain_distribution': domain_counts,
            'statistical_templates': statistical_count,
            'non_statistical_templates': len(self.templates) - statistical_count,
            'total_variables': len(self.variable_options),
            'domains': len(ResearchDomain)
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_ground_truth_examples():
    """Create example ground truth data for evaluation"""
    ground_truth = {
        "test_queries": [
            {
                "query": "What is the relationship between perceived usefulness and behavioral intention in UAM adoption?",
                "expected_papers": ["chen2023", "zhang2024", "smith2023", "lee2022", "wang2023"],
                "expected_findings": [
                    {
                        "key_terms": ["perceived usefulness", "positive effect", "behavioral intention"],
                        "expected_statistics": ["β > 0.5", "p < 0.001"]
                    }
                ]
            },
            {
                "query": "How does trust influence UAM acceptance compared to perceived risk?",
                "expected_papers": ["johnson2023", "kim2024", "garcia2023"],
                "expected_findings": [
                    {
                        "key_terms": ["trust", "stronger predictor", "perceived risk"],
                        "expected_statistics": ["comparative analysis", "effect sizes"]
                    }
                ]
            }
        ],
        "temporal_trends": {
            "trust": {
                "trend": "increasing",
                "change_points": [2020],
                "recent_effect": 0.65
            }
        },
        "contradictions": [
            "social influence effects vary by culture",
            "age moderates technology acceptance differently across studies"
        ]
    }
    
    # Save ground truth
    with open('uam_ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth


def demonstrate_template_usage():
    """Demonstrate how to use the template library"""
    print("🎯 UAM Query Template Library Demo")
    print("="*50)
    
    # Initialize library
    library = UAMQueryTemplateLibrary()
    
    # Show statistics
    stats = library.get_statistics()
    print(f"📊 Library Statistics:")
    print(f"  Total Templates: {stats['total_templates']}")
    print(f"  Domains: {stats['domains']}")
    print(f"  Statistical Templates: {stats['statistical_templates']}")
    
    # Show domain distribution
    print(f"\n📑 Domain Distribution:")
    for domain, count in stats['domain_distribution'].items():
        print(f"  {domain}: {count} templates")
    
    # Generate example queries
    print(f"\n🔍 Example Queries:")
    for domain in ResearchDomain:
        templates = library.get_template_by_domain(domain)
        if templates:
            template = templates[0]  # Get first template
            print(f"\n{domain.value}:")
            print(f"  {template.example_query}")
    
    # Generate test suite
    print(f"\n🧪 Test Suite Generation:")
    test_suite = library.generate_test_suite(n_queries_per_domain=1)
    print(f"  Generated {len(test_suite)} test queries")
    
    # Show progressive complexity
    print(f"\n📈 Progressive Complexity for 'trust':")
    progressive = library.get_progressive_queries("trust")
    for i, (level, query) in enumerate(progressive, 1):
        print(f"  Level {i} ({level}): {query[:80]}...")


if __name__ == "__main__":
    # Demo the template library
    demonstrate_template_usage()
    
    # Create ground truth examples
    create_ground_truth_examples()
    
    print("\n✅ Query template system ready!")
    print("📁 Ground truth examples saved to 'uam_ground_truth.json'")