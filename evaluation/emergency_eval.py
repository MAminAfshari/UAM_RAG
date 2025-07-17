"""
Emergency Evaluation
Quick evaluation that works without ground truth data.
"""

import re
from typing import Dict, List, Tuple


def emergency_evaluation(rag_system) -> Tuple[float, List[Dict]]:
    """Quick evaluation that works without ground truth"""
    
    test_queries = [
        "What is the relationship between trust and UAM adoption?",
        "How does perceived usefulness affect UAM acceptance?", 
        "What are the main barriers to UAM adoption?",
        "What statistical models are used in UAM research?"
    ]
    
    print("🚀 EMERGENCY RAG EVALUATION")
    print("="*60)
    print("Testing core functionality without ground truth data")
    print("="*60)
    
    total_score = 0
    detailed_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        try:
            # Get response
            response, sources = rag_system.answer_literature_query(query)
            
            # Analyze response quality
            analysis = analyze_response_quality(response, sources, query)
            
            # Print results
            print_analysis(analysis)
            
            total_score += analysis['total_score']
            detailed_results.append({
                'query': query,
                'analysis': analysis,
                'response_preview': response[:200] + "..." if len(response) > 200 else response
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            detailed_results.append({
                'query': query,
                'error': str(e),
                'analysis': {'total_score': 0}
            })
    
    # Overall results
    overall_score = total_score / len(test_queries)
    print("\n" + "="*60)
    print("🎯 OVERALL RESULTS")
    print("="*60)
    
    status = get_status(overall_score)
    print(f"Overall Score: {overall_score:.2f} {status}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    recommendations = generate_recommendations(detailed_results)
    for rec in recommendations:
        print(f"• {rec}")
    
    return overall_score, detailed_results


def analyze_response_quality(response: str, sources: List[str], query: str) -> Dict:
    """Analyze response quality on measurable dimensions"""
    
    analysis = {
        'response_length': len(response),
        'sources_count': len(sources),
        'unique_sources': len(set(sources)) if sources else 0,
        'has_citations': bool(re.search(r'\[[^\]]+\]', response)),
        'citation_count': len(re.findall(r'\[[^\]]+\]', response)),
        'has_statistics': bool(re.search(r'β\s*=|p\s*[<>=]|N\s*=|R²\s*=', response)),
        'statistic_count': len(re.findall(r'β\s*=\s*[0-9.-]+|p\s*[<>=]\s*[0-9.]+|N\s*=\s*[0-9,]+', response)),
        'has_academic_language': bool(re.search(r'study|research|found|showed|reported|significant', response, re.IGNORECASE)),
        'has_comparisons': bool(re.search(r'compared|versus|however|in contrast|similarly', response, re.IGNORECASE)),
        'paragraph_count': len([p for p in response.split('\n\n') if p.strip()]),
        'coherence_indicators': count_coherence_indicators(response)
    }
    
    # Calculate component scores
    scores = calculate_component_scores(analysis)
    analysis.update(scores)
    
    return analysis


def count_coherence_indicators(text: str) -> Dict:
    """Count indicators of good academic writing structure"""
    indicators = {
        'transitions': len(re.findall(r'\b(however|moreover|furthermore|additionally|similarly|in contrast|consequently)\b', text, re.IGNORECASE)),
        'sequence_markers': len(re.findall(r'\b(first|second|third|finally|additionally)\b', text, re.IGNORECASE)),
        'evidence_phrases': len(re.findall(r'\b(found that|showed that|demonstrated|indicated|reported)\b', text, re.IGNORECASE)),
        'quantifiers': len(re.findall(r'\b(significant|strong|weak|moderate|substantial)\b', text, re.IGNORECASE))
    }
    return indicators


def calculate_component_scores(analysis: Dict) -> Dict:
    """Calculate scores for different components"""
    
    # Source Quality Score (0-1)
    source_score = 0
    if analysis['sources_count'] > 0:
        source_score += 0.3
    if analysis['sources_count'] >= 3:
        source_score += 0.3
    if analysis['unique_sources'] >= 2:
        source_score += 0.4
    
    # Citation Quality Score (0-1)
    citation_score = 0
    if analysis['has_citations']:
        citation_score += 0.5
    if analysis['citation_count'] >= 3:
        citation_score += 0.3
    if analysis['citation_count'] >= 5:
        citation_score += 0.2
    
    # Statistical Content Score (0-1)
    stats_score = 0
    if analysis['has_statistics']:
        stats_score += 0.6
    if analysis['statistic_count'] >= 2:
        stats_score += 0.4
    
    # Academic Quality Score (0-1)
    academic_score = 0
    if analysis['has_academic_language']:
        academic_score += 0.3
    if analysis['has_comparisons']:
        academic_score += 0.3
    if analysis['paragraph_count'] >= 2:
        academic_score += 0.2
    if sum(analysis['coherence_indicators'].values()) >= 3:
        academic_score += 0.2
    
    # Length Quality Score (0-1)
    length_score = 0
    if 200 <= analysis['response_length'] <= 2000:
        length_score = 1.0
    elif 100 <= analysis['response_length'] < 200:
        length_score = 0.7
    elif analysis['response_length'] > 2000:
        length_score = 0.8
    else:
        length_score = 0.3
    
    # Overall score (weighted average)
    total_score = (
        source_score * 0.25 +
        citation_score * 0.25 +
        stats_score * 0.20 +
        academic_score * 0.20 +
        length_score * 0.10
    )
    
    return {
        'source_score': source_score,
        'citation_score': citation_score,
        'stats_score': stats_score,
        'academic_score': academic_score,
        'length_score': length_score,
        'total_score': total_score
    }


def print_analysis(analysis: Dict):
    """Print analysis results"""
    
    def status_icon(score):
        return "✅" if score > 0.7 else "⚠️" if score > 0.4 else "❌"
    
    print(f"   Sources: {analysis['sources_count']} unique: {analysis['unique_sources']} {status_icon(analysis['source_score'])} ({analysis['source_score']:.2f})")
    print(f"   Citations: {analysis['citation_count']} found {status_icon(analysis['citation_score'])} ({analysis['citation_score']:.2f})")
    print(f"   Statistics: {analysis['statistic_count']} found {status_icon(analysis['stats_score'])} ({analysis['stats_score']:.2f})")
    print(f"   Academic Quality: {status_icon(analysis['academic_score'])} ({analysis['academic_score']:.2f})")
    print(f"   Length: {analysis['response_length']} chars {status_icon(analysis['length_score'])} ({analysis['length_score']:.2f})")
    print(f"   → Total Score: {analysis['total_score']:.2f} {status_icon(analysis['total_score'])}")


def get_status(score: float) -> str:
    """Get status emoji and text"""
    if score > 0.75:
        return "✅ Excellent"
    elif score > 0.6:
        return "✅ Good"
    elif score > 0.45:
        return "⚠️ Needs Improvement"
    else:
        return "❌ Critical Issues"


def generate_recommendations(results: List[Dict]) -> List[str]:
    """Generate specific recommendations"""
    recommendations = []
    
    # Analyze patterns
    avg_sources = sum(r['analysis']['sources_count'] for r in results if 'analysis' in r) / len(results)
    avg_citations = sum(r['analysis']['citation_count'] for r in results if 'analysis' in r) / len(results)
    has_stats = sum(1 for r in results if 'analysis' in r and r['analysis']['has_statistics'])
    
    if avg_sources < 2:
        recommendations.append("🔍 LOW SOURCE RETRIEVAL: Increase search parameters or check corpus size")
    
    if avg_citations < 2:
        recommendations.append("📝 CITATION ISSUES: Verify citation extraction patterns in responses")
    
    if has_stats < len(results) / 2:
        recommendations.append("📊 STATISTICAL REPORTING: Improve statistical information extraction")
    
    errors = sum(1 for r in results if 'error' in r)
    if errors > 0:
        recommendations.append(f"🚨 SYSTEM ERRORS: {errors} queries failed - check logs")
    
    empty_responses = sum(1 for r in results if 'analysis' in r and r['analysis']['response_length'] < 50)
    if empty_responses > 0:
        recommendations.append("📭 EMPTY RESPONSES: Check if papers are properly ingested")
    
    if not recommendations:
        recommendations.append("🎉 System appears to be working well! Consider fine-tuning for better performance.")
    
    return recommendations