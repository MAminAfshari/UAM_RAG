# eval.py - Main Evaluation Entry Point
"""
Main evaluation script for the UAM Literature Review RAG System.
Refactored from eval_script.py for better modularity and maintainability.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import core system and evaluation components
from core.rag_system import UAMRAGSystem
from evaluation.evaluator import UAMResearchEvaluator
from evaluation.metrics import RAGEvaluationSystem, EvaluationResult
from evaluation.emergency_eval import emergency_evaluation
from query_template import UAMQueryTemplateLibrary, ResearchDomain
from config import Config


class EvaluationOrchestrator:
    """Main orchestrator for comprehensive RAG system evaluation"""
    
    def __init__(self):
        self.rag_system = None
        self.evaluator = None
        self.integrated_evaluator = None
        self.results = {}
        
    def setup_evaluation_environment(self):
        """Set up the evaluation environment"""
        print("🚀 Setting up UAM RAG Evaluation Environment")
        print("="*60)
        
        try:
            # Initialize RAG system
            print("1. Initializing RAG system...")
            self.rag_system = UAMRAGSystem()
            
            # Check if papers are ingested
            stats = self.rag_system.get_corpus_statistics()
            if not stats or stats.get('total_papers', 0) == 0:
                print("❌ No papers found in corpus. Please run ingestion first.")
                return False
            
            print(f"✅ Found {stats.get('total_papers', 0)} papers in corpus")
            
            # Initialize evaluation components
            print("\n2. Initializing evaluation components...")
            self.evaluator = RAGEvaluationSystem(self.rag_system)
            self.integrated_evaluator = UAMResearchEvaluator(self.rag_system)
            
            print("✅ Evaluation environment ready")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup evaluation environment: {e}")
            return False
    
    def run_quick_health_check(self) -> Dict:
        """Run a quick health check with simple queries"""
        print("\n🏥 Running Quick Health Check")
        print("="*60)
        
        health_check_queries = [
            "What is the relationship between trust and UAM adoption?",
            "How does perceived usefulness affect UAM acceptance?", 
            "What are the main barriers to UAM adoption?",
            "What statistical models are used in UAM research?"
        ]
        
        results = []
        for query in health_check_queries:
            print(f"\nTesting: {query}")
            result = self.evaluator.evaluate_query(query, evaluate_all=False)
            
            score = result.overall_score
            status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
            
            print(f"  Score: {score:.2f} {status}")
            print(f"  Retrieval F1: {np.mean(list(result.retrieval_metrics.f1_at_k.values())):.2f}")
            print(f"  Generation Coherence: {result.generation_metrics.coherence_score:.2f}")
            
            results.append({
                'query': query,
                'score': score,
                'status': status
            })
        
        avg_score = np.mean([r['score'] for r in results])
        overall_status = "✅ Healthy" if avg_score > 0.7 else "⚠️ Needs Attention" if avg_score > 0.5 else "❌ Critical"
        
        print(f"\n📊 Overall Health: {overall_status} (Average Score: {avg_score:.2f})")
        
        return {
            'results': results,
            'average_score': avg_score,
            'status': overall_status
        }
    
    def run_focused_domain_evaluation(self, domain: ResearchDomain, detailed: bool = True) -> pd.DataFrame:
        """Run detailed evaluation for a specific domain"""
        print(f"\n🎯 Focused Evaluation: {domain.value}")
        print("="*60)
        
        # Run benchmark for domain
        results = self.integrated_evaluator.run_domain_benchmark(domain, n_queries=5)
        
        if detailed:
            print("\nDetailed Results:")
            print(results.to_string(index=False))
            
            # Identify weak areas
            weak_areas = results[results['overall_score'] < 0.7]
            if not weak_areas.empty:
                print("\n⚠️ Queries needing improvement:")
                for _, row in weak_areas.iterrows():
                    print(f"  - {row['query']}")
                    print(f"    Score: {row['overall_score']:.2f}")
                    print(f"    Missing elements: {row['missing_elements']}")
        
        return results
    
    def test_statistical_reporting_quality(self) -> Dict:
        """Test quality of statistical reporting"""
        print("\n📊 Testing Statistical Reporting Quality")
        print("="*60)
        
        # Statistical reporting test queries
        stat_queries = [
            "What are the path coefficients between trust and UAM adoption intention?",
            "What is the effect size of perceived risk on UAM acceptance?",
            "What are the R-squared values for UAM acceptance models?"
        ]
        
        results = []
        for query in stat_queries:
            print(f"\nQuery: {query}")
            response, _ = self.rag_system.answer_literature_query(query)
            
            # Extract statistics
            stats = self.evaluator._extract_statistics(response)
            
            # Count different types
            stat_types = {}
            for stat in stats:
                stat_type = stat['type']
                stat_types[stat_type] = stat_types.get(stat_type, 0) + 1
            
            print(f"  Statistics found: {stat_types}")
            
            results.append({
                'query': query,
                'total_stats': len(stats),
                'stat_types': stat_types
            })
        
        return results
    
    def run_progressive_complexity_test(self, topic: str = "perceived usefulness") -> pd.DataFrame:
        """Test system performance on increasingly complex queries"""
        print(f"\n📈 Progressive Complexity Test: {topic}")
        print("="*60)
        
        results = self.integrated_evaluator.test_progressive_complexity(topic)
        
        # Visualize progression
        print("\nComplexity Progression:")
        for _, row in results.iterrows():
            level = row['complexity_level']
            score = row['overall_score']
            time = row['execution_time']
            
            # Create visual bar
            bar_length = int(score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            print(f"Level {level}: {bar} {score:.2f} ({time:.1f}s)")
            print(f"         {row['query_type']}")
        
        return results
    
    def run_emergency_evaluation(self) -> Dict:
        """Run emergency evaluation without ground truth"""
        print("\n🚨 Emergency Evaluation (No Ground Truth)")
        print("="*60)
        
        return emergency_evaluation(self.rag_system)
    
    def generate_improvement_recommendations(self) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Check overall health
        health_check = self.results.get('health_check', {})
        if health_check.get('average_score', 0) < 0.7:
            recommendations.append(
                "🔧 Overall system performance is below threshold. "
                "Consider retraining embeddings on UAM-specific content."
            )
        
        # Check domain-specific issues
        domain_results = self.results.get('domain_results', {})
        for domain, results in domain_results.items():
            if results.get('mean_score', 0) < 0.7:
                recommendations.append(
                    f"📚 {domain}: Low performance (score: {results['mean_score']:.2f}). "
                    f"Review and expand training data for this domain."
                )
        
        # Check statistical reporting
        stat_results = self.results.get('statistical_quality', [])
        if stat_results:
            avg_stats = np.mean([r['total_stats'] for r in stat_results])
            if avg_stats < 3:
                recommendations.append(
                    "📊 Statistical reporting is weak. "
                    "Enhance statistical extraction patterns and prompts."
                )
        
        # Check complexity handling
        complexity_test = self.results.get('complexity_test')
        if complexity_test is not None and len(complexity_test) > 3:
            # Check if performance degrades with complexity
            early_score = complexity_test.iloc[:2]['overall_score'].mean()
            late_score = complexity_test.iloc[-2:]['overall_score'].mean()
            
            if late_score < early_score * 0.8:
                recommendations.append(
                    "🎯 Performance degrades significantly with query complexity. "
                    "Implement query decomposition and multi-hop retrieval."
                )
        
        return recommendations
    
    def create_evaluation_report(self, output_path: str = "uam_evaluation_report.md") -> str:
        """Create a comprehensive evaluation report"""
        report = f"""# UAM RAG System Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Overall System Health**: {self.results.get('health_check', {}).get('status', 'Unknown')}
**Average Performance Score**: {self.results.get('health_check', {}).get('average_score', 0.0):.2f}

## Detailed Results

### 1. Health Check Results
"""
        
        health_results = self.results.get('health_check', {}).get('results', [])
        for result in health_results:
            report += f"- {result['query']}: {result['score']:.2f} {result['status']}\n"
        
        report += "\n### 2. Domain-Specific Performance\n\n"
        report += "| Domain | Mean Score | Best Query | Worst Query |\n"
        report += "|--------|------------|------------|-------------|\n"
        
        domain_results = self.results.get('domain_results', {})
        for domain, metrics in domain_results.items():
            report += f"| {domain} | {metrics['mean_score']:.2f} | "
            report += f"{metrics.get('best_score', 'N/A'):.2f} | "
            report += f"{metrics.get('worst_score', 'N/A'):.2f} |\n"
        
        report += "\n### 3. Statistical Reporting Quality\n\n"
        stat_results = self.results.get('statistical_quality', [])
        if stat_results:
            total_stats = sum(r['total_stats'] for r in stat_results)
            report += f"Total statistics extracted: {total_stats}\n"
            report += f"Average per query: {total_stats/len(stat_results):.1f}\n"
        
        report += "\n### 4. Complexity Handling\n\n"
        complexity_test = self.results.get('complexity_test')
        if complexity_test is not None:
            report += "Performance vs Complexity:\n"
            for _, row in complexity_test.iterrows():
                report += f"- Level {row['complexity_level']}: {row['overall_score']:.2f}\n"
        
        report += "\n## Recommendations\n\n"
        recommendations = self.results.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += "\n## Next Steps\n\n"
        report += "1. Address critical issues identified in recommendations\n"
        report += "2. Re-run evaluation after improvements\n"
        report += "3. Compare results to track progress\n"
        report += "4. Focus on domains with lowest scores\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {output_path}")
        
        return report
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run complete evaluation workflow"""
        print("🎯 UAM RAG System Comprehensive Evaluation")
        print("="*60)
        
        # Setup
        if not self.setup_evaluation_environment():
            return {}
        
        # 1. Quick health check
        self.results['health_check'] = self.run_quick_health_check()
        
        # 2. Domain-specific evaluation
        domain_results = {}
        test_domains = [
            ResearchDomain.BEHAVIORAL_DETERMINANTS,
            ResearchDomain.TRUST_SAFETY_RISK
        ]
        
        for domain in test_domains:
            results_df = self.run_focused_domain_evaluation(domain, detailed=False)
            domain_results[domain.value] = {
                'mean_score': results_df['overall_score'].mean(),
                'best_score': results_df['overall_score'].max(),
                'worst_score': results_df['overall_score'].min(),
                'results': results_df
            }
        
        self.results['domain_results'] = domain_results
        
        # 3. Statistical reporting quality
        self.results['statistical_quality'] = self.test_statistical_reporting_quality()
        
        # 4. Progressive complexity test
        self.results['complexity_test'] = self.run_progressive_complexity_test("trust")
        
        # 5. Generate recommendations
        self.results['recommendations'] = self.generate_improvement_recommendations()
        
        # 6. Create report
        report = self.create_evaluation_report()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EVALUATION COMPLETE")
        print("="*60)
        print(f"Overall Score: {self.results['health_check']['average_score']:.2f}")
        print(f"Recommendations: {len(self.results['recommendations'])}")
        print("\nTop 3 Recommendations:")
        for rec in self.results['recommendations'][:3]:
            print(f"  • {rec}")
        
        return self.results
    
    def save_detailed_results(self, output_path: str = "evaluation_results_detailed.json"):
        """Save detailed results to JSON"""
        try:
            # Convert DataFrames to dict for JSON serialization
            json_safe_results = {}
            for key, value in self.results.items():
                if isinstance(value, pd.DataFrame):
                    json_safe_results[key] = value.to_dict('records')
                elif isinstance(value, dict):
                    json_safe_results[key] = value
                else:
                    json_safe_results[key] = str(value)
            
            with open(output_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2, default=str)
            
            print(f"📁 Detailed results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving detailed results: {e}")


def run_quick_evaluation():
    """Run a quick evaluation for basic system health"""
    print("🏃 Quick Evaluation Mode")
    print("="*30)
    
    evaluator = EvaluationOrchestrator()
    
    if evaluator.setup_evaluation_environment():
        health_results = evaluator.run_quick_health_check()
        
        print("\n📊 QUICK EVALUATION SUMMARY")
        print("="*30)
        print(f"Status: {health_results['status']}")
        print(f"Average Score: {health_results['average_score']:.2f}")
        
        return health_results
    else:
        print("❌ Quick evaluation failed - system not ready")
        return {}


def run_emergency_evaluation():
    """Run emergency evaluation without ground truth"""
    print("🚨 Emergency Evaluation Mode")
    print("="*30)
    
    evaluator = EvaluationOrchestrator()
    
    if evaluator.setup_evaluation_environment():
        return evaluator.run_emergency_evaluation()
    else:
        print("❌ Emergency evaluation failed - system not ready")
        return {}


def run_statistical_evaluation():
    """Run focused statistical reporting evaluation"""
    print("📊 Statistical Evaluation Mode")
    print("="*30)
    
    evaluator = EvaluationOrchestrator()
    
    if evaluator.setup_evaluation_environment():
        stat_results = evaluator.test_statistical_reporting_quality()
        
        print("\n📈 STATISTICAL EVALUATION SUMMARY")
        print("="*30)
        total_stats = sum(r['total_stats'] for r in stat_results)
        print(f"Total Statistics Found: {total_stats}")
        print(f"Average per Query: {total_stats/len(stat_results):.1f}")
        
        return stat_results
    else:
        print("❌ Statistical evaluation failed - system not ready")
        return {}


def run_domain_evaluation(domain_name: str):
    """Run evaluation for a specific domain"""
    print(f"🎯 Domain Evaluation: {domain_name}")
    print("="*40)
    
    # Map domain name to enum
    domain_mapping = {
        'behavioral': ResearchDomain.BEHAVIORAL_DETERMINANTS,
        'trust': ResearchDomain.TRUST_SAFETY_RISK,
        'affect': ResearchDomain.AFFECT_EMOTION,
        'demographics': ResearchDomain.DEMOGRAPHICS
    }
    
    domain = domain_mapping.get(domain_name.lower())
    if not domain:
        print(f"❌ Unknown domain: {domain_name}")
        print(f"Available domains: {list(domain_mapping.keys())}")
        return {}
    
    evaluator = EvaluationOrchestrator()
    
    if evaluator.setup_evaluation_environment():
        results = evaluator.run_focused_domain_evaluation(domain, detailed=True)
        
        print(f"\n📊 DOMAIN EVALUATION SUMMARY: {domain_name}")
        print("="*40)
        print(f"Mean Score: {results['overall_score'].mean():.2f}")
        print(f"Best Score: {results['overall_score'].max():.2f}")
        print(f"Worst Score: {results['overall_score'].min():.2f}")
        
        return results
    else:
        print("❌ Domain evaluation failed - system not ready")
        return {}


def main():
    """Main evaluation entry point with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UAM RAG System Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                           # Run comprehensive evaluation
  python eval.py --mode quick              # Quick health check
  python eval.py --mode emergency          # Emergency evaluation
  python eval.py --mode statistical        # Statistical reporting test
  python eval.py --mode domain --domain trust  # Domain-specific evaluation
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['comprehensive', 'quick', 'emergency', 'statistical', 'domain'],
        default='comprehensive',
        help='Evaluation mode to run'
    )
    
    parser.add_argument(
        '--domain',
        choices=['behavioral', 'trust', 'affect', 'demographics'],
        help='Domain to evaluate (for domain mode)'
    )
    
    parser.add_argument(
        '--output',
        default='uam_evaluation_report.md',
        help='Output file for evaluation report'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save detailed results as JSON'
    )
    
    args = parser.parse_args()
    
    # Run appropriate evaluation mode
    if args.mode == 'comprehensive':
        evaluator = EvaluationOrchestrator()
        results = evaluator.run_comprehensive_evaluation()
        
        if args.save_json:
            evaluator.save_detailed_results()
    
    elif args.mode == 'quick':
        results = run_quick_evaluation()
    
    elif args.mode == 'emergency':
        results = run_emergency_evaluation()
    
    elif args.mode == 'statistical':
        results = run_statistical_evaluation()
    
    elif args.mode == 'domain':
        if not args.domain:
            print("❌ Domain mode requires --domain argument")
            return
        results = run_domain_evaluation(args.domain)
    
    print("\n✅ Evaluation complete!")
    
    if args.mode == 'comprehensive':
        print(f"📄 Check '{args.output}' for detailed report")


if __name__ == "__main__":
    main()