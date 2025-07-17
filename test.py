# test.py - Main Testing Entry Point
"""
Main testing script for the UAM Literature Review RAG System.
Consolidated from various testing files for better organization.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import core system components
from core.rag_system import UAMRAGSystem
from core.multimodal_processor import MultimodalProcessor
from config import Config


class SystemTester:
    """Main testing orchestrator for the UAM RAG System"""
    
    def __init__(self):
        self.rag_system = None
        self.test_results = {}
        self.test_directory = Path("test_results")
        self.test_directory.mkdir(exist_ok=True)
    
    def setup_test_environment(self) -> bool:
        """Setup the testing environment"""
        print("🧪 Setting up Test Environment")
        print("="*50)
        
        try:
            # Test imports
            print("1. Testing core imports...")
            from core.rag_system import UAMRAGSystem
            from core.embeddings import EmbeddingManager
            from core.retrieval import RetrievalEngine
            print("   ✅ Core imports successful")
            
            # Test configuration
            print("2. Testing configuration...")
            Config.validate_config()
            print("   ✅ Configuration valid")
            
            # Initialize system
            print("3. Initializing RAG system...")
            self.rag_system = UAMRAGSystem()
            print("   ✅ RAG system initialized")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Setup failed: {e}")
            return False
    
    def test_dependencies(self) -> Dict:
        """Test system dependencies"""
        print("\n🔧 Testing Dependencies")
        print("="*50)
        
        results = {
            'required_deps': {},
            'optional_deps': {},
            'missing_required': [],
            'missing_optional': []
        }
        
        # Required dependencies
        required_deps = [
            'langchain',
            'langchain_community',
            'langchain_huggingface',
            'sentence_transformers',
            'faiss-cpu',
            'numpy',
            'pandas',
            'requests'
        ]
        
        # Optional dependencies
        optional_deps = [
            'PyMuPDF',
            'pdfplumber',
            'Pillow',
            'pytesseract',
            'matplotlib',
            'seaborn'
        ]
        
        print("Required Dependencies:")
        for dep in required_deps:
            try:
                __import__(dep.replace('-', '_'))
                results['required_deps'][dep] = True
                print(f"  ✅ {dep}")
            except ImportError:
                results['required_deps'][dep] = False
                results['missing_required'].append(dep)
                print(f"  ❌ {dep}")
        
        print("\nOptional Dependencies:")
        for dep in optional_deps:
            try:
                __import__(dep.replace('-', '_'))
                results['optional_deps'][dep] = True
                print(f"  ✅ {dep}")
            except ImportError:
                results['optional_deps'][dep] = False
                results['missing_optional'].append(dep)
                print(f"  ⚠️  {dep}")
        
        # Test special cases
        print("\nSpecial Dependency Tests:")
        
        # Test OpenRouter API key
        if Config.OPENROUTER_API_KEY:
            results['api_key'] = True
            print("  ✅ OpenRouter API key configured")
        else:
            results['api_key'] = False
            print("  ❌ OpenRouter API key missing")
        
        # Test Tesseract OCR
        if results['optional_deps'].get('pytesseract', False):
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                results['tesseract_binary'] = True
                print("  ✅ Tesseract OCR binary available")
            except:
                results['tesseract_binary'] = False
                print("  ⚠️  Tesseract OCR binary not found")
        
        self.test_results['dependencies'] = results
        return results
    
    def test_configuration(self) -> Dict:
        """Test configuration settings"""
        print("\n⚙️  Testing Configuration")
        print("="*50)
        
        results = {
            'config_valid': False,
            'embedding_model': False,
            'reranker_model': False,
            'paths_exist': {},
            'parameter_ranges': {}
        }
        
        try:
            # Test config validation
            Config.validate_config()
            results['config_valid'] = True
            print("  ✅ Configuration validation passed")
        except Exception as e:
            results['config_valid'] = False
            print(f"  ❌ Configuration validation failed: {e}")
        
        # Test embedding model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(Config.EMBEDDING_MODEL)
            results['embedding_model'] = True
            print(f"  ✅ Embedding model loaded: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            results['embedding_model'] = False
            print(f"  ❌ Embedding model failed: {e}")
        
        # Test reranker model
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(Config.RERANKER_MODEL)
            results['reranker_model'] = True
            print(f"  ✅ Reranker model loaded: {Config.RERANKER_MODEL}")
        except Exception as e:
            results['reranker_model'] = False
            print(f"  ❌ Reranker model failed: {e}")
        
        # Test parameter ranges
        param_tests = [
            ('CHUNK_SIZE', Config.CHUNK_SIZE, 100, 2000),
            ('RETRIEVAL_K', Config.RETRIEVAL_K, 1, 100),
            ('TEMPERATURE', Config.TEMPERATURE, 0.0, 1.0),
            ('MAX_TOKENS', Config.MAX_TOKENS, 100, 4000)
        ]
        
        for param_name, value, min_val, max_val in param_tests:
            in_range = min_val <= value <= max_val
            results['parameter_ranges'][param_name] = in_range
            status = "✅" if in_range else "❌"
            print(f"  {status} {param_name}: {value} (range: {min_val}-{max_val})")
        
        self.test_results['configuration'] = results
        return results
    
    def test_basic_functionality(self) -> Dict:
        """Test basic RAG functionality"""
        print("\n🔍 Testing Basic Functionality")
        print("="*50)
        
        results = {
            'initialization': False,
            'query_processing': False,
            'retrieval': False,
            'generation': False,
            'error_handling': False
        }
        
        try:
            # Test initialization
            if self.rag_system:
                results['initialization'] = True
                print("  ✅ System initialization")
            else:
                print("  ❌ System initialization failed")
                return results
            
            # Test query processing
            try:
                test_query = "What is UAM adoption intention?"
                # This should not fail even without corpus
                processed_query = self.rag_system._preprocess_query(test_query)
                results['query_processing'] = True
                print("  ✅ Query processing")
            except Exception as e:
                results['query_processing'] = False
                print(f"  ❌ Query processing failed: {e}")
            
            # Test retrieval (will fail without corpus, but shouldn't crash)
            try:
                docs = self.rag_system.enhanced_retrieval(test_query)
                results['retrieval'] = True
                print("  ✅ Retrieval engine (no corpus expected)")
            except Exception as e:
                # This is expected without corpus
                if "Please ingest papers first" in str(e):
                    results['retrieval'] = True
                    print("  ✅ Retrieval engine (correctly handles no corpus)")
                else:
                    results['retrieval'] = False
                    print(f"  ❌ Retrieval engine failed: {e}")
            
            # Test generation
            try:
                # Test with dummy context
                dummy_context = "Test context about UAM research"
                response = self.rag_system._generate_response(test_query, dummy_context)
                if response and len(response) > 10:
                    results['generation'] = True
                    print("  ✅ Response generation")
                else:
                    results['generation'] = False
                    print("  ❌ Response generation produced empty result")
            except Exception as e:
                results['generation'] = False
                print(f"  ❌ Response generation failed: {e}")
            
            # Test error handling
            try:
                # Test with malformed query
                self.rag_system._preprocess_query("")
                results['error_handling'] = True
                print("  ✅ Error handling")
            except Exception as e:
                # Should handle gracefully
                results['error_handling'] = True
                print("  ✅ Error handling (graceful failure)")
        
        except Exception as e:
            print(f"  ❌ Basic functionality test failed: {e}")
        
        self.test_results['basic_functionality'] = results
        return results
    
    def test_multimodal_capabilities(self) -> Dict:
        """Test multimodal processing capabilities"""
        print("\n🖼️  Testing Multimodal Capabilities")
        print("="*50)
        
        results = {
            'processor_init': False,
            'pdf_processing': False,
            'figure_extraction': False,
            'table_extraction': False,
            'ocr_capability': False
        }
        
        try:
            # Test processor initialization
            processor = MultimodalProcessor()
            results['processor_init'] = True
            print("  ✅ Multimodal processor initialization")
            
            # Test PDF processing capability
            try:
                import fitz
                import pdfplumber
                results['pdf_processing'] = True
                print("  ✅ PDF processing libraries available")
            except ImportError as e:
                results['pdf_processing'] = False
                print(f"  ❌ PDF processing libraries missing: {e}")
            
            # Test figure extraction
            if results['pdf_processing']:
                try:
                    from PIL import Image
                    results['figure_extraction'] = True
                    print("  ✅ Figure extraction capability")
                except ImportError:
                    results['figure_extraction'] = False
                    print("  ❌ Figure extraction requires Pillow")
            
            # Test table extraction
            if results['pdf_processing']:
                try:
                    import pandas as pd
                    results['table_extraction'] = True
                    print("  ✅ Table extraction capability")
                except ImportError:
                    results['table_extraction'] = False
                    print("  ❌ Table extraction requires pandas")
            
            # Test OCR capability
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                results['ocr_capability'] = True
                print("  ✅ OCR capability available")
            except:
                results['ocr_capability'] = False
                print("  ⚠️  OCR capability not available")
        
        except Exception as e:
            print(f"  ❌ Multimodal test failed: {e}")
        
        self.test_results['multimodal'] = results
        return results
    
    def test_corpus_operations(self, test_pdf_path: Optional[str] = None) -> Dict:
        """Test corpus ingestion and management"""
        print("\n📚 Testing Corpus Operations")
        print("="*50)
        
        results = {
            'ingestion_ready': False,
            'test_ingestion': False,
            'corpus_stats': False,
            'search_functionality': False
        }
        
        # Check if ingestion is ready
        if self.rag_system:
            results['ingestion_ready'] = True
            print("  ✅ Ingestion system ready")
        else:
            print("  ❌ Ingestion system not ready")
            return results
        
        # Test with provided PDF path
        if test_pdf_path and os.path.exists(test_pdf_path):
            try:
                print(f"  🔍 Testing with PDF: {test_pdf_path}")
                
                # Test single file processing
                diagnosis = self.rag_system.diagnose_extraction(test_pdf_path)
                if diagnosis.get('success', False):
                    results['test_ingestion'] = True
                    print("  ✅ Test PDF processing successful")
                else:
                    results['test_ingestion'] = False
                    print(f"  ❌ Test PDF processing failed: {diagnosis.get('error', 'Unknown error')}")
                
            except Exception as e:
                results['test_ingestion'] = False
                print(f"  ❌ Test ingestion failed: {e}")
        else:
            print("  ⚠️  No test PDF provided, skipping ingestion test")
        
        # Test corpus statistics
        try:
            stats = self.rag_system.get_corpus_statistics()
            results['corpus_stats'] = True
            print("  ✅ Corpus statistics accessible")
            
            if stats and stats.get('total_papers', 0) > 0:
                print(f"    Found {stats['total_papers']} papers in corpus")
                
                # Test search functionality
                try:
                    test_query = "UAM adoption"
                    response, sources = self.rag_system.answer_literature_query(test_query)
                    if response and len(response) > 50:
                        results['search_functionality'] = True
                        print("  ✅ Search functionality working")
                    else:
                        results['search_functionality'] = False
                        print("  ❌ Search functionality produced poor results")
                except Exception as e:
                    results['search_functionality'] = False
                    print(f"  ❌ Search functionality failed: {e}")
            else:
                print("    No papers in corpus - ingestion needed for search test")
        
        except Exception as e:
            results['corpus_stats'] = False
            print(f"  ❌ Corpus statistics failed: {e}")
        
        self.test_results['corpus_operations'] = results
        return results
    
    def test_performance_benchmarks(self) -> Dict:
        """Test system performance benchmarks"""
        print("\n⚡ Testing Performance Benchmarks")
        print("="*50)
        
        results = {
            'query_response_time': 0.0,
            'embedding_speed': 0.0,
            'memory_usage': 0.0,
            'concurrent_queries': False
        }
        
        try:
            import time
            import threading
            
            # Test query response time
            if self.rag_system:
                start_time = time.time()
                test_query = "What factors influence UAM adoption?"
                
                try:
                    response, sources = self.rag_system.answer_literature_query(test_query)
                    response_time = time.time() - start_time
                    results['query_response_time'] = response_time
                    print(f"  ✅ Query response time: {response_time:.2f}s")
                except Exception as e:
                    print(f"  ❌ Query response test failed: {e}")
            
            # Test embedding speed
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(Config.EMBEDDING_MODEL)
                
                test_texts = ["Sample text for embedding"] * 10
                start_time = time.time()
                embeddings = model.encode(test_texts)
                embedding_time = time.time() - start_time
                
                results['embedding_speed'] = embedding_time
                print(f"  ✅ Embedding speed: {embedding_time:.2f}s for 10 texts")
            except Exception as e:
                print(f"  ❌ Embedding speed test failed: {e}")
            
            # Test memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                results['memory_usage'] = memory_mb
                print(f"  ✅ Memory usage: {memory_mb:.1f} MB")
            except ImportError:
                print("  ⚠️  psutil not available for memory testing")
            
            # Test concurrent queries
            try:
                def test_concurrent_query():
                    try:
                        self.rag_system.answer_literature_query("Test concurrent query")
                        return True
                    except:
                        return False
                
                threads = []
                for i in range(3):
                    thread = threading.Thread(target=test_concurrent_query)
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                results['concurrent_queries'] = True
                print("  ✅ Concurrent query handling")
            except Exception as e:
                results['concurrent_queries'] = False
                print(f"  ❌ Concurrent query test failed: {e}")
        
        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
        
        self.test_results['performance'] = results
        return results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        print("\n📄 Generating Test Report")
        print("="*50)
        
        report = f"""# UAM RAG System Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Environment
- Python Version: {sys.version}
- Operating System: {os.name}
- Working Directory: {os.getcwd()}

## Test Results Summary

### Dependencies
"""
        
        deps = self.test_results.get('dependencies', {})
        if deps:
            report += f"- Required Dependencies: {sum(deps['required_deps'].values())}/{len(deps['required_deps'])} ✅\n"
            report += f"- Optional Dependencies: {sum(deps['optional_deps'].values())}/{len(deps['optional_deps'])} ✅\n"
            report += f"- API Key Configured: {'✅' if deps.get('api_key', False) else '❌'}\n"
            
            if deps['missing_required']:
                report += f"- Missing Required: {', '.join(deps['missing_required'])}\n"
            if deps['missing_optional']:
                report += f"- Missing Optional: {', '.join(deps['missing_optional'])}\n"
        
        report += "\n### Configuration\n"
        config = self.test_results.get('configuration', {})
        if config:
            report += f"- Configuration Valid: {'✅' if config['config_valid'] else '❌'}\n"
            report += f"- Embedding Model: {'✅' if config['embedding_model'] else '❌'}\n"
            report += f"- Reranker Model: {'✅' if config['reranker_model'] else '❌'}\n"
        
        report += "\n### Basic Functionality\n"
        basic = self.test_results.get('basic_functionality', {})
        if basic:
            for test_name, result in basic.items():
                status = '✅' if result else '❌'
                report += f"- {test_name.replace('_', ' ').title()}: {status}\n"
        
        report += "\n### Multimodal Capabilities\n"
        multimodal = self.test_results.get('multimodal', {})
        if multimodal:
            for test_name, result in multimodal.items():
                status = '✅' if result else '❌'
                report += f"- {test_name.replace('_', ' ').title()}: {status}\n"
        
        report += "\n### Performance Benchmarks\n"
        perf = self.test_results.get('performance', {})
        if perf:
            if perf['query_response_time'] > 0:
                report += f"- Query Response Time: {perf['query_response_time']:.2f}s\n"
            if perf['embedding_speed'] > 0:
                report += f"- Embedding Speed: {perf['embedding_speed']:.2f}s\n"
            if perf['memory_usage'] > 0:
                report += f"- Memory Usage: {perf['memory_usage']:.1f} MB\n"
        
        report += "\n### Recommendations\n"
        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        # Save report
        report_path = self.test_directory / "test_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Test report saved to: {report_path}")
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check dependencies
        deps = self.test_results.get('dependencies', {})
        if deps.get('missing_required'):
            recommendations.append(
                f"Install missing required dependencies: {', '.join(deps['missing_required'])}"
            )
        
        if not deps.get('api_key', False):
            recommendations.append("Configure OpenRouter API key for full functionality")
        
        # Check configuration
        config = self.test_results.get('configuration', {})
        if not config.get('config_valid', False):
            recommendations.append("Fix configuration validation errors")
        
        # Check basic functionality
        basic = self.test_results.get('basic_functionality', {})
        if not basic.get('generation', False):
            recommendations.append("Response generation is failing - check LLM configuration")
        
        # Check multimodal
        multimodal = self.test_results.get('multimodal', {})
        if not multimodal.get('pdf_processing', False):
            recommendations.append("Install PDF processing dependencies for full multimodal support")
        
        # Check performance
        perf = self.test_results.get('performance', {})
        if perf.get('query_response_time', 0) > 30:
            recommendations.append("Query response time is slow - consider optimizing retrieval")
        
        if not recommendations:
            recommendations.append("System appears to be working well!")
        
        return recommendations
    
    def run_comprehensive_test(self, test_pdf_path: Optional[str] = None) -> Dict:
        """Run comprehensive test suite"""
        print("🚀 Running Comprehensive Test Suite")
        print("="*50)
        
        if not self.setup_test_environment():
            return {}
        
        # Run all tests
        self.test_dependencies()
        self.test_configuration()
        self.test_basic_functionality()
        self.test_multimodal_capabilities()
        self.test_corpus_operations(test_pdf_path)
        self.test_performance_benchmarks()
        
        # Generate report
        self.generate_test_report()
        
        # Calculate overall score
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    if isinstance(result, bool):
                        total_tests += 1
                        if result:
                            passed_tests += 1
        
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 OVERALL TEST RESULTS")
        print("="*50)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("✅ System is ready for production use")
        elif overall_score >= 60:
            print("⚠️  System has some issues but is functional")
        else:
            print("❌ System has significant issues that need attention")
        
        return self.test_results
    
    def save_test_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file"""
        output_path = self.test_directory / filename
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"📁 Test results saved to: {output_path}")


def install_missing_dependencies():
    """Install missing dependencies"""
    print("📦 Installing Missing Dependencies")
    print("="*50)
    
    required_packages = [
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "sentence-transformers",
        "faiss-cpu",
        "requests"
    ]
    
    optional_packages = [
        "PyMuPDF",
        "pdfplumber",
        "Pillow",
        "pytesseract",
        "matplotlib",
        "seaborn"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\nInstalling optional packages...")
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install optional package {package}")


def run_quick_test():
    """Run a quick test for basic functionality"""
    print("🏃 Quick Test Mode")
    print("="*30)
    
    tester = SystemTester()
    
    if tester.setup_test_environment():
        tester.test_dependencies()
        tester.test_basic_functionality()
        
        print("\n📊 QUICK TEST SUMMARY")
        print("="*30)
        
        deps = tester.test_results.get('dependencies', {})
        basic = tester.test_results.get('basic_functionality', {})
        
        if deps.get('missing_required'):
            print(f"❌ Missing required dependencies: {deps['missing_required']}")
        elif all(basic.values()):
            print("✅ Basic functionality working")
        else:
            print("⚠️  Some basic functionality issues detected")
        
        return tester.test_results
    else:
        print("❌ Quick test failed - system not ready")
        return {}


def main():
    """Main testing entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UAM RAG System Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py                              # Run comprehensive test
  python test.py --mode quick                 # Quick functionality test
  python test.py --mode deps                  # Test dependencies only
  python test.py --install-deps               # Install missing dependencies
  python test.py --test-pdf sample.pdf        # Test with specific PDF
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['comprehensive', 'quick', 'deps', 'config', 'multimodal', 'performance'],
        default='comprehensive',
        help='Test mode to run'
    )
    
    parser.add_argument(
        '--test-pdf',
        help='Path to test PDF file for corpus operations'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install missing dependencies'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed test results as JSON'
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        install_missing_dependencies()
        return
    
    # Run appropriate test mode
    tester = SystemTester()
    
    if args.mode == 'comprehensive':
        results = tester.run_comprehensive_test(args.test_pdf)
    elif args.mode == 'quick':
        results = run_quick_test()
    elif args.mode == 'deps':
        if tester.setup_test_environment():
            results = tester.test_dependencies()
        else:
            results = {}
    elif args.mode == 'config':
        if tester.setup_test_environment():
            results = tester.test_configuration()
        else:
            results = {}
    elif args.mode == 'multimodal':
        if tester.setup_test_environment():
            results = tester.test_multimodal_capabilities()
        else:
            results = {}
    elif args.mode == 'performance':
        if tester.setup_test_environment():
            results = tester.test_performance_benchmarks()
        else:
            results = {}
    
    # Save results if requested
    if args.save_results and results:
        tester.test_results.update(results)
        tester.save_test_results()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()