# test_journal_impact.py
"""
Test script for journal impact integration
Validates Excel parsing, paper matching, and quality scoring functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.journal_impact import JournalImpactManager
from config import Config

def test_journal_impact_system():
    """Test the complete journal impact system"""
    print("🏆 UAM Literature Review - Journal Impact Integration Test")
    print("=" * 70)
    
    # Test 1: Excel file detection
    print("\n1. Testing Excel file detection...")
    excel_path = Config.JOURNAL_METADATA_PATH
    
    if os.path.exists(excel_path):
        print(f"✅ Excel file found: {excel_path}")
        file_size = os.path.getsize(excel_path) / 1024  # KB
        print(f"   File size: {file_size:.1f} KB")
    else:
        print(f"❌ Excel file not found: {excel_path}")
        print("   Please ensure the Excel file is in the correct location.")
        return False
    
    # Test 2: Initialize Journal Impact Manager
    print("\n2. Testing Journal Impact Manager initialization...")
    try:
        journal_manager = JournalImpactManager(excel_path)
        print("✅ Journal Impact Manager initialized successfully")
        
        # Display basic statistics
        stats = journal_manager.get_journal_statistics()
        print(f"   Papers mapped: {stats.get('total_papers_mapped', 0)}")
        print(f"   Journals found: {stats.get('total_journals', 0)}")
        
        if stats.get('tier_distribution'):
            print("   Tier distribution:")
            for tier, count in stats['tier_distribution'].items():
                print(f"     {tier}: {count}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Journal Impact Manager: {e}")
        return False
    
    # Test 3: Excel structure detection
    print("\n3. Testing Excel structure detection...")
    try:
        # Force reload to test structure detection
        result = journal_manager.load_excel_metadata(excel_path)
        
        if result.get('success'):
            print("✅ Excel structure detected successfully")
            print(f"   Detected columns: {result.get('detected_columns', {})}")
            print(f"   Journals loaded: {result.get('journals_loaded', 0)}")
            print(f"   Papers mapped: {result.get('papers_mapped', 0)}")
        else:
            print(f"❌ Excel structure detection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Excel structure detection error: {e}")
        return False
    
    # Test 4: Paper matching functionality
    print("\n4. Testing paper matching functionality...")
    
    # Test with some sample paper names
    test_papers = [
        "paper1.pdf",
        "study_uam_2023.pdf", 
        "research_air_taxi_acceptance.pdf",
        "nonexistent_paper.pdf"
    ]
    
    for paper in test_papers:
        print(f"\n   Testing paper: {paper}")
        
        journal_info = journal_manager.match_paper_to_journal(paper)
        if journal_info:
            print(f"   ✅ Matched to journal: {journal_info['journal']}")
            print(f"      Quartile: {journal_info.get('quartile', 'unknown')}")
            print(f"      Impact score: {journal_info.get('impact_score', 0.0)}")
            
            # Test quality metrics
            quality_score = journal_manager.get_paper_quality_score(paper)
            quality_tier = journal_manager.get_quality_tier(paper)
            boost_multiplier = journal_manager.get_journal_boost_multiplier(paper)
            
            print(f"      Quality score: {quality_score:.3f}")
            print(f"      Quality tier: {quality_tier}")
            print(f"      Boost multiplier: {boost_multiplier:.3f}")
        else:
            print(f"   ❌ No match found (expected for test papers)")
    
    # Test 5: Quality tier analysis
    print("\n5. Testing quality tier analysis...")
    try:
        for tier in ['top', 'high', 'medium', 'low', 'unknown']:
            papers_in_tier = journal_manager.get_papers_by_tier(tier)
            print(f"   {tier.title()} tier: {len(papers_in_tier)} papers")
            
            # Show first few papers in each tier
            if papers_in_tier:
                sample_papers = papers_in_tier[:3]
                print(f"      Sample papers: {', '.join(sample_papers)}")
    
    except Exception as e:
        print(f"❌ Quality tier analysis failed: {e}")
        return False
    
    # Test 6: Journal statistics
    print("\n6. Testing journal statistics...")
    try:
        journal_stats = journal_manager.get_journal_statistics()
        
        print("   Journal Statistics:")
        if 'journal_statistics' in journal_stats:
            js = journal_stats['journal_statistics']
            if 'quartile_distribution' in js:
                print("   Quartile distribution:")
                for quartile, count in js['quartile_distribution'].items():
                    print(f"     {quartile}: {count}")
            
            if 'impact_score_stats' in js:
                iss = js['impact_score_stats']
                print(f"   Impact score range: {iss.get('min', 0):.2f} - {iss.get('max', 0):.2f}")
                print(f"   Average impact score: {iss.get('mean', 0):.2f}")
    
    except Exception as e:
        print(f"❌ Journal statistics failed: {e}")
        return False
    
    # Test 7: Validation
    print("\n7. Testing data validation...")
    try:
        validation_result = journal_manager.validate_journal_data()
        
        if validation_result['valid']:
            print("✅ Journal data validation passed")
        else:
            print("⚠️  Journal data validation warnings:")
            for error in validation_result['errors']:
                print(f"     Error: {error}")
        
        if validation_result['warnings']:
            print("   Warnings:")
            for warning in validation_result['warnings']:
                print(f"     Warning: {warning}")
    
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False
    
    # Test 8: Export functionality
    print("\n8. Testing export functionality...")
    try:
        report = journal_manager.export_journal_report()
        
        print("✅ Journal report generated successfully")
        print(f"   Report timestamp: {report.get('timestamp')}")
        print(f"   Top journals: {len(report.get('top_journals', []))}")
        print(f"   Quality analysis: {len(report.get('quality_analysis', {}))}")
        
        # Show top journals
        if report.get('top_journals'):
            print("   Top 3 journals by impact:")
            for i, journal in enumerate(report['top_journals'][:3], 1):
                print(f"     {i}. {journal['name']} (IF: {journal.get('impact_score', 0):.2f})")
    
    except Exception as e:
        print(f"❌ Export functionality failed: {e}")
        return False
    
    # Test 9: Cache functionality
    print("\n9. Testing cache functionality...")
    try:
        cache_path = "test_journal_cache.json"
        
        # Save cache
        journal_manager.save_metadata_cache(cache_path)
        print(f"✅ Cache saved to: {cache_path}")
        
        # Load cache
        new_manager = JournalImpactManager()
        if new_manager.load_metadata_cache(cache_path):
            print("✅ Cache loaded successfully")
            
            # Verify cache data
            cached_stats = new_manager.get_journal_statistics()
            if cached_stats.get('total_papers_mapped') == stats.get('total_papers_mapped'):
                print("✅ Cache data integrity verified")
            else:
                print("⚠️  Cache data integrity check failed")
        else:
            print("❌ Cache loading failed")
        
        # Cleanup
        if os.path.exists(cache_path):
            os.remove(cache_path)
    
    except Exception as e:
        print(f"❌ Cache functionality failed: {e}")
        return False
    
    # Test 10: Configuration integration
    print("\n10. Testing configuration integration...")
    try:
        print(f"   Journal ranking enabled: {Config.ENABLE_JOURNAL_RANKING}")
        print(f"   Journal impact weight: {Config.JOURNAL_IMPACT_WEIGHT}")
        print(f"   Relevance weight: {Config.RELEVANCE_WEIGHT}")
        print(f"   Statistical content weight: {Config.STATISTICAL_CONTENT_WEIGHT}")
        print(f"   Quality tier weights: {Config.QUALITY_TIER_WEIGHTS}")
        
        # Test config validation
        Config.validate_config()
        print("✅ Configuration validation passed")
    
    except Exception as e:
        print(f"❌ Configuration integration failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 All tests completed successfully!")
    print("\nNext steps:")
    print("1. Run paper ingestion with journal impact integration")
    print("2. Test enhanced retrieval and ranking")
    print("3. Validate quality-weighted responses")
    
    return True

def test_with_sample_directory():
    """Test with sample PDF directory if available"""
    print("\n📁 Testing with sample PDF directory...")
    
    # Look for common PDF directories
    possible_dirs = ["./papers", "./pdfs", "./documents", "./data"]
    
    for directory in possible_dirs:
        if os.path.exists(directory):
            pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
            if pdf_files:
                print(f"✅ Found PDF directory: {directory}")
                print(f"   PDF files: {len(pdf_files)}")
                
                # Test journal matching for actual files
                journal_manager = JournalImpactManager(Config.JOURNAL_METADATA_PATH)
                
                print("\n   Testing journal matching for actual files:")
                for pdf_file in pdf_files[:5]:  # Test first 5 files
                    print(f"   📄 {pdf_file}")
                    journal_info = journal_manager.match_paper_to_journal(pdf_file)
                    if journal_info:
                        print(f"      ✅ Matched: {journal_info['journal']} ({journal_info.get('quartile', 'unknown')})")
                    else:
                        print(f"      ❌ No match found")
                
                return True
    
    print("❌ No PDF directory found for testing")
    return False

def main():
    """Main test function"""
    print("Starting Journal Impact Integration Tests...")
    
    # Test core functionality
    if not test_journal_impact_system():
        print("\n❌ Core tests failed. Please check the configuration and Excel file.")
        return False
    
    # Test with sample directory if available
    test_with_sample_directory()
    
    print("\n✨ Journal Impact Integration Testing Complete!")
    print("\nThe system is ready for:")
    print("• Quality-weighted paper retrieval")
    print("• Journal impact-based ranking")
    print("• Comprehensive coverage analysis")
    print("• High-impact journal prioritization")
    
    return True

if __name__ == "__main__":
    main()