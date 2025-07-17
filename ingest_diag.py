# test_ingestion.py
"""
Test script for the integrated ingestion system
"""

import os
import sys
import logging

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.rag_system import UAMRAGSystem

def test_ingestion():
    """Test the integrated ingestion system"""
    
    print("🎓 UAM Literature Review - Integrated Ingestion Test")
    print("=" * 60)
    
    # Initialize system
    print("\n1. Initializing UAM RAG System...")
    try:
        rag_system = UAMRAGSystem()
        print(f"✅ System initialized: {rag_system}")
        
        # Check system status
        status = rag_system.get_system_status()
        print(f"   Components: {status['components']}")
        print(f"   Multimodal available: {status['components']['multimodal_available']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Get PDF directory
    print("\n2. Getting PDF directory...")
    pdf_directory = input("Enter path to PDF directory (or press Enter for './papers'): ").strip()
    if not pdf_directory:
        pdf_directory = "./papers"
    
    if not os.path.exists(pdf_directory):
        print(f"❌ Directory not found: {pdf_directory}")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_directory}")
        return
    
    print(f"✅ Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    # Ask for processing mode
    print("\n3. Choose processing mode:")
    print("   1. Automatic (multimodal if available, text-only fallback)")
    print("   2. Force text-only processing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    force_text_only = choice == "2"
    
    # Test single PDF diagnosis first
    print("\n4. Diagnosing sample PDF...")
    sample_pdf = os.path.join(pdf_directory, pdf_files[0])
    diagnosis = rag_system.diagnose_extraction(sample_pdf)
    
    print(f"   File: {diagnosis['filename']}")
    print(f"   Success: {diagnosis['success']}")
    
    if diagnosis['success']:
        text_info = diagnosis['text_extraction']
        print(f"   Text extraction: {text_info['pages']} pages, {text_info['total_text_length']} chars")
        
        if diagnosis['multimodal_extraction']:
            mm_info = diagnosis['multimodal_extraction']
            if mm_info['success']:
                print(f"   Multimodal extraction: {mm_info['figures']} figures, {mm_info['tables']} tables")
            else:
                print(f"   Multimodal extraction failed: {mm_info['error']}")
        else:
            print("   Multimodal extraction: Not available")
    else:
        print(f"   Error: {diagnosis['error']}")
        return
    
    # Confirm ingestion
    print(f"\n5. Ready to ingest {len(pdf_files)} papers")
    mode = "text-only" if force_text_only else "automatic (multimodal if available)"
    print(f"   Processing mode: {mode}")
    
    confirm = input("Proceed with ingestion? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Ingestion cancelled")
        return
    
    # Perform ingestion
    print("\n6. Starting ingestion...")
    try:
        stats = rag_system.ingest_papers(pdf_directory, force_text_only=force_text_only)
        
        print(f"✅ Ingestion completed!")
        print(f"   Processing mode: {stats['processing_mode']}")
        print(f"   Total papers: {stats['total_papers']}")
        print(f"   Successful papers: {stats['successful_papers']}")
        print(f"   Failed papers: {stats['failed_papers']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        
        if stats['processing_mode'] == 'multimodal':
            print(f"   Text chunks: {stats['text_chunks']}")
            print(f"   Figure chunks: {stats['figure_chunks']}")
            print(f"   Table chunks: {stats['table_chunks']}")
        
        if stats['errors']:
            print(f"   Errors: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # Show first 3 errors
                print(f"     - {error['filename']}: {error['error']}")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return
    
    # Test retrieval
    print("\n7. Testing retrieval...")
    try:
        test_queries = [
            "What are the main determinants of UAM adoption?",
            "How does trust influence UAM acceptance?",
            "What role do safety perceptions play?"
        ]
        
        for query in test_queries:
            print(f"   Query: {query}")
            try:
                answer, sources = rag_system.answer_literature_query(query)
                print(f"   ✅ Generated answer ({len(sources)} sources)")
                print(f"   Answer preview: {answer[:150]}...")
                print(f"   Sources: {', '.join(sources[:3])}")
                print()
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
                print()
    
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
    
    # Show corpus statistics
    print("\n8. Corpus statistics:")
    try:
        corpus_stats = rag_system.get_corpus_statistics()
        if corpus_stats:
            print(f"   Total papers: {corpus_stats.get('total_papers', 0)}")
            print(f"   Total chunks: {corpus_stats.get('total_chunks', 0)}")
            print(f"   Processing mode: {corpus_stats.get('configuration', {}).get('processing_mode', 'unknown')}")
            
            # Show top sections
            section_dist = corpus_stats.get('section_distribution', {})
            if section_dist:
                print("   Top sections:")
                for section, count in sorted(section_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     - {section}: {count}")
        else:
            print("   No corpus statistics available")
    
    except Exception as e:
        print(f"❌ Could not get corpus statistics: {e}")
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    test_ingestion()