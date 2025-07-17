# core/generation.py
"""
Response Generation Module
Generates literature review responses from retrieved documents.
"""

import logging
from typing import List, Optional
from langchain.schema import Document

from .llm_client import OpenRouterLLM
from config import Config

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates literature review responses"""
    
    def __init__(self):
        """Initialize response generator"""
        self.llm = OpenRouterLLM(Config.OPENROUTER_API_KEY)
        logger.info("Response generator initialized")
    
    def generate_response(self, query: str, documents: List[Document], chapter_topic: str = None) -> str:
        """Generate a literature review response"""
        if not documents:
            return "No relevant documents found for this query."
        
        # Format context
        context = self._format_context(documents)
        
        # Generate response
        response = self._generate_with_context(query, context, chapter_topic)
        
        return response
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context for generation"""
        context_parts = []
        
        # Group documents by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Format each source
        for source, source_docs in docs_by_source.items():
            title = source_docs[0].metadata.get('title', 'Unknown Title')
            year = source_docs[0].metadata.get('year', 'unknown')
            
            context_entry = f"=== PAPER: [{source}] ===\n"
            context_entry += f"Title: {title}\n"
            context_entry += f"Year: {year}\n\n"
            
            for doc in source_docs:
                section = doc.metadata.get('section_type', 'other')
                context_entry += f"SECTION: {section.upper()}\n"
                context_entry += f"CONTENT: {doc.page_content}\n"
                context_entry += "---\n"
            
            context_entry += "\n"
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _generate_with_context(self, query: str, context: str, chapter_topic: str = None) -> str:
        """Generate response with formatted context"""
        # Get appropriate system prompt
        if chapter_topic:
            system_prompt = Config.get_chapter_specific_prompt(chapter_topic)
        else:
            system_prompt = Config.get_system_prompt()
        
        # Create user prompt
        user_prompt = f"""
RESEARCH QUESTION: {query}

CHAPTER FOCUS: {chapter_topic.replace('_', ' ').title() if chapter_topic else 'General Literature Review'}

LITERATURE CONTEXT:
{context}

INSTRUCTIONS:
Write a comprehensive literature review response that:
1. Synthesizes findings across multiple studies
2. Uses proper academic citation format [citation_key] for all references
3. Reports specific statistical results with exact values
4. Compares and contrasts findings across studies
5. Maintains academic writing style suitable for literature review

Begin your response with the most significant findings and support each claim with citations.

RESPONSE:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm.generate_response(messages, max_tokens=Config.MAX_TOKENS)
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error generating response: {e}"
