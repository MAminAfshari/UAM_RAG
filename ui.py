# ui.py - Complete Enhanced UI with Journal Impact Integration
"""
Complete user interface for the UAM Literature Review RAG System.
Enhanced with journal impact integration, Excel validation, and Q1 paper prioritization.
"""

import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
import threading
import os
import json
import subprocess
import platform
from datetime import datetime
import webbrowser

# Import core system components
from core.rag_system import UAMRAGSystem
from config import Config, CHAPTER_MAPPING


class ModernToolTip:
    """Create a modern tooltip for tkinter widgets"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=self.text, 
                        background="#333333", foreground="white",
                        relief="solid", borderwidth=1, font=("Arial", 9))
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class UAMLiteratureReviewUI:
    """Enhanced UI class for the UAM Literature Review Assistant with Journal Impact"""
    
    def __init__(self):
        self.rag_system = UAMRAGSystem()
        self.current_query = ""
        self.current_chapter = ""
        self.search_history = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI"""
        self.root = tk.Tk()
        self.root.title("UAM Literature Review Assistant - Enhanced with Journal Impact")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f8f9fa')
        
        # Configure modern styling
        self.setup_styles()
        
        # Create main layout
        self.create_main_layout()
        
        # Create notebook for tabbed interface
        self.create_notebook()
        
        # Create tabs
        self.create_main_tab()
        self.create_corpus_tab()
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar()
        
        # Initialize system
        self.update_system_status()
        
        # Configure keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Modern color palette
        colors = {
            'primary': '#2563eb',
            'secondary': '#64748b',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'background': '#f8f9fa',
            'surface': '#ffffff',
            'text': '#1f2937',
            'text_secondary': '#6b7280'
        }
        
        # Configure styles
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), 
                       background=colors['background'], foreground=colors['text'])
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), 
                       background=colors['background'], foreground=colors['text'])
        style.configure('Info.TLabel', font=('Segoe UI', 9), 
                       background=colors['background'], foreground=colors['text_secondary'])
        style.configure('Success.TLabel', font=('Segoe UI', 9), 
                       background=colors['background'], foreground=colors['success'])
        style.configure('Warning.TLabel', font=('Segoe UI', 9), 
                       background=colors['background'], foreground=colors['warning'])
        style.configure('Error.TLabel', font=('Segoe UI', 9), 
                       background=colors['background'], foreground=colors['error'])
        
        # Button styles
        style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('Secondary.TButton', font=('Segoe UI', 10))
        
        # Frame styles
        style.configure('Card.TFrame', background=colors['surface'], relief='flat', borderwidth=1)
        style.configure('Modern.TLabelframe', background=colors['surface'], 
                       borderwidth=1, relief='solid')
        style.configure('Modern.TLabelframe.Label', background=colors['surface'], 
                       font=('Segoe UI', 11, 'bold'), foreground=colors['text'])
    
    def create_main_layout(self):
        """Create main application layout"""
        # Main container
        self.main_container = ttk.Frame(self.root, style='Card.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(self.main_container, style='Card.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title with icon
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, text="🎓 UAM Literature Review Assistant", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(title_frame, text="v2.1 Enhanced + Journal Impact", 
                                 style='Info.TLabel')
        version_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, 
                                  text="AI-powered synthesis with Q1 journal prioritization and Excel integration",
                                  style='Info.TLabel')
        subtitle_label.pack(pady=(5, 0))
    
    def create_notebook(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab frames
        self.main_tab = ttk.Frame(self.notebook)
        self.corpus_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        # Add tabs
        self.notebook.add(self.main_tab, text="📝 Literature Review")
        self.notebook.add(self.corpus_tab, text="📊 Corpus Analysis")
        self.notebook.add(self.settings_tab, text="⚙️ Settings")
    
    def create_main_tab(self):
        """Create the main literature review tab"""
        # Create main horizontal layout
        main_layout = ttk.Frame(self.main_tab)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel container with fixed width
        left_container = ttk.Frame(main_layout)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.configure(width=580)
        left_container.pack_propagate(False)
        
        # Create scrollable canvas for left panel
        self.left_canvas = tk.Canvas(left_container, bg='#f8f9fa', highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.scrollable_left_frame = ttk.Frame(self.left_canvas)
        
        # Configure scrolling
        self.scrollable_left_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )
        
        self.left_canvas.create_window((0, 0), window=self.scrollable_left_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Pack scrollable components
        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        self._bind_mousewheel()
        
        # Right panel for results
        right_panel = ttk.Frame(main_layout)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Add all the cards to the scrollable frame
        self.create_system_status_card(self.scrollable_left_frame)
        self.create_paper_management_card(self.scrollable_left_frame)
        self.create_chapter_selection_card(self.scrollable_left_frame)
        self.create_query_input_card(self.scrollable_left_frame)
        
        # Add some bottom padding
        ttk.Frame(self.scrollable_left_frame, height=20).pack(fill=tk.X)
        
        # Results Panel
        self.create_results_panel(right_panel)
        
        # Update canvas scroll region
        self.main_tab.after(100, self.update_left_scroll_region)
    
    def _bind_mousewheel(self):
        """Bind mousewheel events to the canvas"""
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.left_canvas.unbind_all("<MouseWheel>")
        
        self.left_canvas.bind('<Enter>', _bind_to_mousewheel)
        self.left_canvas.bind('<Leave>', _unbind_from_mousewheel)
    
    def update_left_scroll_region(self):
        """Update the scroll region of the left canvas"""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
    
    def create_system_status_card(self, parent):
        """Create enhanced system status card with journal impact info"""
        card = ttk.LabelFrame(parent, text="📊 System Status", style='Modern.TLabelframe', padding="15")
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Status indicators
        self.status_frame = ttk.Frame(card)
        self.status_frame.pack(fill=tk.X)
        
        # API Status
        api_frame = ttk.Frame(self.status_frame)
        api_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.api_status_label = ttk.Label(api_frame, text="🔑 API:", style='Info.TLabel')
        self.api_status_label.pack(side=tk.LEFT)
        
        self.api_status_value = ttk.Label(api_frame, text="Checking...", style='Info.TLabel')
        self.api_status_value.pack(side=tk.RIGHT)
        
        # Corpus Status
        corpus_frame = ttk.Frame(self.status_frame)
        corpus_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.corpus_status_label = ttk.Label(corpus_frame, text="📚 Papers:", style='Info.TLabel')
        self.corpus_status_label.pack(side=tk.LEFT)
        
        self.corpus_status_value = ttk.Label(corpus_frame, text="0", style='Info.TLabel')
        self.corpus_status_value.pack(side=tk.RIGHT)
        
        # Journal Impact Status
        journal_frame = ttk.Frame(self.status_frame)
        journal_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.journal_status_label = ttk.Label(journal_frame, text="🏆 Journal Impact:", style='Info.TLabel')
        self.journal_status_label.pack(side=tk.LEFT)
        
        self.journal_status_value = ttk.Label(journal_frame, text="Checking...", style='Info.TLabel')
        self.journal_status_value.pack(side=tk.RIGHT)
        
        # Model Status
        model_frame = ttk.Frame(self.status_frame)
        model_frame.pack(fill=tk.X)
        
        self.model_status_label = ttk.Label(model_frame, text="🤖 Model:", style='Info.TLabel')
        self.model_status_label.pack(side=tk.LEFT)
        
        self.model_status_value = ttk.Label(model_frame, text=Config.LLM_MODEL.split('/')[-1], style='Info.TLabel')
        self.model_status_value.pack(side=tk.RIGHT)
        
        # Refresh button
        refresh_btn = ttk.Button(card, text="🔄 Refresh", command=self.update_system_status)
        refresh_btn.pack(pady=(10, 0))
        ModernToolTip(refresh_btn, "Refresh system status")
    
    def create_paper_management_card(self, parent):
        """Create enhanced paper management card with Excel validation"""
        card = ttk.LabelFrame(parent, text="📚 Paper Management", style='Modern.TLabelframe', padding="15")
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Directory selection
        ttk.Label(card, text="PDF Directory:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        path_frame = ttk.Frame(card)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pdf_path_var = tk.StringVar()
        self.pdf_path_var.set(Config.DEFAULT_PDF_DIRECTORY)  # Set default directory
        self.pdf_entry = ttk.Entry(path_frame, textvariable=self.pdf_path_var, font=('Segoe UI', 9))
        self.pdf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(path_frame, text="📁 Browse", command=self.browse_pdf_directory)
        browse_btn.pack(side=tk.RIGHT, padx=(0, 5))
        ModernToolTip(browse_btn, "Select directory containing UAM research papers")
        
        # Excel validation section
        validation_frame = ttk.Frame(card)
        validation_frame.pack(fill=tk.X, pady=(0, 10))
        
        validate_btn = ttk.Button(validation_frame, text="🔍 Validate Excel Matching", 
                                 command=self.validate_excel_matching)
        validate_btn.pack(side=tk.LEFT, padx=(0, 5))
        ModernToolTip(validate_btn, "Check which PDF files match Excel entries")
        
        report_btn = ttk.Button(validation_frame, text="📋 Generate Report", 
                               command=self.generate_matching_report)
        report_btn.pack(side=tk.LEFT, padx=(0, 5))
        ModernToolTip(report_btn, "Generate detailed PDF-Excel matching report")
        
        # Validation status
        self.validation_status = ttk.Label(card, text="", style='Info.TLabel')
        self.validation_status.pack(pady=(5, 0))
        
        # Ingest button with progress
        ingest_frame = ttk.Frame(card)
        ingest_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.ingest_btn = ttk.Button(ingest_frame, text="📥 Ingest Excel Papers Only", 
                                    command=self.ingest_papers, style='Primary.TButton')
        self.ingest_btn.pack(side=tk.LEFT)
        ModernToolTip(self.ingest_btn, "Process only papers listed in Excel file")
        
        # Progress indicator
        self.ingest_progress = ttk.Progressbar(ingest_frame, mode='indeterminate', length=100)
        self.ingest_progress.pack(side=tk.RIGHT, padx=(10, 0))
    
    def create_chapter_selection_card(self, parent):
        """Create chapter selection card"""
        card = ttk.LabelFrame(parent, text="📖 Chapter Focus", style='Modern.TLabelframe', padding="15")
        card.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(card, text="Select research focus:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        self.chapter_var = tk.StringVar(value="")
        
        # General option
        general_frame = ttk.Frame(card)
        general_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(general_frame, text="🌐 General Review", 
                       variable=self.chapter_var, value="").pack(side=tk.LEFT)
        
        # Chapter options with descriptions
        chapter_descriptions = {
            'core_determinants': "Core behavioral determinants (TAM, TPB, UTAUT)",
            'trust_risk_safety': "Trust, risk perception, and safety concerns",
            'affect_emotion': "Emotional factors and personality traits",
            'contextual_demographic': "Demographics and contextual factors"
        }
        
        for key, desc in chapter_descriptions.items():
            frame = ttk.Frame(card)
            frame.pack(fill=tk.X, pady=(0, 5))
            
            radio = ttk.Radiobutton(frame, text=f"📑 {key.replace('_', ' ').title()}", 
                                   variable=self.chapter_var, value=key)
            radio.pack(side=tk.LEFT)
            
            ModernToolTip(radio, desc)
    
    def create_query_input_card(self, parent):
        """Create query input card"""
        card = ttk.LabelFrame(parent, text="🔍 Research Query", style='Modern.TLabelframe', padding="15")
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Quick queries
        ttk.Label(card, text="Quick Queries:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        self.uam_queries = [
            "What are the main determinants of UAM adoption intention?",
            "How does trust influence UAM acceptance?",
            "What role do safety and risk perceptions play?",
            "How do demographic factors moderate adoption?",
            "What are utilitarian vs hedonic motivations?",
            "How do cultural factors influence acceptance?",
            "What methodological approaches are used?",
            "How do attitude and intention relate?",
            "What are the main barriers to adoption?",
            "How does social influence affect adoption?",
            "What statistical methods are commonly used?",
            "How do environmental concerns influence adoption?"
        ]
        
        self.query_dropdown = ttk.Combobox(card, values=self.uam_queries, 
                                          font=('Segoe UI', 9), height=12)
        self.query_dropdown.pack(fill=tk.X, pady=(0, 10))
        self.query_dropdown.bind('<<ComboboxSelected>>', self.on_query_selected)
        
        # Custom query
        ttk.Label(card, text="Custom Query:", style='Heading.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        self.query_text = scrolledtext.ScrolledText(card, wrap=tk.WORD, height=4, 
                                                   font=('Segoe UI', 9))
        self.query_text.pack(fill=tk.X, pady=(0, 10))
        
        # Search options
        options_frame = ttk.Frame(card)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_sources_var = tk.BooleanVar(value=True)
        self.show_stats_var = tk.BooleanVar(value=True)
        self.academic_format_var = tk.BooleanVar(value=True)
        self.comprehensive_mode_var = tk.BooleanVar(value=Config.ENABLE_COMPREHENSIVE_COVERAGE)
        
        ttk.Checkbutton(options_frame, text="Show citations", 
                       variable=self.show_sources_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Include statistics", 
                       variable=self.show_stats_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Academic format", 
                       variable=self.academic_format_var).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Comprehensive mode (more papers)", 
                       variable=self.comprehensive_mode_var).pack(anchor=tk.W)
        
        # Search button
        self.search_btn = ttk.Button(card, text="🔍 Generate Q1-Prioritized Literature Review", 
                                   command=self.search_literature, style='Primary.TButton')
        self.search_btn.pack(fill=tk.X, pady=(10, 0))
        ModernToolTip(self.search_btn, "Generate comprehensive literature review prioritizing Q1 journals")
        
        # Search progress
        self.search_progress = ttk.Progressbar(card, mode='indeterminate')
        self.search_progress.pack(fill=tk.X, pady=(5, 0))
    
    def create_results_panel(self, parent):
        """Create results panel"""
        # Results header
        results_header = ttk.Frame(parent)
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_header, text="📄 Literature Review Output", 
                 style='Heading.TLabel').pack(side=tk.LEFT)
        
        # Action buttons
        actions_frame = ttk.Frame(results_header)
        actions_frame.pack(side=tk.RIGHT)
        
        self.copy_btn = ttk.Button(actions_frame, text="📋 Copy", 
                                  command=self.copy_to_clipboard)
        self.copy_btn.pack(side=tk.LEFT, padx=(0, 5))
        ModernToolTip(self.copy_btn, "Copy results to clipboard")
        
        self.save_btn = ttk.Button(actions_frame, text="💾 Save", 
                                  command=self.save_results)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))
        ModernToolTip(self.save_btn, "Save results to file")
        
        self.export_btn = ttk.Button(actions_frame, text="📤 Export", 
                                    command=self.export_results)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 5))
        ModernToolTip(self.export_btn, "Export as formatted document")
        
        self.clear_btn = ttk.Button(actions_frame, text="🗑️ Clear", 
                                   command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT)
        ModernToolTip(self.clear_btn, "Clear all results")
        
        # Results notebook
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_tab = ttk.Frame(results_notebook)
        results_notebook.add(results_tab, text="📄 Review")
        
        # Sources tab
        sources_tab = ttk.Frame(results_notebook)
        results_notebook.add(sources_tab, text="📚 Sources")
        
        # History tab
        history_tab = ttk.Frame(results_notebook)
        results_notebook.add(history_tab, text="📋 History")
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_tab, wrap=tk.WORD, 
                                                     font=('Times New Roman', 11),
                                                     bg='white', fg='black')
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text formatting
        self.setup_text_formatting()
        
        # Sources text area
        self.sources_text = scrolledtext.ScrolledText(sources_tab, wrap=tk.WORD,
                                                     font=('Segoe UI', 9),
                                                     bg='#f8f9fa')
        self.sources_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History listbox
        history_frame = ttk.Frame(history_tab)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.history_listbox = tk.Listbox(history_frame, font=('Segoe UI', 9))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind('<Double-1>', self.load_from_history)
        
        # Status label
        self.status_label = ttk.Label(parent, text="Ready to generate Q1-prioritized literature review...", 
                                     style='Info.TLabel')
        self.status_label.pack(pady=(10, 0))
    
    def setup_text_formatting(self):
        """Setup text formatting tags"""
        self.results_text.tag_configure("citation", foreground="#2563eb", 
                                       font=('Times New Roman', 11, 'bold'))
        self.results_text.tag_configure("statistic", foreground="#10b981", 
                                       font=('Times New Roman', 11, 'italic'))
        self.results_text.tag_configure("heading", foreground="#dc2626", 
                                       font=('Times New Roman', 12, 'bold'))
        self.results_text.tag_configure("emphasis", foreground="#7c3aed", 
                                       font=('Times New Roman', 11, 'italic'))
        self.results_text.tag_configure("q1_journal", foreground="#059669", 
                                       font=('Times New Roman', 11, 'bold'))
    
    def create_corpus_tab(self):
        """Create corpus analysis tab"""
        # Corpus statistics
        stats_frame = ttk.LabelFrame(self.corpus_tab, text="📊 Corpus Statistics", 
                                    style='Modern.TLabelframe', padding="15")
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.corpus_stats_text = scrolledtext.ScrolledText(stats_frame, height=15, 
                                                          font=('Segoe UI', 9))
        self.corpus_stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        refresh_corpus_btn = ttk.Button(stats_frame, text="🔄 Refresh Statistics", 
                                       command=self.update_corpus_stats)
        refresh_corpus_btn.pack(pady=(10, 0))
        
        # Corpus management
        management_frame = ttk.LabelFrame(self.corpus_tab, text="⚙️ Corpus Management", 
                                         style='Modern.TLabelframe', padding="15")
        management_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Management buttons
        btn_frame = ttk.Frame(management_frame)
        btn_frame.pack(fill=tk.X)
        
        rebuild_btn = ttk.Button(btn_frame, text="🔄 Rebuild Index", 
                               command=self.rebuild_index)
        rebuild_btn.pack(side=tk.LEFT, padx=(0, 10))
        ModernToolTip(rebuild_btn, "Rebuild the entire search index")
        
        export_corpus_btn = ttk.Button(btn_frame, text="📤 Export Corpus", 
                                      command=self.export_corpus)
        export_corpus_btn.pack(side=tk.LEFT, padx=(0, 10))
        ModernToolTip(export_corpus_btn, "Export corpus statistics")
        
        backup_btn = ttk.Button(btn_frame, text="💾 Backup", 
                               command=self.backup_corpus)
        backup_btn.pack(side=tk.LEFT)
        ModernToolTip(backup_btn, "Create corpus backup")
    
    def create_settings_tab(self):
        """Create enhanced settings tab with journal impact options"""
        # Model settings
        model_frame = ttk.LabelFrame(self.settings_tab, text="🤖 Model Settings", 
                                    style='Modern.TLabelframe', padding="15")
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Temperature setting
        temp_frame = ttk.Frame(model_frame)
        temp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(temp_frame, text="Temperature:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=Config.TEMPERATURE)
        temp_scale = ttk.Scale(temp_frame, from_=0.0, to=1.0, variable=self.temp_var, 
                              orient=tk.HORIZONTAL, length=200)
        temp_scale.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Max tokens setting
        tokens_frame = ttk.Frame(model_frame)
        tokens_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tokens_frame, text="Max Tokens:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.tokens_var = tk.IntVar(value=Config.MAX_TOKENS)
        tokens_entry = ttk.Entry(tokens_frame, textvariable=self.tokens_var, width=10)
        tokens_entry.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Retrieval settings
        retrieval_frame = ttk.LabelFrame(self.settings_tab, text="🔍 Retrieval Settings", 
                                        style='Modern.TLabelframe', padding="15")
        retrieval_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Retrieval K setting
        k_frame = ttk.Frame(retrieval_frame)
        k_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(k_frame, text="Retrieval K:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=Config.RETRIEVAL_K)
        k_entry = ttk.Entry(k_frame, textvariable=self.k_var, width=10)
        k_entry.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Min/Max papers settings
        papers_frame = ttk.Frame(retrieval_frame)
        papers_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(papers_frame, text="Min Papers:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.min_papers_var = tk.IntVar(value=Config.MIN_PAPERS_PER_RESPONSE)
        min_papers_entry = ttk.Entry(papers_frame, textvariable=self.min_papers_var, width=10)
        min_papers_entry.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(papers_frame, text="Max Papers:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.max_papers_var = tk.IntVar(value=Config.MAX_PAPERS_PER_RESPONSE)
        max_papers_entry = ttk.Entry(papers_frame, textvariable=self.max_papers_var, width=10)
        max_papers_entry.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Journal Impact settings
        journal_frame = ttk.LabelFrame(self.settings_tab, text="🏆 Journal Impact Settings", 
                                      style='Modern.TLabelframe', padding="15")
        journal_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Journal impact weight
        impact_weight_frame = ttk.Frame(journal_frame)
        impact_weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(impact_weight_frame, text="Journal Impact Weight:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.impact_weight_var = tk.DoubleVar(value=Config.JOURNAL_IMPACT_WEIGHT)
        impact_weight_scale = ttk.Scale(impact_weight_frame, from_=0.0, to=1.0, 
                                      variable=self.impact_weight_var, orient=tk.HORIZONTAL, length=200)
        impact_weight_scale.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Statistical content weight
        stats_weight_frame = ttk.Frame(journal_frame)
        stats_weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(stats_weight_frame, text="Statistical Content Weight:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.stats_weight_var = tk.DoubleVar(value=Config.STATISTICAL_CONTENT_WEIGHT)
        stats_weight_scale = ttk.Scale(stats_weight_frame, from_=0.0, to=1.0, 
                                     variable=self.stats_weight_var, orient=tk.HORIZONTAL, length=200)
        stats_weight_scale.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Q1 boost multiplier
        q1_boost_frame = ttk.Frame(journal_frame)
        q1_boost_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(q1_boost_frame, text="Q1 Paper Boost:", style='Heading.TLabel').pack(side=tk.LEFT)
        self.q1_boost_var = tk.DoubleVar(value=Config.JOURNAL_BOOST_MULTIPLIERS['Q1'])
        q1_boost_scale = ttk.Scale(q1_boost_frame, from_=1.0, to=3.0, 
                                 variable=self.q1_boost_var, orient=tk.HORIZONTAL, length=200)
        q1_boost_scale.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Enhanced Feature toggles
        features_frame = ttk.LabelFrame(self.settings_tab, text="⚡ Enhanced Features", 
                                       style='Modern.TLabelframe', padding="15")
        features_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Initialize variables
        self.journal_ranking_var = tk.BooleanVar(value=Config.ENABLE_JOURNAL_RANKING)
        self.comprehensive_var = tk.BooleanVar(value=Config.ENABLE_COMPREHENSIVE_COVERAGE)
        self.tier_balancing_var = tk.BooleanVar(value=Config.ENABLE_TIER_BALANCING)
        self.paper_diversity_var = tk.BooleanVar(value=Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT)
        self.q1_priority_var = tk.BooleanVar(value=Config.ENABLE_Q1_PAPER_PRIORITY)
        self.hyde_var = tk.BooleanVar(value=Config.ENABLE_HYDE)
        self.multi_query_var = tk.BooleanVar(value=Config.ENABLE_MULTI_QUERY)
        self.stats_extraction_var = tk.BooleanVar(value=Config.ENABLE_STATISTICAL_EXTRACTION)
        self.multimodal_var = tk.BooleanVar(value=Config.ENABLE_MULTIMODAL)
        
        # Feature checkboxes
        ttk.Checkbutton(features_frame, text="🏆 Enable Journal Impact Ranking", 
                       variable=self.journal_ranking_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="📊 Enable Comprehensive Coverage", 
                       variable=self.comprehensive_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="⚖️ Enable Quality Tier Balancing", 
                       variable=self.tier_balancing_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="🎯 Enable Paper Diversity Enforcement", 
                       variable=self.paper_diversity_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="🥇 Enable Q1 Paper Priority", 
                       variable=self.q1_priority_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="🔮 Enable HyDE", 
                       variable=self.hyde_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="🔄 Enable Multi-Query", 
                       variable=self.multi_query_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="📈 Enable Statistical Extraction", 
                       variable=self.stats_extraction_var).pack(anchor=tk.W)
        ttk.Checkbutton(features_frame, text="🖼️ Enable Multimodal Processing", 
                       variable=self.multimodal_var).pack(anchor=tk.W)
        
        # Save settings button
        save_settings_btn = ttk.Button(self.settings_tab, text="💾 Save Enhanced Settings", 
                                      command=self.save_enhanced_settings, style='Primary.TButton')
        save_settings_btn.pack(pady=(20, 0))
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.main_container)
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Status text
        self.status_text = ttk.Label(self.status_bar, text="Ready", style='Info.TLabel')
        self.status_text.pack(side=tk.LEFT)
        
        # Progress bar
        self.main_progress = ttk.Progressbar(self.status_bar, length=200, mode='determinate')
        self.main_progress.pack(side=tk.RIGHT, padx=(10, 0))
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-s>', lambda e: self.save_results())
        self.root.bind('<Control-c>', lambda e: self.copy_to_clipboard())
        self.root.bind('<Control-r>', lambda e: self.search_literature())
        self.root.bind('<F5>', lambda e: self.update_system_status())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
    
    # =============================================================================
    # Event Handlers
    # =============================================================================
    
    def on_query_selected(self, event):
        """Handle query selection from dropdown"""
        selected_query = self.query_dropdown.get()
        self.query_text.delete("1.0", tk.END)
        self.query_text.insert("1.0", selected_query)
    
    def update_system_status(self):
        """Update system status display with journal impact information"""
        try:
            # Check API key
            if Config.OPENROUTER_API_KEY:
                self.api_status_value.config(text="✅ Configured", style='Success.TLabel')
            else:
                self.api_status_value.config(text="❌ Missing", style='Error.TLabel')
            
            # Get enhanced system status
            system_status = self.rag_system.get_system_status()
            
            # Check corpus with journal impact info
            if system_status.get('is_ready', False):
                corpus_stats = system_status.get('corpus_stats', {})
                paper_count = corpus_stats.get('total_papers', 0)
                
                # Add journal impact information
                journal_status = system_status.get('journal_impact_status', {})
                if journal_status.get('metadata_loaded', False):
                    journal_info = journal_status.get('journal_statistics', {})
                    papers_with_impact = journal_info.get('total_papers_mapped', 0)
                    status_text = f"✅ {paper_count} papers ({papers_with_impact} with journal info)"
                else:
                    status_text = f"✅ {paper_count} papers (no journal info)"
                
                self.corpus_status_value.config(text=status_text, style='Success.TLabel')
                
                # Update journal impact status
                if journal_status.get('metadata_loaded', False):
                    self.journal_status_value.config(text="✅ Enabled", style='Success.TLabel')
                else:
                    self.journal_status_value.config(text="❌ Disabled", style='Error.TLabel')
                
                self.update_corpus_stats()
            else:
                self.corpus_status_value.config(text="⚠️ No papers", style='Warning.TLabel')
                self.journal_status_value.config(text="⚠️ Not ready", style='Warning.TLabel')
                
        except Exception as e:
            self.api_status_value.config(text="❌ Error", style='Error.TLabel')
            print(f"Status update error: {e}")
    
    def update_corpus_stats(self):
        """Update corpus statistics display with journal impact information"""
        try:
            stats = self.rag_system.get_corpus_statistics()
            
            if stats:
                # Enhanced stats display with journal impact
                stats_text = f"""
📊 CORPUS OVERVIEW
{'='*50}
Total Papers: {stats.get('total_papers', 0)}
Total Chunks: {stats.get('total_chunks', 0)}
Average Chunks per Paper: {stats.get('total_chunks', 0) / max(stats.get('total_papers', 1), 1):.1f}
Processing Mode: {stats.get('ingestion_stats', {}).get('processing_mode', 'unknown')}

🏆 JOURNAL IMPACT ANALYSIS
{'='*50}
"""
                
                # Add journal impact statistics
                journal_stats = stats.get('journal_impact_stats', {})
                if journal_stats:
                    stats_text += f"Papers with Journal Info: {journal_stats.get('papers_with_journal_info', 0)}\n"
                    stats_text += f"Papers without Journal Info: {journal_stats.get('papers_without_journal_info', 0)}\n"
                    
                    # Quality tier distribution
                    tier_dist = journal_stats.get('tier_distribution', {})
                    if tier_dist:
                        stats_text += f"\nQuality Tier Distribution:\n"
                        for tier, count in tier_dist.items():
                            emoji = "🥇" if tier == "top" else "🥈" if tier == "high" else "🥉" if tier == "medium" else "📄"
                            stats_text += f"  {emoji} {tier.title()}: {count}\n"
                    
                    # Journal distribution
                    journal_dist = journal_stats.get('journal_distribution', {})
                    if journal_dist:
                        stats_text += f"\nTop Journals:\n"
                        sorted_journals = sorted(journal_dist.items(), key=lambda x: x[1], reverse=True)
                        for journal, count in sorted_journals[:10]:
                            stats_text += f"  • {journal}: {count}\n"
                
                stats_text += f"\n📑 SECTION DISTRIBUTION\n{'='*50}\n"
                
                section_dist = stats.get('section_distribution', {})
                for section, count in sorted(section_dist.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / stats.get('total_chunks', 1)) * 100
                    emoji = "📊" if section in ['results', 'findings'] else "💭" if section == 'discussion' else "📝"
                    stats_text += f"{emoji} {section.title()}: {count} ({percentage:.1f}%)\n"
                
                # Statistical content analysis
                stats_text += f"\n📈 STATISTICAL CONTENT\n{'='*50}\n"
                
                statistical_dist = stats.get('statistical_content_distribution', {})
                if statistical_dist:
                    total_statistical = sum(statistical_dist.values())
                    stats_text += f"Total Statistical Content: {total_statistical}\n"
                    for section, count in sorted(statistical_dist.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"  📊 {section.title()}: {count}\n"
                
                # Top keywords
                stats_text += f"\n🔍 TOP KEYWORDS\n{'='*50}\n"
                
                keyword_dist = stats.get('keyword_distribution', {})
                for keyword, count in list(keyword_dist.items())[:15]:
                    stats_text += f"🔹 {keyword}: {count}\n"
                
                # Processed papers
                stats_text += f"\n📚 PROCESSED PAPERS\n{'='*50}\n"
                
                papers = stats.get('papers_processed', [])
                for paper in sorted(papers)[:20]:
                    stats_text += f"✅ {paper}\n"
                
                if len(papers) > 20:
                    stats_text += f"... and {len(papers) - 20} more papers\n"
                
                self.corpus_stats_text.delete("1.0", tk.END)
                self.corpus_stats_text.insert("1.0", stats_text)
            else:
                self.corpus_stats_text.delete("1.0", tk.END)
                self.corpus_stats_text.insert("1.0", "No corpus statistics available. Please ingest papers first.")
                
        except Exception as e:
            self.corpus_stats_text.delete("1.0", tk.END)
            self.corpus_stats_text.insert("1.0", f"Error loading corpus statistics: {e}")
    
    def validate_excel_matching(self):
        """Validate PDF files against Excel entries"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path or not os.path.exists(pdf_path):
            messagebox.showerror("Error", "Please select a valid PDF directory")
            return
        
        try:
            # Get validation results
            validation_results = self.rag_system.document_ingester.validate_pdf_directory(pdf_path)
            
            if 'error' in validation_results:
                messagebox.showerror("Validation Error", validation_results['error'])
                return
            
            # Update status
            match_rate = validation_results.get('match_rate', 0)
            matched_count = validation_results.get('matched_count', 0)
            total_excel = validation_results.get('total_excel_papers', 0)
            
            if match_rate >= 90:
                status_text = f"✅ Excellent match: {matched_count}/{total_excel} papers ({match_rate:.1f}%)"
                self.validation_status.config(text=status_text, style='Success.TLabel')
            elif match_rate >= 70:
                status_text = f"⚠️ Good match: {matched_count}/{total_excel} papers ({match_rate:.1f}%)"
                self.validation_status.config(text=status_text, style='Warning.TLabel')
            else:
                status_text = f"❌ Poor match: {matched_count}/{total_excel} papers ({match_rate:.1f}%)"
                self.validation_status.config(text=status_text, style='Error.TLabel')
            
            # Show detailed results
            if validation_results.get('unmatched_excel_papers'):
                missing_count = len(validation_results['unmatched_excel_papers'])
                messagebox.showwarning(
                    "Missing PDFs", 
                    f"{missing_count} papers in Excel have no matching PDF files.\n\n" +
                    "Click 'Generate Report' for detailed information."
                )
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating files: {e}")
    
    def generate_matching_report(self):
        """Generate comprehensive PDF-Excel matching report"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path or not os.path.exists(pdf_path):
            messagebox.showerror("Error", "Please select a valid PDF directory")
            return
        
        try:
            # Generate report
            report = self.rag_system.document_ingester.generate_pre_ingestion_report(pdf_path)
            
            # Save report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"pdf_excel_matching_report_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Show success message
            messagebox.showinfo(
                "Report Generated", 
                f"Matching report saved to: {report_filename}\n\n" +
                "This report shows which PDF files match Excel entries."
            )
            
            # Ask if user wants to open the report
            if messagebox.askyesno("Open Report", "Would you like to open the report file?"):
                if platform.system() == "Windows":
                    os.startfile(report_filename)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", report_filename])
                else:  # Linux
                    subprocess.run(["xdg-open", report_filename])
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Error generating report: {e}")
    
    def browse_pdf_directory(self):
        """Browse for PDF directory with default to data/pdfs"""
        initial_dir = self.pdf_path_var.get() or Config.DEFAULT_PDF_DIRECTORY
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~/Documents")
        
        directory = filedialog.askdirectory(
            title="Select Directory with UAM Research Papers",
            initialdir=initial_dir
        )
        
        if directory:
            self.pdf_path_var.set(directory)
            
            # Auto-validate when directory is selected
            try:
                validation_results = self.rag_system.document_ingester.validate_pdf_directory(directory)
                
                if 'error' in validation_results:
                    self.validation_status.config(text=f"❌ {validation_results['error']}", style='Error.TLabel')
                else:
                    match_rate = validation_results.get('match_rate', 0)
                    matched_count = validation_results.get('matched_count', 0)
                    total_pdfs = validation_results.get('total_pdfs', 0)
                    
                    status_text = f"Found {total_pdfs} PDFs, {matched_count} matched to Excel ({match_rate:.1f}%)"
                    
                    if match_rate >= 90:
                        self.validation_status.config(text=f"✅ {status_text}", style='Success.TLabel')
                    elif match_rate >= 70:
                        self.validation_status.config(text=f"⚠️ {status_text}", style='Warning.TLabel')
                    else:
                        self.validation_status.config(text=f"❌ {status_text}", style='Error.TLabel')
                        
            except Exception as e:
                self.validation_status.config(text=f"❌ Validation error: {e}", style='Error.TLabel')
    
    def ingest_papers(self):
        """Ingest papers with enhanced Excel validation"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path or not os.path.exists(pdf_path):
            messagebox.showerror("Error", "Please select a valid PDF directory")
            return
        
        try:
            # Validate directory first
            validation_results = self.rag_system.document_ingester.validate_pdf_directory(pdf_path)
            
            if 'error' in validation_results:
                messagebox.showerror("Validation Error", validation_results['error'])
                return
            
            if not validation_results.get('ready_for_ingestion', False):
                messagebox.showerror(
                    "Ingestion Error", 
                    "No PDF files are ready for ingestion.\n\n" +
                    "This could mean:\n" +
                    "• No PDF files match Excel entries\n" +
                    "• Excel file is missing or corrupted\n" +
                    "• Journal metadata not loaded\n\n" +
                    "Click 'Validate Excel Matching' for more details."
                )
                return
            
            # Show confirmation with details
            matched_count = validation_results.get('matched_count', 0)
            total_pdfs = validation_results.get('total_pdfs', 0)
            unmatched_count = validation_results.get('unmatched_pdfs', 0)
            
            confirmation_msg = f"Ready to ingest {matched_count} PDF files that match Excel entries.\n\n"
            confirmation_msg += f"• Total PDFs in directory: {total_pdfs}\n"
            confirmation_msg += f"• Matched to Excel: {matched_count}\n"
            confirmation_msg += f"• Will be ignored: {unmatched_count}\n\n"
            confirmation_msg += "This process may take several minutes. Continue?"
            
            if not messagebox.askyesno("Confirm Ingestion", confirmation_msg):
                return
            
            # Start ingestion
            self.status_text.config(text=f"Processing {matched_count} Excel-matched PDF files...")
            self.ingest_progress.start()
            self.ingest_btn.config(state=tk.DISABLED)
            self.main_progress.config(mode='indeterminate')
            self.main_progress.start()
            
            def ingest_thread():
                try:
                    self.rag_system.ingest_papers(pdf_path)
                    self.root.after(0, self.on_ingestion_complete, matched_count)
                except Exception as e:
                    self.root.after(0, self.on_ingestion_error, str(e))
            
            threading.Thread(target=ingest_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Ingestion Error", f"Error preparing ingestion: {e}")
    
    def on_ingestion_complete(self, file_count):
        """Handle successful ingestion completion with Excel details"""
        self.ingest_progress.stop()
        self.main_progress.stop()
        self.ingest_btn.config(state=tk.NORMAL)
        
        # Get detailed stats
        stats = self.rag_system.get_corpus_statistics()
        journal_stats = stats.get('journal_impact_stats', {})
        
        papers_with_journal = journal_stats.get('papers_with_journal_info', 0)
        tier_dist = journal_stats.get('tier_distribution', {})
        q1_papers = tier_dist.get('top', 0)
        
        success_msg = f"Successfully processed {file_count} Excel-matched papers!\n\n"
        success_msg += f"• Papers with journal info: {papers_with_journal}\n"
        success_msg += f"• Q1 papers: {q1_papers}\n"
        success_msg += f"• Total chunks created: {stats.get('total_chunks', 0)}\n\n"
        success_msg += "System is ready for enhanced literature review generation."
        
        self.status_text.config(text=f"✅ Successfully processed {file_count} Excel papers")
        self.update_system_status()
        messagebox.showinfo("Success", success_msg)
    
    def on_ingestion_error(self, error_msg):
        """Handle ingestion error"""
        self.ingest_progress.stop()
        self.main_progress.stop()
        self.ingest_btn.config(state=tk.NORMAL)
        self.status_text.config(text="❌ Error during paper ingestion")
        messagebox.showerror("Ingestion Error", f"Failed to ingest papers:\n{error_msg}")
    
    def search_literature(self):
        """Execute literature search with enhanced journal impact capabilities"""
        query = self.query_dropdown.get().strip()
        if not query:
            query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a research query")
            return
        
        chapter_topic = self.chapter_var.get() if self.chapter_var.get() else None
        
        if not hasattr(self.rag_system, 'db') or not self.rag_system.db:
            messagebox.showerror("Error", "Please ingest papers first before generating literature reviews")
            return
        
        self.current_query = query
        self.current_chapter = chapter_topic or "General"
        
        self.status_text.config(text="🔍 Generating Q1-prioritized literature review...")
        self.search_progress.start()
        self.search_btn.config(state=tk.DISABLED)
        self.main_progress.config(mode='indeterminate')
        self.main_progress.start()
        
        self.results_text.delete("1.0", tk.END)
        self.sources_text.delete("1.0", tk.END)
        
        def search_thread():
            try:
                # Use enhanced literature query method
                result = self.rag_system.answer_literature_query(
                    query, 
                    chapter_topic, 
                    include_figures=True,
                    include_tables=True,
                    comprehensive_mode=self.comprehensive_mode_var.get()
                )
                
                # Handle different return formats
                if len(result) == 3:
                    answer, sources, quality_metadata = result
                    self.root.after(0, self.on_search_complete_enhanced, answer, sources, quality_metadata)
                else:
                    # Fallback for compatibility
                    answer, sources = result
                    self.root.after(0, self.on_search_complete, answer, sources)
                    
            except Exception as e:
                self.root.after(0, self.on_search_error, str(e))
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def on_search_complete_enhanced(self, answer, sources, quality_metadata):
        """Handle successful enhanced search completion with quality metadata"""
        self.search_progress.stop()
        self.main_progress.stop()
        self.search_btn.config(state=tk.NORMAL)
        
        # Display enhanced results
        self.display_formatted_results(answer)
        self.display_sources_enhanced(sources, quality_metadata)
        self.add_to_history(self.current_query, self.current_chapter, answer, quality_metadata)
        
        # Enhanced status message
        total_papers = quality_metadata.get('unique_papers', len(sources))
        tier_dist = quality_metadata.get('tier_distribution', {})
        high_quality_count = tier_dist.get('top', 0) + tier_dist.get('high', 0)
        
        status_msg = f"✅ Q1-Enhanced review generated ({total_papers} papers, {high_quality_count} high-quality)"
        self.status_text.config(text=status_msg)
    
    def on_search_complete(self, answer, sources):
        """Handle successful search completion (fallback)"""
        self.search_progress.stop()
        self.main_progress.stop()
        self.search_btn.config(state=tk.NORMAL)
        
        self.display_formatted_results(answer)
        self.display_sources(sources)
        self.add_to_history(self.current_query, self.current_chapter, answer)
        
        self.status_text.config(text=f"✅ Literature review generated ({len(sources)} sources)")
    
    def on_search_error(self, error_msg):
        """Handle search error"""
        self.search_progress.stop()
        self.main_progress.stop()
        self.search_btn.config(state=tk.NORMAL)
        self.status_text.config(text="❌ Error generating literature review")
        
        error_display = f"Error generating literature review:\n\n{error_msg}"
        self.results_text.insert(tk.END, error_display)
        messagebox.showerror("Search Error", error_display)
    
    def display_formatted_results(self, answer):
        """Display results with enhanced formatting"""
        self.results_text.delete("1.0", tk.END)
        
        # Add header
        header = f"Enhanced Q1-Prioritized Literature Review Results\n"
        header += f"Query: {self.current_query}\n"
        header += f"Focus: {self.current_chapter}\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 80 + "\n\n"
        
        self.results_text.insert(tk.END, header, "heading")
        
        # Format the answer with enhanced syntax highlighting
        lines = answer.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                # Process citations with Q1 journal detection
                parts = line.split('[')
                self.results_text.insert(tk.END, parts[0])
                
                for part in parts[1:]:
                    if ']' in part:
                        citation_parts = part.split(']', 1)
                        citation_text = f"[{citation_parts[0]}]"
                        
                        # Check if this might be a Q1 journal citation
                        if any(q1_term in citation_text.lower() for q1_term in ['q1', 'top-tier', 'leading']):
                            self.results_text.insert(tk.END, citation_text, "q1_journal")
                        else:
                            self.results_text.insert(tk.END, citation_text, "citation")
                        
                        if len(citation_parts) > 1:
                            self.results_text.insert(tk.END, citation_parts[1])
                    else:
                        self.results_text.insert(tk.END, f"[{part}")
            else:
                # Check for statistical information
                if any(stat in line.lower() for stat in ['β =', 'p <', 'r² =', 'n =', 'α =', 'rmsea', 'cfi']):
                    self.results_text.insert(tk.END, line, "statistic")
                else:
                    self.results_text.insert(tk.END, line)
            
            self.results_text.insert(tk.END, "\n")
        
        self.results_text.see("1.0")
    
    def display_sources_enhanced(self, sources, quality_metadata):
        """Display sources with enhanced quality information"""
        self.sources_text.delete("1.0", tk.END)
        
        if sources:
            # Enhanced sources header with quality summary
            total_papers = quality_metadata.get('unique_papers', len(sources))
            quality_score = quality_metadata.get('quality_score', 0.0)
            
            sources_header = f"Enhanced Q1-Prioritized Sources ({total_papers} papers, Quality Score: {quality_score:.2f})\n"
            sources_header += "=" * 70 + "\n\n"
            
            # Quality distribution summary
            tier_dist = quality_metadata.get('tier_distribution', {})
            if tier_dist:
                sources_header += "🏆 Quality Distribution:\n"
                for tier, count in tier_dist.items():
                    if count > 0:
                        emoji = "🥇" if tier == "top" else "🥈" if tier == "high" else "🥉" if tier == "medium" else "📄"
                        sources_header += f"  {emoji} {tier.title()}: {count} papers\n"
                sources_header += "\n"
            
            # Statistical content summary
            statistical_count = quality_metadata.get('statistical_content_count', 0)
            total_docs = quality_metadata.get('total_documents', 0)
            if total_docs > 0:
                statistical_pct = (statistical_count / total_docs) * 100
                sources_header += f"📊 Statistical Content: {statistical_count}/{total_docs} documents ({statistical_pct:.1f}%)\n\n"
            
            # Section distribution
            section_dist = quality_metadata.get('section_distribution', {})
            if section_dist:
                sources_header += "📑 Section Distribution:\n"
                for section, count in sorted(section_dist.items(), key=lambda x: x[1], reverse=True):
                    emoji = "📊" if section in ['results', 'findings'] else "💭" if section == 'discussion' else "📝"
                    sources_header += f"  {emoji} {section.title()}: {count}\n"
                sources_header += "\n"
            
            self.sources_text.insert(tk.END, sources_header)
            
            # Individual sources with quality information
            for i, source in enumerate(sources, 1):
                source_text = f"{i}. {source}\n"
                
                # Add quality indicators if available
                try:
                    if hasattr(self.rag_system, 'journal_manager') and self.rag_system.journal_manager:
                        # Extract source name (remove quality info if already present)
                        clean_source = source.split(' (')[0] if ' (' in source else source
                        
                        journal_info = self.rag_system.journal_manager.match_paper_to_journal(clean_source)
                        if journal_info:
                            tier = self.rag_system.journal_manager.get_quality_tier(clean_source)
                            quality_score = self.rag_system.journal_manager.get_paper_quality_score(clean_source)
                            
                            tier_emoji = "🥇" if tier == "top" else "🥈" if tier == "high" else "🥉" if tier == "medium" else "📄"
                            source_text += f"   {tier_emoji} Quality: {tier.title()} (Score: {quality_score:.2f})\n"
                            source_text += f"   📖 Journal: {journal_info.get('journal', 'Unknown')}\n"
                        else:
                            source_text += f"   ❓ Quality: Unknown (no journal info)\n"
                    else:
                        source_text += f"   ✅ Status: Processed and indexed\n"
                except Exception as e:
                    source_text += f"   ✅ Status: Processed (quality info unavailable)\n"
                
                source_text += "\n"
                self.sources_text.insert(tk.END, source_text)
        else:
            self.sources_text.insert(tk.END, "No sources found for this query.")
    
    def display_sources(self, sources):
        """Display sources with basic formatting (fallback)"""
        self.sources_text.delete("1.0", tk.END)
        
        if sources:
            sources_header = f"Sources Used ({len(sources)} papers)\n"
            sources_header += "=" * 50 + "\n\n"
            self.sources_text.insert(tk.END, sources_header)
            
            for i, source in enumerate(sources, 1):
                source_text = f"{i}. [{source}]\n"
                try:
                    stats = self.rag_system.get_corpus_statistics()
                    if stats and 'papers_processed' in stats:
                        source_text += f"   Status: ✅ Processed and indexed\n"
                except:
                    pass
                
                source_text += "\n"
                self.sources_text.insert(tk.END, source_text)
        else:
            self.sources_text.insert(tk.END, "No sources found for this query.")
    
    def add_to_history(self, query, chapter, answer, quality_metadata=None):
        """Add search to history with enhanced quality information"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Create enhanced history entry
        history_entry = {
            'timestamp': timestamp,
            'query': query,
            'chapter': chapter,
            'answer': answer,
            'quality_metadata': quality_metadata,
            'preview': query[:50] + "..." if len(query) > 50 else query
        }
        
        self.search_history.append(history_entry)
        
        # Enhanced display text with quality info
        quality_info = ""
        if quality_metadata:
            unique_papers = quality_metadata.get('unique_papers', 0)
            quality_score = quality_metadata.get('quality_score', 0.0)
            tier_dist = quality_metadata.get('tier_distribution', {})
            q1_count = tier_dist.get('top', 0)
            quality_info = f" | {unique_papers}p, Q1:{q1_count}, QS:{quality_score:.2f}"
        
        display_text = f"[{timestamp}] {chapter}: {history_entry['preview']}{quality_info}"
        self.history_listbox.insert(0, display_text)
        
        # Limit history size
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]
            self.history_listbox.delete(50, tk.END)
    
    def load_from_history(self, event):
        """Load query from history"""
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.search_history):
                entry = self.search_history[-(index + 1)]
                
                self.query_text.delete("1.0", tk.END)
                self.query_text.insert("1.0", entry['query'])
                
                if entry['chapter'] != "General":
                    chapter_key = entry['chapter'].lower().replace(' ', '_')
                    self.chapter_var.set(chapter_key)
                
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert("1.0", entry['answer'])
                
                self.notebook.select(self.main_tab)
    
    def copy_to_clipboard(self):
        """Copy results to clipboard"""
        try:
            content = self.results_text.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_text.config(text="✅ Results copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
    
    def save_results(self):
        """Save results to file"""
        try:
            content = self.results_text.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("Warning", "No results to save")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Literature Review Results"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"UAM Literature Review Results - Enhanced with Journal Impact\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Query: {self.current_query}\n")
                    f.write(f"Focus: {self.current_chapter}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(content)
                
                self.status_text.config(text=f"✅ Results saved to {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def export_results(self):
        """Export results in various formats"""
        try:
            content = self.results_text.get("1.0", tk.END)
            if not content.strip():
                messagebox.showwarning("Warning", "No results to export")
                return
            
            format_choice = messagebox.askyesno(
                "Export Format", 
                "Choose export format:\nYes = Rich Text Format (RTF)\nNo = Plain Text"
            )
            
            if format_choice:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".rtf",
                    filetypes=[("Rich Text Format", "*.rtf"), ("All files", "*.*")],
                    title="Export as RTF"
                )
                
                if filename:
                    self.export_as_rtf(filename, content)
            else:
                self.save_results()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def export_as_rtf(self, filename, content):
        """Export results as RTF with formatting"""
        try:
            rtf_content = "{\\rtf1\\ansi\\deff0"
            rtf_content += "{\\fonttbl{\\f0 Times New Roman;}{\\f1 Arial;}}"
            rtf_content += "{\\colortbl;\\red0\\green0\\blue255;\\red16\\green185\\blue129;\\red220\\green38\\blue38;\\red5\\green150\\blue105;}"
            rtf_content += "\\f0\\fs22 "
            
            # Add header
            rtf_content += f"\\b\\fs28 UAM Literature Review Results - Enhanced with Journal Impact\\b0\\fs22\\par"
            rtf_content += f"\\b Generated: \\b0 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\par"
            rtf_content += f"\\b Query: \\b0 {self.current_query}\\par"
            rtf_content += f"\\b Focus: \\b0 {self.current_chapter}\\par"
            rtf_content += "\\line\\par"
            
            # Process content with enhanced formatting
            lines = content.split('\n')
            for line in lines:
                if '[' in line and ']' in line:
                    if any(q1_term in line.lower() for q1_term in ['q1', 'top-tier', 'leading']):
                        line = line.replace('[', '{\\cf4\\b [').replace(']', ']\\cf0\\b0}')  # Q1 journals in green
                    else:
                        line = line.replace('[', '{\\cf1\\b [').replace(']', ']\\cf0\\b0}')  # Regular citations in blue
                
                if any(stat in line.lower() for stat in ['β =', 'p <', 'r² =', 'n =', 'α =', 'rmsea', 'cfi']):
                    line = f"{{\\cf2\\i {line}\\cf0\\i0}}"  # Statistics in green italic
                
                rtf_content += line + "\\par"
            
            rtf_content += "}"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(rtf_content)
            
            self.status_text.config(text=f"✅ Results exported to {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export as RTF: {e}")
    
    def clear_results(self):
        """Clear all results"""
        self.results_text.delete("1.0", tk.END)
        self.sources_text.delete("1.0", tk.END)
        self.status_text.config(text="Results cleared")
    
    def rebuild_index(self):
        """Rebuild the search index"""
        result = messagebox.askyesno(
            "Rebuild Index", 
            "This will rebuild the entire search index. This may take several minutes. Continue?"
        )
        if not result:
            return
        
        pdf_path = self.pdf_path_var.get()
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF directory first")
            return
        
        self.status_text.config(text="🔄 Rebuilding search index...")
        self.main_progress.start()
        
        def rebuild_thread():
            try:
                self.rag_system.ingest_papers(pdf_path)
                self.root.after(0, self.on_rebuild_complete)
            except Exception as e:
                self.root.after(0, self.on_rebuild_error, str(e))
        
        threading.Thread(target=rebuild_thread, daemon=True).start()
    
    def on_rebuild_complete(self):
        """Handle rebuild completion"""
        self.main_progress.stop()
        self.status_text.config(text="✅ Search index rebuilt successfully")
        self.update_system_status()
        messagebox.showinfo("Success", "Search index rebuilt successfully!")
    
    def on_rebuild_error(self, error_msg):
        """Handle rebuild error"""
        self.main_progress.stop()
        self.status_text.config(text="❌ Error rebuilding index")
        messagebox.showerror("Rebuild Error", f"Failed to rebuild index:\n{error_msg}")
    
    def export_corpus(self):
        """Export corpus statistics"""
        try:
            stats = self.rag_system.get_corpus_statistics()
            if not stats:
                messagebox.showwarning("Warning", "No corpus statistics available")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Corpus Statistics"
            )
            
            if filename:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2, ensure_ascii=False)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"UAM Literature Review Corpus Statistics\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(json.dumps(stats, indent=2, ensure_ascii=False))
                
                self.status_text.config(text=f"✅ Corpus statistics exported to {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export corpus statistics: {e}")
    
    def backup_corpus(self):
        """Create corpus backup"""
        try:
            backup_dir = filedialog.askdirectory(title="Select Backup Directory")
            if not backup_dir:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"uam_corpus_backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            self.status_text.config(text="💾 Creating corpus backup...")
            self.main_progress.start()
            
            def backup_thread():
                try:
                    import shutil
                    os.makedirs(backup_path)
                    
                    # Copy vector database
                    if os.path.exists(Config.VECTOR_DB_PATH):
                        shutil.copytree(Config.VECTOR_DB_PATH, 
                                       os.path.join(backup_path, "vector_db"))
                    
                    # Copy metadata
                    if os.path.exists(Config.METADATA_DB_PATH):
                        shutil.copy2(Config.METADATA_DB_PATH, backup_path)
                    
                    # Create backup info
                    backup_info = {
                        'timestamp': timestamp,
                        'backup_path': backup_path,
                        'corpus_stats': self.rag_system.get_corpus_statistics()
                    }
                    
                    with open(os.path.join(backup_path, 'backup_info.json'), 'w') as f:
                        json.dump(backup_info, f, indent=2)
                    
                    self.root.after(0, self.on_backup_complete, backup_name)
                    
                except Exception as e:
                    self.root.after(0, self.on_backup_error, str(e))
            
            threading.Thread(target=backup_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup: {e}")
    
    def on_backup_complete(self, backup_name):
        """Handle backup completion"""
        self.main_progress.stop()
        self.status_text.config(text=f"✅ Backup created: {backup_name}")
        messagebox.showinfo("Success", f"Corpus backup created successfully:\n{backup_name}")
    
    def on_backup_error(self, error_msg):
        """Handle backup error"""
        self.main_progress.stop()
        self.status_text.config(text="❌ Error creating backup")
        messagebox.showerror("Backup Error", f"Failed to create backup:\n{error_msg}")
    
    def save_enhanced_settings(self):
        """Save enhanced settings including journal impact options"""
        try:
            # Update config values
            Config.TEMPERATURE = self.temp_var.get()
            Config.MAX_TOKENS = self.tokens_var.get()
            Config.RETRIEVAL_K = self.k_var.get()
            Config.MIN_PAPERS_PER_RESPONSE = self.min_papers_var.get()
            Config.MAX_PAPERS_PER_RESPONSE = self.max_papers_var.get()
            Config.JOURNAL_IMPACT_WEIGHT = self.impact_weight_var.get()
            Config.STATISTICAL_CONTENT_WEIGHT = self.stats_weight_var.get()
            Config.JOURNAL_BOOST_MULTIPLIERS['Q1'] = self.q1_boost_var.get()
            
            # Feature toggles
            Config.ENABLE_JOURNAL_RANKING = self.journal_ranking_var.get()
            Config.ENABLE_COMPREHENSIVE_COVERAGE = self.comprehensive_var.get()
            Config.ENABLE_TIER_BALANCING = self.tier_balancing_var.get()
            Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT = self.paper_diversity_var.get()
            Config.ENABLE_Q1_PAPER_PRIORITY = self.q1_priority_var.get()
            Config.ENABLE_HYDE = self.hyde_var.get()
            Config.ENABLE_MULTI_QUERY = self.multi_query_var.get()
            Config.ENABLE_STATISTICAL_EXTRACTION = self.stats_extraction_var.get()
            Config.ENABLE_MULTIMODAL = self.multimodal_var.get()
            
            # Save enhanced settings to file
            enhanced_settings = {
                'temperature': Config.TEMPERATURE,
                'max_tokens': Config.MAX_TOKENS,
                'retrieval_k': Config.RETRIEVAL_K,
                'min_papers_per_response': Config.MIN_PAPERS_PER_RESPONSE,
                'max_papers_per_response': Config.MAX_PAPERS_PER_RESPONSE,
                'journal_impact_weight': Config.JOURNAL_IMPACT_WEIGHT,
                'statistical_content_weight': Config.STATISTICAL_CONTENT_WEIGHT,
                'q1_boost_multiplier': Config.JOURNAL_BOOST_MULTIPLIERS['Q1'],
                'enable_journal_ranking': Config.ENABLE_JOURNAL_RANKING,
                'enable_comprehensive_coverage': Config.ENABLE_COMPREHENSIVE_COVERAGE,
                'enable_tier_balancing': Config.ENABLE_TIER_BALANCING,
                'enable_paper_diversity_enforcement': Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT,
                'enable_q1_paper_priority': Config.ENABLE_Q1_PAPER_PRIORITY,
                'enable_hyde': Config.ENABLE_HYDE,
                'enable_multi_query': Config.ENABLE_MULTI_QUERY,
                'enable_statistical_extraction': Config.ENABLE_STATISTICAL_EXTRACTION,
                'enable_multimodal': Config.ENABLE_MULTIMODAL
            }
            
            with open('uam_enhanced_settings.json', 'w') as f:
                json.dump(enhanced_settings, f, indent=2)
            
            # Reinitialize journal manager if settings changed
            if hasattr(self.rag_system, 'journal_manager'):
                self.rag_system.reload_system()
            
            self.status_text.config(text="✅ Enhanced settings saved successfully")
            messagebox.showinfo("Success", "Enhanced settings saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save enhanced settings: {e}")
    
    def load_enhanced_settings(self):
        """Load enhanced settings including journal impact options"""
        try:
            settings_file = 'uam_enhanced_settings.json'
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                # Load basic settings
                self.temp_var.set(settings.get('temperature', Config.TEMPERATURE))
                self.tokens_var.set(settings.get('max_tokens', Config.MAX_TOKENS))
                self.k_var.set(settings.get('retrieval_k', Config.RETRIEVAL_K))
                self.min_papers_var.set(settings.get('min_papers_per_response', Config.MIN_PAPERS_PER_RESPONSE))
                self.max_papers_var.set(settings.get('max_papers_per_response', Config.MAX_PAPERS_PER_RESPONSE))
                
                # Load journal impact settings
                self.impact_weight_var.set(settings.get('journal_impact_weight', Config.JOURNAL_IMPACT_WEIGHT))
                self.stats_weight_var.set(settings.get('statistical_content_weight', Config.STATISTICAL_CONTENT_WEIGHT))
                self.q1_boost_var.set(settings.get('q1_boost_multiplier', Config.JOURNAL_BOOST_MULTIPLIERS['Q1']))
                
                # Load feature toggles
                self.journal_ranking_var.set(settings.get('enable_journal_ranking', Config.ENABLE_JOURNAL_RANKING))
                self.comprehensive_var.set(settings.get('enable_comprehensive_coverage', Config.ENABLE_COMPREHENSIVE_COVERAGE))
                self.tier_balancing_var.set(settings.get('enable_tier_balancing', Config.ENABLE_TIER_BALANCING))
                self.paper_diversity_var.set(settings.get('enable_paper_diversity_enforcement', Config.ENABLE_PAPER_DIVERSITY_ENFORCEMENT))
                self.q1_priority_var.set(settings.get('enable_q1_paper_priority', Config.ENABLE_Q1_PAPER_PRIORITY))
                self.hyde_var.set(settings.get('enable_hyde', Config.ENABLE_HYDE))
                self.multi_query_var.set(settings.get('enable_multi_query', Config.ENABLE_MULTI_QUERY))
                self.stats_extraction_var.set(settings.get('enable_statistical_extraction', Config.ENABLE_STATISTICAL_EXTRACTION))
                self.multimodal_var.set(settings.get('enable_multimodal', Config.ENABLE_MULTIMODAL))
                
        except Exception as e:
            print(f"Error loading enhanced settings: {e}")
    
    def run(self):
        """Run the application"""
        self.load_enhanced_settings()
        self.root.mainloop()


def main():
    """Main application entry point"""
    try:
        app = UAMLiteratureReviewUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{e}")


if __name__ == "__main__":
    main()