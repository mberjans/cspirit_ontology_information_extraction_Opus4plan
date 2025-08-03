# AIM2 Project: AI-First Ontology Information Extraction

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-orange)]()

## Overview

The AIM2 (AI-Integrated Management and Mining) Project is a comprehensive Python framework for ontology information extraction that takes an **AI-first approach** to automating traditionally manual tasks in scientific literature processing and knowledge graph construction. By leveraging state-of-the-art Large Language Models (LLMs) and advanced NLP techniques, AIM2 eliminates the need for extensive manual annotation and curation.

## Key Features

### 🤖 AI-First Architecture
- **Automated ontology trimming**: Reduce large ontologies (e.g., 2008 anatomical terms → 293 relevant terms) using LLM-based relevance scoring
- **Synthetic data generation**: Replace manual annotation with AI-generated training and testing datasets
- **Intelligent conflict resolution**: Automatically resolve ontology integration conflicts using LLM reasoning
- **Zero-shot entity recognition**: Extract entities without requiring pre-trained domain-specific models

### 🔧 Comprehensive Toolset
- **Multi-source ontology integration**: ChEBI, Plant Ontology, NCBI Taxonomy, Gene Ontology, and more
- **Advanced information extraction**: Ensemble of BERT-based and LLM-based Named Entity Recognition (NER)
- **Relationship extraction**: Hierarchical and context-dependent relationship identification
- **Literature corpus building**: Automated paper collection from PubMed, PMC, and arXiv with API rate limiting
- **Synthetic benchmarking**: Generate gold-standard evaluation datasets using LLMs

### 🏗️ Modular Design
- **Standalone operation**: Each module can run independently or be imported as a library
- **Flexible LLM backends**: Support for local models (Llama) and cloud APIs (GPT-4, Gemma)
- **Multiple ontology formats**: OWL, CSV, JSON-LD support with automatic format detection
- **Configurable pipelines**: YAML-based configuration for all components

## Project Structure

```
aim2_project/
├── aim2_ontology/          # Ontology Development and Management
│   ├── ontology_manager.py     # Main ontology operations coordinator
│   ├── ontology_trimmer.py     # AI-assisted ontology pruning
│   ├── ontology_integrator.py  # Multi-source ontology merger
│   ├── relationship_manager.py # Relationship graph management
│   └── ontology_exporter.py    # Format conversion and export
├── aim2_extraction/        # Information Extraction Pipeline
│   ├── corpus_builder.py       # Literature corpus construction
│   ├── text_processor.py       # Document parsing and preprocessing
│   ├── ner_extractor.py        # Multi-model entity recognition
│   ├── relationship_extractor.py # LLM-based relationship extraction
│   ├── ontology_mapper.py      # Entity-to-ontology mapping
│   └── evaluation_benchmarker.py # Performance evaluation system
├── aim2_utils/             # Core Utilities
│   ├── llm_interface.py        # Unified LLM provider interface
│   ├── synthetic_data_generator.py # AI-generated training data
│   └── config_manager.py       # Configuration management
├── data/                   # Data Storage
│   ├── ontologies/            # Ontology files (OWL, CSV, JSON-LD)
│   ├── corpus/               # Literature corpus
│   ├── synthetic/            # Generated datasets
│   └── benchmarks/           # Evaluation benchmarks
└── configs/               # Configuration Files
    └── default_config.yaml    # Default settings template
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cspirit_ontology_information_extraction_Opus4plan
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r aim2_project/requirements.txt
   ```

4. **Configure the system**:
   ```bash
   cp aim2_project/configs/default_config.yaml config.yaml
   # Edit config.yaml with your LLM API keys and preferences
   ```

## Quick Start

### Standalone Module Execution

```bash
# Trim a large ontology using AI
python -m aim2_ontology.ontology_manager --config config.yaml --trim --target-size 300

# Extract entities from a document collection
python -m aim2_extraction.ner_extractor --input data/corpus/ --output results/

# Generate synthetic training data
python -m aim2_utils.synthetic_data_generator --ontology data/ontologies/chebi.owl --count 100
```

### Library Usage

```python
from aim2_ontology import OntologyManager, OntologyTrimmer
from aim2_extraction import NERExtractor, RelationshipExtractor
from aim2_utils import LLMInterface

# Initialize components
llm = LLMInterface(model='llama-70b', local=True)
ontology_manager = OntologyManager(config_path='config.yaml')
ner = NERExtractor(model_type='ensemble')

# Load and trim ontology
ontology = ontology_manager.load_ontologies()
trimmer = OntologyTrimmer(llm)
trimmed_ontology = trimmer.auto_trim_ontology(ontology, target_size=300)

# Extract information from text
text = "Quercetin is a flavonoid found in onions that exhibits antioxidant properties."
entities = ner.extract_entities_llm(text)
print(f"Extracted entities: {entities}")

# Extract relationships
rel_extractor = RelationshipExtractor(llm)
relationships = rel_extractor.extract_relationships(text, entities)
print(f"Extracted relationships: {relationships}")
```

## Module Descriptions

### 🧬 Ontology Management (`aim2_ontology/`)
- **OntologyManager**: Coordinates loading, integration, and export of multiple ontology sources
- **OntologyTrimmer**: Uses LLMs to intelligently reduce ontology size while preserving relevance
- **OntologyIntegrator**: Merges structural, source, and functional annotation ontologies
- **RelationshipManager**: Manages complex hierarchical relationships and infers missing connections

### 📄 Information Extraction (`aim2_extraction/`)
- **CorpusBuilder**: Automated literature collection with deduplication and metadata management
- **NERExtractor**: Ensemble approach combining BERT-based and LLM-based entity recognition
- **RelationshipExtractor**: Sophisticated relationship extraction using advanced LLM prompting
- **EvaluationBenchmarker**: Comprehensive evaluation system with synthetic gold standards

### 🔧 Utilities (`aim2_utils/`)
- **LLMInterface**: Unified interface supporting multiple LLM providers with rate limiting and caching
- **SyntheticDataGenerator**: Generates realistic training and testing data to replace manual annotation
- **ConfigManager**: YAML-based configuration system with environment variable support

## Advanced Features

### Multi-Model Ensemble
```python
# Use multiple models for higher accuracy
ner = NERExtractor(model_type='ensemble')
entities_bert = ner.extract_entities_bert(text, model='cybert')
entities_llm = ner.extract_entities_llm(text, model='llama-70b')
entities_combined = ner.ensemble_extraction(text)
```

### Batch Processing
```python
# Process large document collections efficiently
corpus_builder = CorpusBuilder()
documents = corpus_builder.search_and_download(['plant metabolites'], limit=1000)
batch_results = ner.batch_process(documents)
```

### Synthetic Benchmark Generation
```python
# Generate evaluation datasets automatically
benchmarker = EvaluationBenchmarker()
gold_standard = benchmarker.generate_gold_standard(n_papers=25)
results = benchmarker.benchmark_models(['llama', 'gpt4', 'gemma'], gold_standard)
```

## Configuration

The system uses YAML configuration files for all settings:

```yaml
# config.yaml
llm:
  default_model: "llama-70b"
  api_keys:
    openai: "your-openai-key"
    google: "your-google-key"
  local_models:
    llama_path: "/path/to/llama/model"

ontology:
  sources:
    - "chebi"
    - "plant_ontology"
    - "ncbi_taxonomy"
  cache_dir: "data/ontologies"

extraction:
  ner_models: ["bert", "llm"]
  confidence_threshold: 0.7
  batch_size: 32
```

## Evaluation and Benchmarking

AIM2 includes comprehensive evaluation capabilities:

- **Automated benchmarking**: Compare multiple models on standardized tasks
- **Synthetic evaluation datasets**: Generate domain-specific test sets
- **Performance metrics**: Precision, recall, F1-score, and custom domain metrics
- **Resource tracking**: Monitor computational costs and processing times

## Implementation Status

The project is currently in active development:

- ✅ **Repository Setup**: Git structure, dependencies, configuration
- ✅ **Basic Infrastructure**: Package structure, imports, configuration system
- 🚧 **Core Modules**: Ontology management and extraction pipelines (in progress)
- ⏳ **Integration Testing**: End-to-end workflow validation (planned)
- ⏳ **Documentation**: API docs and tutorials (planned)

See [docs/checklist.md](docs/checklist.md) for detailed progress tracking.

## Contributing

We welcome contributions to the AIM2 project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and conventions
- Testing requirements
- Pull request process
- Issue reporting

## Research Applications

AIM2 is designed for researchers working on:

- **Plant biology**: Secondary metabolite research, pathway analysis
- **Pharmacology**: Drug discovery, compound-target relationships
- **Systems biology**: Multi-omics integration, pathway reconstruction
- **Knowledge graphs**: Scientific literature mining, ontology construction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on advances in transformer models and large language models
- Integrates multiple public ontology sources (ChEBI, GO, Plant Ontology, etc.)
- Designed for the scientific research community

## Support

For questions, issues, or feature requests:

1. Check the [documentation](docs/)
2. Search existing [issues](../../issues)
3. Create a new issue with detailed information
4. Join our community discussions

---

**Note**: This is a research project under active development. APIs and functionality may change between versions. Please refer to the [roadmap](docs/roadmap.md) for planned features and timeline.