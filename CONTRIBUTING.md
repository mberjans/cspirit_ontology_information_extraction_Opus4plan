# Contributing to AIM2 Project

Welcome to the AIM2 (AI-Integrated Management and Mining) Project! We're excited to have you contribute to this AI-first ontology information extraction framework. This guide will help you get started with contributing to our research project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Style & Standards](#code-style--standards)
3. [Testing Requirements](#testing-requirements)
4. [Git Workflow](#git-workflow)
5. [Issue Reporting](#issue-reporting)
6. [Community Guidelines](#community-guidelines)
7. [Module-Specific Guidelines](#module-specific-guidelines)
8. [AI/LLM Considerations](#aillm-considerations)
9. [Documentation](#documentation)
10. [Review Process](#review-process)

## Development Setup

### Prerequisites

- Python 3.8+ (tested with 3.8, 3.9, 3.10, 3.11)
- Git 2.20+
- 8GB+ RAM (recommended for LLM operations)
- GPU support (optional, for local LLM inference)

### Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/cspirit_ontology_information_extraction_Opus4plan.git
   cd cspirit_ontology_information_extraction_Opus4plan
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r aim2_project/requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Configure the Project**
   ```bash
   cp aim2_project/configs/default_config.yaml config.yaml
   # Edit config.yaml with your API keys and preferences
   ```

5. **Verify Installation**
   ```bash
   python -c "from aim2_utils.config_manager import ConfigManager; print('Setup successful!')"
   ```

### Development Dependencies

Create a `requirements-dev.txt` file with:
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
pre-commit>=3.0.0
sphinx>=5.0.0
jupyter>=1.0.0
```

### IDE Configuration

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm
- Set Python interpreter to `./venv/bin/python`
- Enable pytest as test runner
- Configure Black as code formatter

## Code Style & Standards

### Python Style Guide

We follow PEP 8 with some project-specific conventions:

#### Formatting
- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings, single quotes for dict keys when possible
- **Imports**: Organized using isort with Black-compatible settings

#### Naming Conventions
```python
# Classes: PascalCase
class OntologyManager:
    pass

# Functions and variables: snake_case
def extract_entities_llm(text: str) -> List[Entity]:
    entity_list = []
    return entity_list

# Constants: UPPER_SNAKE_CASE
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 3

# Private methods: _snake_case
def _preprocess_text(self, text: str) -> str:
    pass

# Module-level private: _snake_case
_global_cache = {}
```

#### Type Hints
Use type hints for all function signatures:
```python
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

def process_ontology(
    ontology_path: Path,
    output_format: str = "json",
    include_metadata: bool = True
) -> Dict[str, Union[str, List[str]]]:
    """Process ontology with specified format and options."""
    pass
```

#### Documentation Standards
Use Google-style docstrings:
```python
def trim_ontology(
    self,
    ontology: Ontology,
    target_size: int,
    relevance_threshold: float = 0.7
) -> Ontology:
    """Trim ontology using AI-assisted relevance scoring.
    
    Args:
        ontology: Source ontology to trim
        target_size: Desired number of terms in trimmed ontology
        relevance_threshold: Minimum relevance score for term inclusion
        
    Returns:
        Trimmed ontology with reduced term count
        
    Raises:
        ValueError: If target_size is larger than source ontology
        LLMError: If language model fails to provide relevance scores
        
    Example:
        >>> manager = OntologyManager()
        >>> ontology = manager.load_ontology("chebi.owl")
        >>> trimmed = manager.trim_ontology(ontology, target_size=300)
        >>> len(trimmed.terms) <= 300
        True
    """
```

### Code Organization

#### Project Structure
```
aim2_project/
├── aim2_ontology/          # Ontology management modules
│   ├── __init__.py         # Module exports
│   ├── ontology_manager.py # Main coordinator
│   └── tests/              # Module-specific tests
├── aim2_extraction/        # Information extraction modules
├── aim2_utils/             # Core utilities
└── tests/                  # Integration tests
```

#### Module Design Principles
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Pass dependencies through constructors
- **Configuration-Driven**: Use YAML configs instead of hardcoded values
- **Standalone Capability**: Each module can run independently

## Testing Requirements

### Test Structure
```
tests/
├── unit/                   # Unit tests for individual functions
│   ├── test_ontology_manager.py
│   ├── test_ner_extractor.py
│   └── test_llm_interface.py
├── integration/            # Integration tests for workflows
│   ├── test_ontology_pipeline.py
│   └── test_extraction_pipeline.py
├── fixtures/               # Test data and mocks
│   ├── sample_ontologies/
│   └── mock_responses/
└── conftest.py            # Pytest configuration
```

### Testing Standards

#### Unit Tests
- **Coverage**: Minimum 80% line coverage for new code
- **Isolation**: Use mocks for external dependencies (APIs, file I/O)
- **Fast Execution**: Unit tests should run in <1s each

```python
import pytest
from unittest.mock import Mock, patch
from aim2_utils.llm_interface import LLMInterface

class TestLLMInterface:
    @pytest.fixture
    def llm_interface(self):
        config = {"model": "test-model", "api_key": "test-key"}
        return LLMInterface(config)
    
    @patch('aim2_utils.llm_interface.requests.post')
    def test_generate_response_success(self, mock_post, llm_interface):
        # Arrange
        mock_post.return_value.json.return_value = {"response": "test output"}
        mock_post.return_value.status_code = 200
        
        # Act
        result = llm_interface.generate_response("test prompt")
        
        # Assert
        assert result == "test output"
        mock_post.assert_called_once()
```

#### Integration Tests
- **End-to-End Workflows**: Test complete pipelines
- **Real Data**: Use small sample datasets
- **Resource Management**: Clean up test artifacts

```python
@pytest.mark.integration
def test_ontology_trimming_workflow():
    """Test complete ontology trimming workflow."""
    # Use small test ontology
    test_ontology_path = "tests/fixtures/small_chebi.owl"
    manager = OntologyManager()
    
    # Load, trim, and validate
    ontology = manager.load_ontology(test_ontology_path)
    trimmed = manager.trim_ontology(ontology, target_size=10)
    
    assert len(trimmed.terms) <= 10
    assert all(term.relevance_score >= 0.7 for term in trimmed.terms)
```

#### AI/LLM Testing
- **Mock LLM Responses**: Use deterministic test responses
- **Rate Limiting**: Respect API limits in tests
- **Fallback Testing**: Test behavior when LLM services are unavailable

```python
@pytest.fixture
def mock_llm_responses():
    return {
        "relevance_prompt": {
            "response": "Score: 0.85\nReasoning: High relevance to plant metabolism"
        },
        "entity_extraction": {
            "entities": [{"text": "quercetin", "type": "compound", "confidence": 0.9}]
        }
    }
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aim2_project --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/ -m integration

# Run tests for specific module
pytest tests/unit/test_ontology_manager.py -v
```

## Git Workflow

### Branch Naming Convention

```
feature/AIM2-{ticket-number}-{short-description}
bugfix/AIM2-{ticket-number}-{short-description}
hotfix/AIM2-{ticket-number}-{short-description}
docs/AIM2-{ticket-number}-{short-description}
```

Examples:
- `feature/AIM2-015-llm-interface-implementation`
- `bugfix/AIM2-023-ontology-loading-memory-leak`
- `docs/AIM2-031-api-documentation-update`

### Commit Message Format

Follow Conventional Commits:
```
<type>(scope): <description>

<optional body>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```bash
feat(ontology): implement AI-assisted ontology trimming

Add LLM-based relevance scoring for ontology term filtering.
Reduces ChEBI ontology from 2008 to ~300 relevant terms.

Closes #AIM2-015

fix(extraction): handle empty NER model responses

Add fallback mechanism when BERT model returns no entities.
Prevents pipeline crashes on edge case inputs.

Fixes #AIM2-023
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/AIM2-015-llm-interface-implementation
   ```

2. **Make Changes with Tests**
   - Implement feature with comprehensive tests
   - Update documentation if needed
   - Ensure all tests pass locally

3. **Commit Following Standards**
   ```bash
   git add .
   git commit -m "feat(utils): implement unified LLM interface

   Add support for multiple LLM providers (OpenAI, local Llama, Gemma)
   with automatic rate limiting and response caching.

   Implements #AIM2-015"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/AIM2-015-llm-interface-implementation
   ```

5. **PR Requirements**
   - Clear title and description
   - Link to issue/ticket
   - Screenshots for UI changes
   - Test results summary
   - Breaking change notes (if any)

### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Related Issue
Closes #AIM2-XXX

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] AI/LLM functionality tested with mock responses

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Comments added for complex logic
- [ ] No hardcoded values (uses configuration)
```

## Issue Reporting

### Bug Reports

Use this template for bug reports:
```markdown
## Bug Description
Clear description of what the bug is.

## Environment
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- AIM2 Version: [e.g., 0.1.0]
- LLM Provider: [e.g., OpenAI GPT-4, Local Llama]

## Steps to Reproduce
1. Configure with '...'
2. Run command '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Logs
```
Paste relevant error messages and stack traces
```

## Additional Context
Any other context about the problem.
```

### Feature Requests

```markdown
## Feature Description
Clear description of the proposed feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
Detailed description of how you envision this working.

## Alternative Solutions
Other approaches you've considered.

## AI/LLM Considerations
How does this feature interact with LLM components?

## Additional Context
Any other context or screenshots.
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Collaborative**: Work together towards common goals
- **Be Inclusive**: Welcome newcomers and diverse perspectives
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Professional**: Maintain professional communication

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **Pull Request Reviews**: Code-specific feedback and suggestions
- **Documentation**: Project wiki for design decisions and architecture

### Response Time Expectations

- **Bug Reports**: Initial response within 48 hours
- **Feature Requests**: Review within 1 week
- **Pull Requests**: Review within 3-5 business days
- **Security Issues**: Response within 24 hours

## Module-Specific Guidelines

### Ontology Module (`aim2_ontology/`)

#### Design Principles
- **Format Agnostic**: Support OWL, CSV, JSON-LD formats
- **Memory Efficient**: Handle large ontologies (50K+ terms)
- **Caching**: Cache loaded ontologies to avoid reprocessing

#### Implementation Guidelines
```python
class OntologyManager:
    """Main coordinator for ontology operations."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.cache = {}
        
    def load_ontology(self, source: Union[str, Path]) -> Ontology:
        """Load ontology with automatic format detection."""
        # Implement format detection logic
        # Use caching for performance
        # Handle large files efficiently
```

#### Testing Requirements
- Test with multiple ontology formats
- Memory usage tests for large ontologies
- Integration tests with real ontology sources

### Extraction Module (`aim2_extraction/`)

#### Design Principles
- **Model Agnostic**: Support BERT, LLM, and ensemble approaches
- **Batch Processing**: Efficient handling of large document collections
- **Error Recovery**: Graceful handling of processing failures

#### Implementation Guidelines
```python
class NERExtractor:
    """Multi-model entity recognition."""
    
    def extract_entities_ensemble(
        self, 
        text: str,
        models: List[str] = ["bert", "llm"]
    ) -> List[Entity]:
        """Combine multiple model outputs for higher accuracy."""
        # Implement ensemble logic
        # Handle model failures gracefully
        # Provide confidence scoring
```

### Utils Module (`aim2_utils/`)

#### Design Principles
- **Provider Agnostic**: Unified interface for multiple LLM providers
- **Rate Limiting**: Respect API limits and costs
- **Caching**: Cache responses to improve performance and reduce costs

#### LLM Interface Guidelines
```python
class LLMInterface:
    """Unified interface for LLM providers."""
    
    def __init__(self, provider: str, **kwargs):
        self.provider = self._initialize_provider(provider, **kwargs)
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache()
        
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate response with rate limiting and caching."""
        # Check cache first
        # Apply rate limiting
        # Handle provider-specific errors
```

## AI/LLM Considerations

### Working with Language Models

#### Prompt Engineering
- **Consistent Templates**: Use standardized prompt formats
- **Context Management**: Handle context window limitations
- **Error Handling**: Graceful degradation when LLM fails

```python
# Example prompt template
ENTITY_EXTRACTION_PROMPT = """
Extract named entities from the following text related to {domain}.

Text: {text}

Return entities in JSON format:
{
  "entities": [
    {"text": "entity_text", "type": "entity_type", "confidence": 0.95}
  ]
}

Entity types: {entity_types}
"""
```

#### Response Validation
- **Schema Validation**: Validate LLM JSON responses
- **Confidence Scoring**: Include confidence in AI decisions
- **Fallback Mechanisms**: Handle malformed responses

```python
def validate_llm_response(response: str, schema: Dict) -> bool:
    """Validate LLM response against expected schema."""
    try:
        parsed = json.loads(response)
        # Validate against schema
        return True
    except (json.JSONDecodeError, ValidationError):
        return False
```

### Synthetic Data Generation

#### Quality Guidelines
- **Diversity**: Generate diverse examples covering edge cases
- **Realism**: Ensure synthetic data resembles real scientific text
- **Validation**: Validate synthetic data quality metrics

```python
class SyntheticDataGenerator:
    def generate_training_examples(
        self,
        ontology: Ontology,
        count: int,
        domain: str = "plant_biology"
    ) -> List[TrainingExample]:
        """Generate realistic training examples for NER models."""
        # Use domain-specific templates
        # Ensure entity type coverage
        # Validate output quality
```

### Cost Management

#### API Usage Optimization
- **Caching**: Cache LLM responses to avoid duplicate calls
- **Batch Processing**: Group similar requests when possible
- **Model Selection**: Use appropriate model size for task complexity

```python
# Cost-aware model selection
def select_optimal_model(task_complexity: str, budget: float) -> str:
    """Select most cost-effective model for task."""
    if task_complexity == "simple" and budget < 0.01:
        return "gpt-3.5-turbo"
    elif task_complexity == "complex":
        return "gpt-4"
    return "llama-70b-local"
```

## Documentation

### API Documentation
- Use Sphinx for generating API docs
- Include examples in docstrings
- Document configuration options

### User Guides
- Provide step-by-step tutorials
- Include common use cases
- Show integration examples

### Architecture Documentation
- Document design decisions
- Explain module interactions
- Provide diagrams for complex workflows

## Review Process

### Code Review Checklist

#### Functionality
- [ ] Code accomplishes stated requirements
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive
- [ ] Performance is acceptable for expected load

#### Code Quality
- [ ] Code follows project style guidelines
- [ ] Functions are properly documented
- [ ] Complex logic is commented
- [ ] No code duplication

#### Testing
- [ ] Unit tests cover new functionality
- [ ] Integration tests pass
- [ ] Test coverage meets requirements
- [ ] Mock objects used appropriately for external dependencies

#### AI/LLM Specific
- [ ] LLM responses are validated
- [ ] Rate limiting is implemented
- [ ] Costs are considered and optimized
- [ ] Fallback mechanisms exist for LLM failures

#### Security
- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] Dependencies are up to date
- [ ] No sensitive data in logs

### Review Process
1. **Automated Checks**: CI runs tests and linting
2. **Peer Review**: At least one team member reviews code
3. **Maintainer Approval**: Project maintainer approves for merge
4. **Documentation Review**: Technical writer reviews documentation changes

## Getting Help

### Resources
- **Documentation**: Check `docs/` directory first
- **Examples**: See `examples/` directory for usage patterns
- **API Reference**: Generated docs at `docs/api/`

### Support Channels
- **GitHub Issues**: Technical questions and bug reports
- **Discussions**: General questions and feature discussions
- **Email**: Maintainer contact for sensitive issues

### Contributing Your First Change
1. Start with "good first issue" labels
2. Ask questions in issue comments
3. Submit small changes first to get familiar with process
4. Review existing code to understand patterns

---

Thank you for contributing to the AIM2 Project! Your contributions help advance AI-first approaches to scientific literature processing and knowledge graph construction.