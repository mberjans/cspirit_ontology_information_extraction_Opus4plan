# AIM2 Configuration System Documentation

## Table of Contents

1. [Overview](#overview)
2. [Configuration Architecture](#configuration-architecture)
3. [Configuration File Structure](#configuration-file-structure)
4. [Environment Variable System](#environment-variable-system)
5. [Configuration Validation](#configuration-validation)
6. [Configuration Merging](#configuration-merging)
7. [Hot Reload and File Watching](#hot-reload-and-file-watching)
8. [Usage Examples](#usage-examples)
9. [Configuration Sections Reference](#configuration-sections-reference)
10. [Schema System](#schema-system)
11. [Migration System](#migration-system)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

## Overview

The AIM2 project features a sophisticated configuration management system designed for flexibility, validation, and security. The system supports multiple configuration sources, environment variable overrides, schema-based validation, and real-time configuration updates.

### Key Features

- **Multi-format Support**: YAML and JSON configuration files
- **Environment Variable Overrides**: Secure configuration via environment variables with `AIM2_` prefix
- **Schema-based Validation**: Built-in validation schemas with custom rules and constraints
- **Configuration Merging**: Intelligent merging from multiple configuration sources
- **Hot Reload**: Real-time configuration updates with file watching
- **Migration System**: Schema versioning and automatic configuration migration
- **Security**: Support for encrypted configuration values and secure defaults

## Configuration Architecture

The configuration system consists of four main components:

### 1. ConfigManager
The main interface for configuration management providing:
- Configuration loading from YAML/JSON files
- Environment variable override processing
- Configuration validation and merging
- Hot reload capabilities with file watching
- Backup and restore functionality

### 2. ConfigValidator
Advanced validation system with:
- Schema-based validation with type checking
- Custom validation rules and constraints
- Nested configuration validation
- Detailed error reporting and warnings
- Support for multiple validation schemas

### 3. Default Configuration Template
Comprehensive YAML template (`default_config.yaml`) covering:
- Project metadata and database settings
- API and logging configurations
- NLP model settings and ontology processing
- LLM interfaces and synthetic data generation
- Evaluation metrics and security settings

### 4. Schema Registry
Centralized schema management with:
- Built-in schemas for common configurations
- External schema loading from files
- Schema versioning and migration support
- Runtime schema registration

## Configuration File Structure

### Primary Configuration File

The main configuration file is `aim2_project/configs/default_config.yaml`:

```yaml
# Project metadata
project:
  name: "AIM2 Ontology Information Extraction"
  version: "1.0.0"
  description: "Comprehensive ontology information extraction system"

# Database configuration
database:
  host: localhost
  port: 5432
  name: aim2_db
  username: aim2_user
  password: ""  # Should be set via environment variable
  ssl_mode: prefer
  pool_size: 5
  timeout: 30

# Additional sections...
```

### Configuration Search Paths

The ConfigManager searches for configuration files in the following order:
1. Explicitly provided path
2. `aim2_project/configs/default_config.yaml`
3. `configs/default_config.yaml` (relative to current directory)
4. Current working directory

## Environment Variable System

### Environment Variable Prefix

All environment variables use the prefix `AIM2_` followed by the configuration path:

```bash
# Format: AIM2_<SECTION>_<FIELD>
AIM2_DATABASE_HOST=localhost
AIM2_DATABASE_PORT=5432
AIM2_API_TIMEOUT=60
AIM2_LOGGING_LEVEL=DEBUG
```

### Nested Configuration Override

For nested configurations, use underscores to separate levels:

```bash
# For nlp.models.ner configuration
AIM2_NLP_MODELS_NER=bert-large-uncased

# For corpus.pubmed.api_key configuration
AIM2_CORPUS_PUBMED_API_KEY=your_api_key_here
```

### Type Conversion

Environment variables are automatically converted to appropriate types:

- `"true"/"false"` → Boolean
- `"null"/"none"` → None
- Numeric strings → int or float
- Other strings → string

### Example Environment Setup

```bash
# Database configuration
export AIM2_DATABASE_HOST=production-db.example.com
export AIM2_DATABASE_PORT=5432
export AIM2_DATABASE_PASSWORD=secure_password_here

# API keys (sensitive data)
export AIM2_CORPUS_PUBMED_API_KEY=your_pubmed_api_key
export AIM2_LLM_MODELS_GPT4_API_KEY=your_openai_api_key

# Feature flags
export AIM2_FEATURES_DEBUG_MODE=true
export AIM2_FEATURES_ENABLE_CACHING=false
```

## Configuration Validation

### Built-in Validation Schemas

The system includes several built-in schemas:

1. **aim2_project**: Complete AIM2 project schema
2. **database_only**: Database configuration validation
3. **api_only**: API configuration validation

### Schema-based Validation

```python
from aim2_utils.config_manager import ConfigManager

# Initialize config manager
config_manager = ConfigManager()
config_manager.load_default_config()

# Validate against AIM2 project schema
is_valid = config_manager.validate_with_aim2_schema()

# Get detailed validation report
report = config_manager.validate_with_aim2_schema(return_report=True)
print(report)
```

### Custom Validation Rules

```python
from aim2_utils.config_validator import ConfigValidator

validator = ConfigValidator()

# Add custom validation rule
def validate_positive_integer(value):
    return isinstance(value, int) and value > 0

validator.add_validation_rule(
    "positive_integer",
    validate_positive_integer,
    "Value must be a positive integer"
)
```

### Validation Constraints

The validation system supports various constraint types:

- **Type constraints**: `str`, `int`, `float`, `bool`, `list`, `dict`
- **Range constraints**: `min`, `max` for numeric values
- **Length constraints**: `min_length`, `max_length` for strings
- **Choice constraints**: `choices` for enumerated values
- **Custom validators**: User-defined validation functions

## Configuration Merging

### Merge Strategies

The ConfigManager supports intelligent configuration merging:

```python
# Load multiple configuration files with merging
config_paths = [
    "configs/base_config.yaml",
    "configs/production_config.yaml",
    "configs/local_overrides.yaml"
]
config_manager.load_configs(config_paths)
```

### Merge Behavior

- **Nested dictionaries**: Recursively merged
- **Lists**: Replaced entirely (not merged)
- **Primitive values**: Later values override earlier ones
- **Environment variables**: Always take highest precedence

### Example Merge Operation

```yaml
# base_config.yaml
database:
  host: localhost
  port: 5432
  pool_size: 5

# production_config.yaml
database:
  host: prod-db.example.com
  ssl_mode: require

# Result after merging
database:
  host: prod-db.example.com  # from production_config.yaml
  port: 5432                 # from base_config.yaml
  pool_size: 5              # from base_config.yaml
  ssl_mode: require         # from production_config.yaml
```

## Hot Reload and File Watching

### Enable File Watching

```python
from aim2_utils.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_default_config()

# Enable file watching for automatic reloads
config_manager.enable_watch_mode("configs/default_config.yaml")

# Watch multiple files
config_manager.add_watch_path("configs/production_overrides.yaml")

# Watch entire directory
config_manager.watch_config_directory("configs/", recursive=True)
```

### Watch Configuration Options

```python
# Set custom debounce delay (default: 0.5 seconds)
config_manager.set_debounce_delay(1.0)

# Get currently watched paths
watched_paths = config_manager.get_watched_paths()

# Check if watching is enabled
is_watching = config_manager.is_watching()

# Disable watching
config_manager.disable_watch_mode()
```

### File Watching Features

- **Debouncing**: Prevents rapid successive reloads
- **Multiple file support**: Watch files and directories simultaneously
- **Recursive directory watching**: Monitor subdirectories for changes
- **Error resilience**: Continues watching even if reload fails
- **Cross-platform support**: Uses `watchdog` library for efficient monitoring

## Usage Examples

### Basic Configuration Setup

```python
from aim2_utils.config_manager import ConfigManager

# Initialize and load default configuration
config_manager = ConfigManager()
config_manager.load_default_config()

# Access configuration values
db_host = config_manager.get("database.host")
api_timeout = config_manager.get("api.timeout", default=30)

# Get nested configuration
nlp_models = config_manager.get("nlp.models")
```

### Environment Variable Configuration

```python
import os
from aim2_utils.config_manager import ConfigManager

# Set environment variables
os.environ["AIM2_DATABASE_HOST"] = "production-db.example.com"
os.environ["AIM2_DATABASE_PASSWORD"] = "secure_password"
os.environ["AIM2_FEATURES_DEBUG_MODE"] = "true"

# Load configuration (environment variables will override defaults)
config_manager = ConfigManager()
config_manager.load_default_config()

# Verify environment overrides
print(config_manager.get("database.host"))  # production-db.example.com
print(config_manager.get("features.debug_mode"))  # True
```

### Configuration Validation

```python
from aim2_utils.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_default_config()

# Validate current configuration
try:
    config_manager.validate_current_config(strict=True)
    print("Configuration is valid!")
except Exception as e:
    print(f"Configuration validation failed: {e}")

# Get detailed validation report
report = config_manager.validate_current_config(return_report=True)
if not report.is_valid:
    print("Validation errors:")
    for error in report.errors:
        print(f"  - {error}")
```

### Custom Schema Loading

```python
from aim2_utils.config_manager import ConfigManager

config_manager = ConfigManager()

# Load external schema
config_manager.load_validation_schema(
    "schemas/custom_schema.yaml",
    name="custom_validation",
    version="1.0.0"
)

# Validate against custom schema
is_valid = config_manager.validate_with_schema(
    "custom_validation",
    strict=True
)
```

### Configuration Export

```python
from aim2_utils.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_default_config()

# Export current configuration to YAML
config_manager.export_config("output/current_config.yaml", format="yaml")

# Export to JSON
config_manager.export_config("output/current_config.json", format="json")
```

## Configuration Sections Reference

### Project Section

```yaml
project:
  name: "AIM2 Ontology Information Extraction"     # Project name
  version: "1.0.0"                                 # Project version
  description: "Comprehensive ontology information extraction system"
```

**Environment Variables**: `AIM2_PROJECT_NAME`, `AIM2_PROJECT_VERSION`

### Database Section

```yaml
database:
  host: localhost          # Database server hostname
  port: 5432              # Database server port
  name: aim2_db           # Database name
  username: aim2_user     # Database username
  password: ""            # Database password (use env var)
  ssl_mode: prefer        # SSL connection mode
  pool_size: 5           # Connection pool size
  timeout: 30            # Connection timeout in seconds
```

**Environment Variables**: `AIM2_DATABASE_HOST`, `AIM2_DATABASE_PORT`, etc.

**Validation Constraints**:
- `port`: 1-65535
- `pool_size`: 1-100
- `timeout`: 1-300 seconds
- `ssl_mode`: disable, allow, prefer, require

### API Section

```yaml
api:
  base_url: "https://api.example.com"    # API base URL
  version: "v1"                          # API version
  timeout: 30                            # Request timeout in seconds
  retry_attempts: 3                      # Number of retry attempts
  rate_limit: 1000                       # Requests per hour
  headers:                               # Default headers
    User-Agent: "AIM2-Client/1.0"
```

**Environment Variables**: `AIM2_API_BASE_URL`, `AIM2_API_TIMEOUT`, etc.

### Logging Section

```yaml
logging:
  level: INFO                                    # Log level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:                                      # Log handlers
    - console
  file_path: null                               # Log file path (optional)
  max_file_size: "10MB"                        # Maximum log file size
  backup_count: 3                              # Number of backup files
```

**Environment Variables**: `AIM2_LOGGING_LEVEL`, `AIM2_LOGGING_FILE_PATH`

**Valid Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### NLP Section

```yaml
nlp:
  models:                                       # Model configurations
    ner: "bert-base-uncased"                   # Named entity recognition model
    relationship: "roberta-base"                # Relationship extraction model
    embedding: "sentence-transformers/all-MiniLM-L6-v2"
  max_sequence_length: 512                     # Maximum sequence length
  batch_size: 16                              # Processing batch size
  cache_dir: "/tmp/aim2_models"               # Model cache directory
  device: "auto"                              # Compute device (auto/cpu/cuda)
```

**Environment Variables**: `AIM2_NLP_MAX_SEQUENCE_LENGTH`, `AIM2_NLP_DEVICE`

### Ontology Section

```yaml
ontology:
  default_namespace: "http://aim2.example.com/ontology#"
  import_paths:                               # Ontology import paths
    - "./data/ontologies"
  export_formats:                             # Supported export formats
    - "owl"
    - "rdf"
  validation:
    strict_mode: false                        # Enable strict validation
    check_consistency: true                   # Check ontology consistency

  sources:                                    # External ontology sources
    chebi:
      enabled: true
      url: "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl"
      local_path: "./data/ontologies/chebi.owl"
      update_frequency: "weekly"
      include_deprecated: false
```

### Features Section

```yaml
features:
  enable_caching: true        # Enable result caching
  cache_ttl: 3600            # Cache time-to-live in seconds
  enable_metrics: false      # Enable performance metrics
  debug_mode: false          # Enable debug mode
  async_processing: true     # Enable asynchronous processing
  max_workers: 2            # Maximum worker threads
```

**Environment Variables**: `AIM2_FEATURES_DEBUG_MODE`, `AIM2_FEATURES_ENABLE_CACHING`

### Corpus Section

```yaml
corpus:
  pubmed:                                     # PubMed configuration
    enabled: true
    api_key: ""                              # Set via AIM2_CORPUS_PUBMED_API_KEY
    base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email: ""                                # Required for API access
    rate_limit: 10                           # Requests per second
    max_records: 10000

  pmc:                                        # PMC configuration
    enabled: true
    base_url: "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    rate_limit: 3
    full_text: true

  arxiv:                                      # ArXiv configuration
    enabled: true
    base_url: "http://export.arxiv.org/api/query"
    rate_limit: 1
    categories:
      - "q-bio"
      - "physics.bio-ph"
```

### LLM Section

```yaml
llm:
  provider: "openai"                          # Default provider
  model: "gpt-3.5-turbo"                     # Default model
  max_tokens: 2048                           # Maximum tokens
  temperature: 0.1                           # Generation temperature
  timeout: 60                                # Request timeout
  retry_attempts: 3                          # Retry attempts

  models:                                     # Model-specific configurations
    gpt4:
      provider: "openai"
      model_name: "gpt-4-turbo-preview"
      api_key: ""                            # Set via AIM2_LLM_MODELS_GPT4_API_KEY
      max_tokens: 4096
      use_for:
        - "complex_reasoning"
        - "ontology_mapping"
```

### Security Section

```yaml
security:
  encrypt_config: false                       # Enable configuration encryption
  allowed_hosts: []                          # Allowed host list
  cors_enabled: false                        # Enable CORS
```

## Schema System

### Built-in Schemas

The configuration system includes several built-in validation schemas:

1. **AIM2 Project Schema**: Complete validation for all configuration sections
2. **Database Schema**: Specialized validation for database configurations
3. **API Schema**: Validation for API-related settings

### Schema Files

- `aim2_project/configs/database_schema.yaml`: Database configuration schema
- `aim2_project/configs/nlp_schema.json`: NLP configuration schema

### Loading External Schemas

```python
from aim2_utils.config_manager import ConfigManager

config_manager = ConfigManager()

# Load schema from external file
config_manager.load_validation_schema(
    "path/to/custom_schema.yaml",
    name="custom_schema",
    version="1.0.0"
)

# List available schemas
schemas = config_manager.get_available_schemas()
print(schemas)  # ['aim2_project', 'database_only', 'api_only', 'custom_schema']
```

### Schema Definition Format

```yaml
# Example schema definition
section_name:
  required:                    # Required fields
    - field1
    - field2
  types:                      # Field types
    field1: str
    field2: int
    field3: [str, null]       # Union types
  constraints:                # Field constraints
    field2:
      min: 1
      max: 100
    field1:
      min_length: 1
      choices: ["option1", "option2"]
  warnings:                   # Deprecation warnings
    old_field: "This field is deprecated"
  nested:                     # Nested schema definitions
    nested_section:
      types:
        nested_field: bool
```

## Migration System

### Schema Versioning

The configuration system supports schema versioning and automatic migration:

```python
from aim2_utils.config_validator import ConfigValidator

validator = ConfigValidator()

# Check latest schema version
latest_version = validator.get_latest_schema_version("aim2_project")

# Migrate configuration to latest version
migrated_config = validator.migrate_config(
    config_dict,
    schema_name="aim2_project",
    from_version="1.0.0",
    to_version="1.1.0"
)
```

### Automatic Migration

```python
# Validate and migrate if necessary
result = validator.validate_and_migrate(
    config_dict,
    schema_name="aim2_project",
    target_version="latest"
)

if isinstance(result, tuple):
    migrated_config, validation_result = result
    print("Configuration was migrated")
else:
    validation_result = result
    print("No migration needed")
```

### Custom Migrations

```python
from aim2_utils.config_validator import SchemaMigration

def migrate_1_0_to_1_1(config):
    """Custom migration function"""
    migrated = config.copy()

    # Add new security section if missing
    if "security" not in migrated:
        migrated["security"] = {
            "encrypt_config": False,
            "allowed_hosts": [],
            "cors_enabled": False
        }

    return migrated

# Create and register migration
migration = SchemaMigration(
    from_version="1.0.0",
    to_version="1.1.0",
    migration_function=migrate_1_0_to_1_1,
    description="Add security section"
)

validator.register_schema_migration("custom_schema", migration)
```

## Best Practices

### Security Best Practices

1. **Never commit sensitive data** to configuration files
2. **Use environment variables** for API keys, passwords, and secrets:
   ```bash
   export AIM2_DATABASE_PASSWORD=your_secure_password
   export AIM2_CORPUS_PUBMED_API_KEY=your_api_key
   ```

3. **Enable configuration encryption** for production environments:
   ```yaml
   security:
     encrypt_config: true
   ```

4. **Use SSL/TLS** for database connections:
   ```yaml
   database:
     ssl_mode: require
   ```

### Configuration Organization

1. **Use layered configurations**:
   - Base configuration with defaults
   - Environment-specific overrides
   - Local development settings

2. **Keep configurations DRY** (Don't Repeat Yourself):
   ```yaml
   # Use YAML anchors for common values
   common: &common_settings
     timeout: 30
     retry_attempts: 3

   api:
     <<: *common_settings
     base_url: "https://api.example.com"
   ```

3. **Validate configurations** in CI/CD pipelines:
   ```python
   # In your test suite
   def test_configuration_valid():
       config_manager = ConfigManager()
       config_manager.load_config("configs/production.yaml")
       assert config_manager.validate_current_config()
   ```

### Performance Optimization

1. **Enable caching** for frequently accessed configurations:
   ```yaml
   features:
     enable_caching: true
     cache_ttl: 3600
   ```

2. **Use appropriate batch sizes** for processing:
   ```yaml
   nlp:
     batch_size: 32  # Adjust based on available memory
   data:
     batch_size: 1000
   ```

3. **Configure connection pooling**:
   ```yaml
   database:
     pool_size: 10  # Adjust based on expected load
   ```

### Development Workflow

1. **Use file watching** during development:
   ```python
   config_manager.enable_watch_mode("configs/dev_config.yaml")
   ```

2. **Test configuration changes** before deployment:
   ```python
   # Create backup before changes
   backup_id = config_manager.create_backup()

   try:
       # Load new configuration
       config_manager.load_config("new_config.yaml")
       config_manager.validate_current_config()
   except Exception:
       # Restore backup if validation fails
       config_manager.restore_backup(backup_id)
   ```

3. **Use schema validation** to catch errors early:
   ```python
   # Validate before deploying
   report = config_manager.validate_current_config(
       strict=True,
       return_report=True
   )

   if not report.is_valid:
       print("Configuration errors found:")
       for error in report.errors:
           print(f"  - {error}")
   ```

## Troubleshooting

### Common Issues

#### 1. Configuration File Not Found

**Problem**: `ConfigError: File not found: config.yaml`

**Solutions**:
- Verify the file path is correct and accessible
- Check file permissions
- Use absolute paths to avoid path resolution issues
- Ensure the file exists in one of the search paths

```python
# Debug configuration loading
import os
from pathlib import Path

config_path = "configs/default_config.yaml"
print(f"Current directory: {os.getcwd()}")
print(f"Config path exists: {Path(config_path).exists()}")
print(f"Config path absolute: {Path(config_path).resolve()}")
```

#### 2. Environment Variable Override Not Working

**Problem**: Environment variables not overriding configuration values

**Solutions**:
- Verify the environment variable name follows the `AIM2_SECTION_FIELD` format
- Check that the environment variable is set in the current process
- Ensure the configuration is loaded after setting environment variables

```python
# Debug environment variables
import os

# Check if environment variable is set
env_var = "AIM2_DATABASE_HOST"
if env_var in os.environ:
    print(f"{env_var} = {os.environ[env_var]}")
else:
    print(f"{env_var} is not set")

# List all AIM2 environment variables
aim2_vars = {k: v for k, v in os.environ.items() if k.startswith("AIM2_")}
print("AIM2 environment variables:", aim2_vars)
```

#### 3. Validation Errors

**Problem**: Configuration validation fails with unclear errors

**Solutions**:
- Use detailed validation reports to understand specific issues
- Check field types and constraints
- Verify required fields are present

```python
# Get detailed validation report
report = config_manager.validate_current_config(return_report=True)

print(f"Validation status: {report.is_valid}")
print(f"Errors: {len(report.errors)}")
print(f"Warnings: {len(report.warnings)}")

if not report.is_valid:
    print("\nDetailed errors:")
    for error in report.errors:
        print(f"  - {error}")

if report.warnings:
    print("\nWarnings:")
    for warning in report.warnings:
        print(f"  - {warning}")
```

#### 4. File Watching Issues

**Problem**: Hot reload not working or causing errors

**Solutions**:
- Check file permissions for the watched files/directories
- Verify the `watchdog` library is installed
- Ensure the file path is accessible
- Check for filesystem-specific limitations

```python
# Debug file watching
config_manager = ConfigManager()

try:
    config_manager.enable_watch_mode("configs/default_config.yaml")
    print("File watching enabled successfully")
    print(f"Watched paths: {config_manager.get_watched_paths()}")
    print(f"Is watching: {config_manager.is_watching()}")
except Exception as e:
    print(f"File watching failed: {e}")
```

#### 5. Schema Loading Errors

**Problem**: External schema files fail to load

**Solutions**:
- Verify schema file format (YAML or JSON)
- Check schema syntax and structure
- Ensure schema follows the expected format

```python
# Debug schema loading
from aim2_utils.config_validator import ConfigValidator

validator = ConfigValidator()

try:
    validator.load_and_register_schema(
        "schemas/custom_schema.yaml",
        name="custom"
    )
    print("Schema loaded successfully")
    print(f"Available schemas: {validator.get_available_schemas()}")
except Exception as e:
    print(f"Schema loading failed: {e}")
```

### Debug Configuration

To enable debug logging for configuration operations:

```python
import logging

# Enable debug logging
logging.getLogger("aim2_utils.config_manager").setLevel(logging.DEBUG)
logging.getLogger("aim2_utils.config_validator").setLevel(logging.DEBUG)

# Create handler if needed
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to loggers
logging.getLogger("aim2_utils.config_manager").addHandler(handler)
logging.getLogger("aim2_utils.config_validator").addHandler(handler)
```

### Configuration Testing

Create test configurations to verify your setup:

```python
# test_config.py
import pytest
from aim2_utils.config_manager import ConfigManager

def test_default_config_loads():
    """Test that default configuration loads successfully"""
    config_manager = ConfigManager()
    config_manager.load_default_config()

    # Verify basic sections exist
    assert config_manager.get("project.name") is not None
    assert config_manager.get("database.host") is not None

def test_environment_override():
    """Test environment variable override"""
    import os

    # Set test environment variable
    os.environ["AIM2_DATABASE_HOST"] = "test-host"

    config_manager = ConfigManager()
    config_manager.load_default_config()

    assert config_manager.get("database.host") == "test-host"

    # Clean up
    del os.environ["AIM2_DATABASE_HOST"]

def test_validation():
    """Test configuration validation"""
    config_manager = ConfigManager()
    config_manager.load_default_config()

    # Should validate successfully
    assert config_manager.validate_current_config()

if __name__ == "__main__":
    test_default_config_loads()
    test_environment_override()
    test_validation()
    print("All configuration tests passed!")
```

### Getting Help

If you encounter issues not covered in this troubleshooting guide:

1. **Check the logs** for detailed error messages
2. **Verify your configuration** against the schema definitions
3. **Test with minimal configurations** to isolate issues
4. **Check file permissions** and access rights
5. **Ensure all dependencies** are properly installed

The configuration system provides extensive logging and error reporting to help diagnose issues. Use the validation reports and debug logging to understand and resolve configuration problems quickly.
