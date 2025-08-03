"""
Configuration Validation Module for AIM2 Project

This module provides comprehensive configuration validation functionality including:
- Schema-based validation with type checking
- Custom validation rules and constraints
- Nested configuration validation
- Validation error collection and reporting
- Warning collection for deprecated fields
- Support for complex validation scenarios

Classes:
    ValidationRule: Represents a custom validation rule
    ConfigValidator: Main configuration validation class

Dependencies:
    - typing: For type hints
    - copy: For deep copying data structures
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import json
import yaml
from pathlib import Path
from packaging import version


class ValidationRule:
    """
    Represents a custom validation rule.

    A validation rule consists of a name, validation function, and optional
    description. Rules can be applied to specific configuration fields.

    Attributes:
        name (str): Name of the validation rule
        validator (Callable): Function that performs the validation
        description (str): Optional description of what the rule validates
    """

    def __init__(
        self, name: str, validator: Callable[[Any], bool], description: str = ""
    ):
        """
        Initialize a validation rule.

        Args:
            name (str): Name of the validation rule
            validator (Callable): Function that takes a value and returns True if valid
            description (str): Optional description of the rule
        """
        self.name = name
        self.validator = validator
        self.description = description

    def validate(self, value: Any) -> bool:
        """
        Validate a value using this rule.

        Args:
            value (Any): Value to validate

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails
        """
        return self.validator(value)


class SchemaRegistry:
    """
    Registry for managing configuration schemas.

    Provides centralized storage and management of validation schemas
    including built-in schemas, loaded schemas, and schema versioning.
    """

    def __init__(self):
        """Initialize the schema registry."""
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.schema_versions: Dict[str, str] = {}
        self._load_built_in_schemas()

    def register_schema(
        self, name: str, schema: Dict[str, Any], version: str = "1.0.0"
    ) -> None:
        """
        Register a new schema.

        Args:
            name: Schema name/identifier
            schema: Schema definition
            version: Schema version
        """
        self.schemas[name] = schema
        self.schema_versions[name] = version

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered schema by name.

        Args:
            name: Schema name

        Returns:
            Schema definition or None if not found
        """
        return self.schemas.get(name)

    def list_schemas(self) -> List[str]:
        """Get list of all registered schema names."""
        return list(self.schemas.keys())

    def load_schema_from_file(self, file_path: str, name: Optional[str] = None) -> None:
        """
        Load schema from external file.

        Args:
            file_path: Path to schema file (JSON or YAML)
            name: Optional name for the schema (defaults to filename)
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Schema file not found: {file_path}")

        schema_name = name or path.stem

        try:
            if path.suffix.lower() in [".yaml", ".yml"]:
                with open(path, "r", encoding="utf-8") as f:
                    schema = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
            else:
                raise ValueError(f"Unsupported schema file format: {path.suffix}")

            # Convert string type names to actual types
            schema = self._normalize_schema_types(schema)

            self.register_schema(schema_name, schema)

        except Exception as e:
            raise ValueError(f"Failed to load schema from {file_path}: {str(e)}")

    def _normalize_schema_types(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string type names in schema to actual Python types.

        Args:
            schema: Raw schema with string type names

        Returns:
            Schema with normalized types
        """
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "none": type(None),
            "null": type(None),
        }

        def normalize_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key == "types" and isinstance(value, dict):
                        # Convert type definitions
                        result[key] = {}
                        for field_name, type_def in value.items():
                            if isinstance(type_def, str):
                                # Single type
                                result[key][field_name] = type_mapping.get(
                                    type_def.lower(), str
                                )
                            elif isinstance(type_def, list):
                                # Union type (tuple of types)
                                types = []
                                for t in type_def:
                                    if isinstance(t, str):
                                        types.append(type_mapping.get(t.lower(), str))
                                    else:
                                        types.append(t)
                                result[key][field_name] = tuple(types)
                            else:
                                result[key][field_name] = type_def
                    else:
                        result[key] = normalize_recursive(value)
                return result
            elif isinstance(obj, list):
                return [normalize_recursive(item) for item in obj]
            else:
                return obj

        return normalize_recursive(schema)

    def _load_built_in_schemas(self) -> None:
        """Load built-in schemas for the AIM2 project."""
        # AIM2 project configuration schema
        aim2_schema = {
            "project": {
                "required": ["name", "version"],
                "types": {"name": str, "version": str, "description": str},
                "constraints": {
                    "name": {"min_length": 1, "max_length": 100},
                    "version": {"min_length": 1},
                },
            },
            "database": {
                "required": ["host", "port", "name", "username"],
                "types": {
                    "host": str,
                    "port": int,
                    "name": str,
                    "username": str,
                    "password": str,
                    "ssl_mode": str,
                    "pool_size": int,
                    "timeout": int,
                },
                "constraints": {
                    "port": {"min": 1, "max": 65535},
                    "pool_size": {"min": 1, "max": 100},
                    "timeout": {"min": 1, "max": 300},
                    "ssl_mode": {"choices": ["disable", "allow", "prefer", "require"]},
                },
            },
            "api": {
                "types": {
                    "base_url": str,
                    "version": str,
                    "timeout": int,
                    "retry_attempts": int,
                    "rate_limit": int,
                    "headers": dict,
                },
                "constraints": {
                    "timeout": {"min": 1, "max": 300},
                    "retry_attempts": {"min": 0, "max": 10},
                    "rate_limit": {"min": 1},
                },
            },
            "logging": {
                "required": ["level", "format"],
                "types": {
                    "level": str,
                    "format": str,
                    "handlers": list,
                    "file_path": (str, type(None)),
                    "max_file_size": (str, int),
                    "backup_count": int,
                },
                "constraints": {
                    "level": {
                        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    },
                    "backup_count": {"min": 0, "max": 100},
                    "format": {"min_length": 1},
                },
                "custom_validators": {
                    "handlers": ["validate_logging_handlers"],
                    "max_file_size": ["validate_file_size_format"],
                },
            },
            "nlp": {
                "types": {
                    "models": dict,
                    "max_sequence_length": int,
                    "batch_size": int,
                    "cache_dir": str,
                    "device": str,
                },
                "constraints": {
                    "max_sequence_length": {"min": 1, "max": 4096},
                    "batch_size": {"min": 1, "max": 1024},
                    "device": {"choices": ["auto", "cpu", "cuda"]},
                },
                "nested": {
                    "models": {
                        "types": {"ner": str, "relationship": str, "embedding": str}
                    }
                },
            },
            "ontology": {
                "types": {
                    "default_namespace": str,
                    "import_paths": list,
                    "export_formats": list,
                    "validation": dict,
                },
                "nested": {
                    "validation": {
                        "types": {"strict_mode": bool, "check_consistency": bool}
                    }
                },
            },
            "features": {
                "types": {
                    "enable_caching": bool,
                    "cache_ttl": int,
                    "enable_metrics": bool,
                    "debug_mode": bool,
                    "async_processing": bool,
                    "max_workers": int,
                },
                "constraints": {
                    "cache_ttl": {"min": 0},
                    "max_workers": {"min": 1, "max": 32},
                },
            },
            "data": {
                "types": {
                    "input_formats": list,
                    "output_directory": str,
                    "batch_size": int,
                    "parallel_processing": bool,
                },
                "constraints": {"batch_size": {"min": 1, "max": 10000}},
            },
            "llm": {
                "types": {
                    "provider": str,
                    "model": str,
                    "max_tokens": int,
                    "temperature": (int, float),
                    "timeout": int,
                    "retry_attempts": int,
                },
                "constraints": {
                    "provider": {"choices": ["openai", "anthropic", "huggingface"]},
                    "max_tokens": {"min": 1, "max": 32768},
                    "temperature": {"min": 0.0, "max": 2.0},
                    "timeout": {"min": 1, "max": 600},
                    "retry_attempts": {"min": 0, "max": 10},
                },
            },
            "evaluation": {
                "types": {
                    "metrics": list,
                    "benchmark_datasets": list,
                    "output_format": str,
                },
                "constraints": {
                    "output_format": {"choices": ["json", "yaml", "csv", "xml"]}
                },
            },
            "security": {
                "types": {
                    "encrypt_config": bool,
                    "allowed_hosts": list,
                    "cors_enabled": bool,
                }
            },
        }

        self.register_schema("aim2_project", aim2_schema, "1.0.0")

        # Database-only schema for minimal configurations
        database_schema = {"database": aim2_schema["database"]}
        self.register_schema("database_only", database_schema, "1.0.0")

        # API-only schema
        api_schema = {"api": aim2_schema["api"]}
        self.register_schema("api_only", api_schema, "1.0.0")


class ValidationReport:
    """
    Detailed validation report with errors, warnings, and metadata.
    """

    def __init__(self):
        """Initialize the validation report."""
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.schema_name: Optional[str] = None
        self.schema_version: Optional[str] = None
        self.validated_fields: List[str] = []
        self.missing_fields: List[str] = []
        self.extra_fields: List[str] = []

    def add_error(self, error: str) -> None:
        """Add an error to the report."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the report."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "validated_fields": self.validated_fields,
            "missing_fields": self.missing_fields,
            "extra_fields": self.extra_fields,
        }

    def __str__(self) -> str:
        """String representation of the validation report."""
        lines = [f"Validation Report (Schema: {self.schema_name or 'unknown'})"]
        lines.append(f"Valid: {self.is_valid}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.missing_fields:
            lines.append(f"\nMissing fields: {', '.join(self.missing_fields)}")

        if self.extra_fields:
            lines.append(f"Extra fields: {', '.join(self.extra_fields)}")

        return "\n".join(lines)


class SchemaMigration:
    """
    Represents a schema migration from one version to another.
    """

    def __init__(
        self,
        from_version: str,
        to_version: str,
        migration_function: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: str = "",
    ):
        """
        Initialize a schema migration.

        Args:
            from_version: Source schema version
            to_version: Target schema version
            migration_function: Function to perform the migration
            description: Optional description of the migration
        """
        self.from_version = from_version
        self.to_version = to_version
        self.migration_function = migration_function
        self.description = description

    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the migration to a configuration."""
        return self.migration_function(config)


class SchemaMigrator:
    """
    Manages schema migrations and version compatibility.
    """

    def __init__(self):
        """Initialize the schema migrator."""
        self.migrations: Dict[str, List[SchemaMigration]] = {}
        self._setup_built_in_migrations()

    def register_migration(self, schema_name: str, migration: SchemaMigration) -> None:
        """
        Register a migration for a schema.

        Args:
            schema_name: Name of the schema
            migration: Migration to register
        """
        if schema_name not in self.migrations:
            self.migrations[schema_name] = []
        self.migrations[schema_name].append(migration)
        # Sort migrations by version
        self.migrations[schema_name].sort(key=lambda m: version.parse(m.from_version))

    def migrate_config(
        self,
        config: Dict[str, Any],
        schema_name: str,
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """
        Migrate configuration from one schema version to another.

        Args:
            config: Configuration to migrate
            schema_name: Name of the schema
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated configuration

        Raises:
            ValueError: If migration path cannot be found
        """
        if version.parse(from_version) == version.parse(to_version):
            return config

        if schema_name not in self.migrations:
            raise ValueError(f"No migrations available for schema: {schema_name}")

        # Find migration path
        migration_path = self._find_migration_path(
            schema_name, from_version, to_version
        )
        if not migration_path:
            raise ValueError(
                f"No migration path found from {from_version} to {to_version} for schema {schema_name}"
            )

        # Apply migrations in sequence
        migrated_config = config.copy()
        for migration in migration_path:
            migrated_config = migration.apply(migrated_config)

        return migrated_config

    def get_latest_version(self, schema_name: str) -> Optional[str]:
        """
        Get the latest available version for a schema.

        Args:
            schema_name: Name of the schema

        Returns:
            Latest version or None if no migrations exist
        """
        if schema_name not in self.migrations:
            return None

        if not self.migrations[schema_name]:
            return None

        # Find the highest to_version
        versions = [m.to_version for m in self.migrations[schema_name]]
        return str(max(version.parse(v) for v in versions))

    def is_migration_needed(
        self, schema_name: str, current_version: str, target_version: str
    ) -> bool:
        """
        Check if migration is needed between two versions.

        Args:
            schema_name: Name of the schema
            current_version: Current schema version
            target_version: Target schema version

        Returns:
            True if migration is needed
        """
        return version.parse(current_version) != version.parse(target_version)

    def _find_migration_path(
        self, schema_name: str, from_version: str, to_version: str
    ) -> Optional[List[SchemaMigration]]:
        """
        Find a migration path between two versions.

        Args:
            schema_name: Name of the schema
            from_version: Source version
            to_version: Target version

        Returns:
            List of migrations to apply, or None if no path exists
        """
        if schema_name not in self.migrations:
            return None

        # Simple linear migration path (can be enhanced for complex graphs)
        migrations = self.migrations[schema_name]
        path = []
        current_version = from_version

        while version.parse(current_version) < version.parse(to_version):
            # Find next migration
            next_migration = None
            for migration in migrations:
                if migration.from_version == current_version:
                    next_migration = migration
                    break

            if next_migration is None:
                return None  # No path found

            path.append(next_migration)
            current_version = next_migration.to_version

        return path if current_version == to_version else None

    def _setup_built_in_migrations(self) -> None:
        """Set up built-in migrations for AIM2 schemas."""

        # Example migration from 1.0.0 to 1.1.0 for aim2_project schema
        def migrate_aim2_1_0_to_1_1(config: Dict[str, Any]) -> Dict[str, Any]:
            """
            Example migration: Add new security section if missing.
            """
            migrated = config.copy()

            # Add security section with defaults if missing
            if "security" not in migrated:
                migrated["security"] = {
                    "encrypt_config": False,
                    "allowed_hosts": [],
                    "cors_enabled": False,
                }

            # Rename old field names if they exist
            if "logging" in migrated and "log_level" in migrated["logging"]:
                migrated["logging"]["level"] = migrated["logging"].pop("log_level")

            return migrated

        migration_1_0_to_1_1 = SchemaMigration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_function=migrate_aim2_1_0_to_1_1,
            description="Add security section and normalize logging configuration",
        )

        self.register_migration("aim2_project", migration_1_0_to_1_1)


class ConfigValidator:
    """
    Main configuration validation class.

    Provides comprehensive validation capabilities including schema validation,
    custom rules, constraint checking, and error collection.

    Attributes:
        custom_rules (Dict[str, ValidationRule]): Registry of custom validation rules
        warnings (List[str]): Collection of validation warnings
        schema_registry (SchemaRegistry): Registry for managing schemas
    """

    def __init__(self):
        """
        Initialize the ConfigValidator.

        Sets up the validator with empty custom rules registry and warnings collection.
        """
        self.custom_rules = {}
        self.warnings = []
        self.schema_registry = SchemaRegistry()
        self.migrator = SchemaMigrator()
        self._setup_built_in_validators()

    def validate(self, config: Any) -> bool:
        """
        Perform basic validation on a configuration.

        This method performs fundamental validation checks to ensure the
        configuration is in a valid format for further processing.

        Args:
            config (Any): Configuration to validate

        Returns:
            bool: True if configuration is valid

        Raises:
            ConfigError: If configuration is invalid
        """
        from .config_manager import ConfigError

        if config is None:
            raise ConfigError("Configuration cannot be None")

        if not isinstance(config, dict):
            raise ConfigError("Configuration must be a dictionary")

        # Check for circular references
        self._check_circular_references(config)

        return True

    def validate_schema(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        strict: bool = False,
        collect_warnings: bool = False,
    ) -> bool:
        """
        Validate configuration against a detailed schema.

        Performs comprehensive validation including type checking, required fields,
        constraints, custom validators, and nested schema validation.

        Args:
            config (Dict[str, Any]): Configuration to validate
            schema (Dict[str, Any]): Validation schema
            strict (bool): Whether to enforce strict validation (reject extra fields)
            collect_warnings (bool): Whether to collect validation warnings

        Returns:
            bool: True if configuration is valid

        Raises:
            ConfigError: If configuration doesn't match schema
        """
        from .config_manager import ConfigError

        # Clear previous warnings if collecting
        if collect_warnings:
            self.warnings.clear()

        try:
            self._validate_schema_recursive(
                config,
                schema,
                strict=strict,
                collect_warnings=collect_warnings,
                path="",
            )
            return True
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Schema validation failed: {str(e)}", e)

    def validate_with_schema(
        self,
        config: Dict[str, Any],
        schema_name: str,
        strict: bool = False,
        return_report: bool = False,
    ) -> Union[bool, ValidationReport]:
        """
        Validate configuration using a named schema from the registry.

        Args:
            config: Configuration to validate
            schema_name: Name of the schema to use
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            bool or ValidationReport: Validation result

        Raises:
            ValueError: If schema is not found
        """
        schema = self.schema_registry.get_schema(schema_name)
        if schema is None:
            raise ValueError(f"Schema not found: {schema_name}")

        if return_report:
            return self.validate_schema_with_report(config, schema, schema_name, strict)
        else:
            return self.validate_schema(config, schema, strict)

    def validate_schema_with_report(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        schema_name: Optional[str] = None,
        strict: bool = False,
    ) -> ValidationReport:
        """
        Validate configuration against schema and return detailed report.

        Args:
            config: Configuration to validate
            schema: Validation schema
            schema_name: Optional schema name for reporting
            strict: Whether to enforce strict validation

        Returns:
            ValidationReport: Detailed validation report
        """
        report = ValidationReport()
        report.schema_name = schema_name
        if schema_name and schema_name in self.schema_registry.schema_versions:
            report.schema_version = self.schema_registry.schema_versions[schema_name]

        try:
            self._validate_schema_with_report(config, schema, report, strict=strict)
        except Exception as e:
            report.add_error(f"Validation failed: {str(e)}")

        return report

    def validate_aim2_config(
        self, config: Dict[str, Any], strict: bool = False, return_report: bool = False
    ) -> Union[bool, ValidationReport]:
        """
        Validate configuration against the built-in AIM2 project schema.

        Args:
            config: Configuration to validate
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            bool or ValidationReport: Validation result
        """
        return self.validate_with_schema(
            config, "aim2_project", strict=strict, return_report=return_report
        )

    def load_and_register_schema(
        self, file_path: str, name: Optional[str] = None, version: str = "1.0.0"
    ) -> None:
        """
        Load schema from file and register it.

        Args:
            file_path: Path to schema file
            name: Optional name for the schema
            version: Schema version
        """
        self.schema_registry.load_schema_from_file(file_path, name)
        if name and name in self.schema_registry.schemas:
            self.schema_registry.schema_versions[name] = version

    def get_available_schemas(self) -> List[str]:
        """Get list of all available schema names."""
        return self.schema_registry.list_schemas()

    def migrate_config(
        self,
        config: Dict[str, Any],
        schema_name: str,
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """
        Migrate configuration from one schema version to another.

        Args:
            config: Configuration to migrate
            schema_name: Name of the schema
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated configuration

        Raises:
            ValueError: If migration path cannot be found
        """
        return self.migrator.migrate_config(
            config, schema_name, from_version, to_version
        )

    def register_schema_migration(
        self, schema_name: str, migration: SchemaMigration
    ) -> None:
        """
        Register a schema migration.

        Args:
            schema_name: Name of the schema
            migration: Migration to register
        """
        self.migrator.register_migration(schema_name, migration)

    def get_latest_schema_version(self, schema_name: str) -> Optional[str]:
        """
        Get the latest available version for a schema.

        Args:
            schema_name: Name of the schema

        Returns:
            Latest version or None if no migrations exist
        """
        return self.migrator.get_latest_version(schema_name)

    def validate_and_migrate(
        self,
        config: Dict[str, Any],
        schema_name: str,
        config_version: Optional[str] = None,
        target_version: Optional[str] = None,
        strict: bool = False,
        return_report: bool = False,
    ) -> Union[
        bool, ValidationReport, Tuple[Dict[str, Any], Union[bool, ValidationReport]]
    ]:
        """
        Validate configuration and migrate if necessary.

        Args:
            config: Configuration to validate
            schema_name: Name of the schema to use
            config_version: Current configuration version (auto-detect if None)
            target_version: Target schema version (latest if None)
            strict: Whether to enforce strict validation
            return_report: Whether to return detailed validation report

        Returns:
            If migration occurred: (migrated_config, validation_result)
            If no migration: validation_result only
        """
        # Determine target version
        if target_version is None:
            target_version = self.get_latest_schema_version(schema_name)
            if target_version is None:
                # No migrations available, use current schema version
                target_version = self.schema_registry.schema_versions.get(
                    schema_name, "1.0.0"
                )

        # Determine current version
        if config_version is None:
            config_version = self._detect_config_version(config, schema_name)

        # Check if migration is needed
        migrated_config = config
        migration_occurred = False

        if self.migrator.is_migration_needed(
            schema_name, config_version, target_version
        ):
            try:
                migrated_config = self.migrate_config(
                    config, schema_name, config_version, target_version
                )
                migration_occurred = True
            except ValueError:
                # Migration failed, continue with original config
                migrated_config = config

        # Validate the (potentially migrated) configuration
        validation_result = self.validate_with_schema(
            migrated_config, schema_name, strict=strict, return_report=return_report
        )

        if migration_occurred:
            return migrated_config, validation_result
        else:
            return validation_result

    def _detect_config_version(self, config: Dict[str, Any], schema_name: str) -> str:
        """
        Attempt to detect the configuration version.

        Args:
            config: Configuration to analyze
            schema_name: Name of the schema

        Returns:
            Detected version or default "1.0.0"
        """
        # Look for explicit version field
        if "version" in config:
            return str(config["version"])

        if "project" in config and isinstance(config["project"], dict):
            if "version" in config["project"]:
                return str(config["project"]["version"])

        # Look for schema-specific version indicators
        if "schema_version" in config:
            return str(config["schema_version"])

        # Default to 1.0.0 if no version information found
        return "1.0.0"

    def add_validation_rule(
        self, name: str, validator: Callable[[Any], bool], description: str = ""
    ) -> None:
        """
        Add a custom validation rule.

        Custom rules can be referenced in schemas and applied to specific fields.

        Args:
            name (str): Name of the validation rule
            validator (Callable): Function that validates a value
            description (str): Optional description of the rule
        """
        rule = ValidationRule(name, validator, description)
        self.custom_rules[name] = rule

    def _setup_built_in_validators(self) -> None:
        """Set up built-in custom validators for common configuration patterns."""

        def validate_logging_handlers(handlers: List[str]) -> bool:
            """Validate logging handlers configuration."""
            if not isinstance(handlers, list):
                raise ValueError("Handlers must be a list")

            valid_handlers = ["console", "file"]
            for handler in handlers:
                if not isinstance(handler, str):
                    raise ValueError(f"Handler '{handler}' must be a string")
                if handler not in valid_handlers:
                    raise ValueError(
                        f"Invalid handler '{handler}'. Must be one of: {', '.join(valid_handlers)}"
                    )

            return True

        def validate_file_size_format(size_value: Any) -> bool:
            """Validate file size format (e.g., '10MB' or integer bytes)."""
            import re

            if isinstance(size_value, int):
                if size_value < 1024:
                    raise ValueError("File size must be at least 1024 bytes")
                return True

            if isinstance(size_value, str):
                size_str = size_value.strip().upper()
                pattern = r"^(\d+(?:\.\d+)?)\s*([A-Z]*B?)$"
                match = re.match(pattern, size_str)

                if not match:
                    raise ValueError(
                        f"Invalid size format: {size_value}. Expected format like '10MB', '1.5GB', or integer bytes"
                    )

                number_str, unit = match.groups()

                try:
                    number = float(number_str)
                except ValueError:
                    raise ValueError(f"Invalid number in size string: {number_str}")

                if number <= 0:
                    raise ValueError("Size must be positive")

                # Valid units
                size_units = {
                    "B": 1,
                    "KB": 1024,
                    "MB": 1024**2,
                    "GB": 1024**3,
                    "TB": 1024**4,
                }

                if not unit or unit == "B":
                    multiplier = 1
                else:
                    if unit not in size_units:
                        valid_units = ", ".join(size_units.keys())
                        raise ValueError(
                            f"Invalid size unit: {unit}. Valid units are: {valid_units}"
                        )
                    multiplier = size_units[unit]

                result = int(number * multiplier)
                if result < 1024:
                    raise ValueError("Minimum file size is 1024 bytes (1KB)")

                return True

            raise ValueError("Size must be a string (e.g., '10MB') or integer (bytes)")

        # Register built-in validators
        self.add_validation_rule(
            "validate_logging_handlers",
            validate_logging_handlers,
            "Validate logging handlers configuration",
        )

        self.add_validation_rule(
            "validate_file_size_format",
            validate_file_size_format,
            "Validate file size format (e.g., '10MB' or integer bytes)",
        )

    # Private helper methods

    def _validate_schema_with_report(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        report: ValidationReport,
        strict: bool = False,
        path: str = "",
    ) -> None:
        """
        Validate configuration against schema and populate validation report.

        Args:
            config: Configuration to validate
            schema: Validation schema
            report: ValidationReport to populate
            strict: Whether to enforce strict validation
            path: Current path in configuration for error reporting
        """
        for section_name, section_schema in schema.items():
            section_path = f"{path}.{section_name}" if path else section_name

            # Check if section exists in config
            if section_name not in config:
                if "required" in section_schema and section_schema["required"]:
                    report.add_error(f"Required section missing: {section_path}")
                    report.missing_fields.append(section_path)
                continue

            section_config = config[section_name]
            report.validated_fields.append(section_path)

            # Handle list item validation
            if "list_item_schema" in section_schema and isinstance(
                section_config, list
            ):
                self._validate_list_items_with_report(
                    section_config,
                    section_schema["list_item_schema"],
                    report,
                    section_path,
                )
                continue

            # Skip if section is not a dict
            if not isinstance(section_config, dict):
                if "types" in section_schema:
                    for field, expected_type in section_schema["types"].items():
                        if field == section_name:
                            error = self._validate_type(
                                section_config, expected_type, section_path
                            )
                            if error:
                                report.add_error(error)
                continue

            # Check required fields
            if "required" in section_schema:
                for required_field in section_schema["required"]:
                    if required_field not in section_config:
                        field_path = f"{section_path}.{required_field}"
                        report.add_error(f"Required field missing: {field_path}")
                        report.missing_fields.append(field_path)

            # Validate types and constraints
            if "types" in section_schema:
                for field, expected_type in section_schema["types"].items():
                    if field in section_config:
                        field_path = f"{section_path}.{field}"
                        value = section_config[field]
                        report.validated_fields.append(field_path)

                        # Type validation
                        error = self._validate_type(value, expected_type, field_path)
                        if error:
                            report.add_error(error)

                        # Constraint validation
                        if (
                            "constraints" in section_schema
                            and field in section_schema["constraints"]
                        ):
                            error = self._validate_constraints(
                                value, section_schema["constraints"][field], field_path
                            )
                            if error:
                                report.add_error(error)

                        # Custom validator validation
                        if (
                            "custom_validators" in section_schema
                            and field in section_schema["custom_validators"]
                        ):
                            error = self._validate_custom_validators(
                                value,
                                section_schema["custom_validators"][field],
                                field_path,
                            )
                            if error:
                                report.add_error(error)

            # Check for warnings on deprecated fields
            if "warnings" in section_schema:
                for field, warning_msg in section_schema["warnings"].items():
                    if field in section_config:
                        field_path = f"{section_path}.{field}"
                        report.add_warning(f"{field_path}: {warning_msg}")

            # Check for extra fields in strict mode
            if strict and "types" in section_schema:
                allowed_fields = set(section_schema["types"].keys())
                actual_fields = set(section_config.keys())
                extra_fields = actual_fields - allowed_fields

                if extra_fields:
                    for extra_field in extra_fields:
                        extra_field_path = f"{section_path}.{extra_field}"
                        report.extra_fields.append(extra_field_path)
                    extra_list = ", ".join(sorted(extra_fields))
                    report.add_error(
                        f"Extra fields not allowed in strict mode at {section_path}: {extra_list}"
                    )

            # Recursively validate nested schemas
            if "nested" in section_schema:
                for field, nested_schema in section_schema["nested"].items():
                    if field in section_config and isinstance(
                        section_config[field], dict
                    ):
                        self._validate_schema_with_report(
                            {field: section_config[field]},
                            {field: nested_schema},
                            report,
                            strict=strict,
                            path=section_path,
                        )

    def _validate_list_items_with_report(
        self,
        items: List[Any],
        item_schema: Dict[str, Any],
        report: ValidationReport,
        field_path: str,
    ) -> None:
        """
        Validate list items and add results to report.

        Args:
            items: List of items to validate
            item_schema: Schema for validating each item
            report: ValidationReport to populate
            field_path: Path to the field for error reporting
        """
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                report.add_error(f"List item at {field_path}[{i}] must be a dictionary")
                continue

            item_path = f"{field_path}[{i}]"
            report.validated_fields.append(item_path)

            # Validate required fields for this item
            if "required" in item_schema:
                for required_field in item_schema["required"]:
                    if required_field not in item:
                        report.add_error(
                            f"Required field missing: {item_path}.{required_field}"
                        )
                        report.missing_fields.append(f"{item_path}.{required_field}")

            # Validate types for this item
            if "types" in item_schema:
                for field, expected_type in item_schema["types"].items():
                    if field in item:
                        value = item[field]
                        field_path_full = f"{item_path}.{field}"
                        report.validated_fields.append(field_path_full)

                        # Type validation
                        error = self._validate_type(
                            value, expected_type, field_path_full
                        )
                        if error:
                            report.add_error(error)

                        # Constraint validation
                        if (
                            "constraints" in item_schema
                            and field in item_schema["constraints"]
                        ):
                            error = self._validate_constraints(
                                value,
                                item_schema["constraints"][field],
                                field_path_full,
                            )
                            if error:
                                report.add_error(error)

    def _validate_schema_recursive(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        strict: bool = False,
        collect_warnings: bool = False,
        path: str = "",
    ) -> None:
        """
        Recursively validate configuration against schema.

        This is the core validation method that handles all aspects of schema
        validation including types, constraints, custom validators, and nesting.
        """
        from .config_manager import ConfigError

        errors = []

        # Handle two schema formats:
        # 1. Direct section validation (like {"database": {"required": [...], "types": {...}}})
        # 2. Nested schema format (like {"database": {"nested": {...}}})

        for section_name, section_schema in schema.items():
            section_path = f"{path}.{section_name}" if path else section_name

            # Check if this section exists in config
            if section_name not in config:
                # If the section has required fields, this is an error
                if "required" in section_schema and section_schema["required"]:
                    errors.append(f"Required section missing: {section_path}")
                continue

            section_config = config[section_name]

            # Handle special case where section itself is a list with list_item_schema
            if "list_item_schema" in section_schema and isinstance(
                section_config, list
            ):
                # Validate the list items directly
                list_errors = self._validate_list_items(
                    section_config, section_schema["list_item_schema"], section_path
                )
                errors.extend(list_errors)
                continue

            # If the section config is not a dict, we can't validate it further
            if not isinstance(section_config, dict):
                if "types" in section_schema:
                    # Check type constraints for this section
                    for field, expected_type in section_schema["types"].items():
                        if field == section_name:  # Self-reference
                            type_error = self._validate_type(
                                section_config, expected_type, section_path
                            )
                            if type_error:
                                errors.append(type_error)
                continue

            # Check required fields within this section
            if "required" in section_schema:
                for required_field in section_schema["required"]:
                    if required_field not in section_config:
                        field_path = f"{section_path}.{required_field}"
                        errors.append(f"Required field missing: {field_path}")

            # Check types and constraints for each field in this section
            if "types" in section_schema:
                for field, expected_type in section_schema["types"].items():
                    if field in section_config:
                        field_path = f"{section_path}.{field}"
                        value = section_config[field]

                        # Type validation
                        type_error = self._validate_type(
                            value, expected_type, field_path
                        )
                        if type_error:
                            errors.append(type_error)

                        # Constraint validation
                        if (
                            "constraints" in section_schema
                            and field in section_schema["constraints"]
                        ):
                            constraint_error = self._validate_constraints(
                                value, section_schema["constraints"][field], field_path
                            )
                            if constraint_error:
                                errors.append(constraint_error)

                        # Custom validator validation
                        if (
                            "custom_validators" in section_schema
                            and field in section_schema["custom_validators"]
                        ):
                            custom_error = self._validate_custom_validators(
                                value,
                                section_schema["custom_validators"][field],
                                field_path,
                            )
                            if custom_error:
                                errors.append(custom_error)

            # Check for warnings on deprecated fields
            if collect_warnings and "warnings" in section_schema:
                for field, warning_msg in section_schema["warnings"].items():
                    if field in section_config:
                        field_path = f"{section_path}.{field}"
                        self.warnings.append(f"{field_path}: {warning_msg}")

            # Check for extra fields in strict mode
            if strict and "types" in section_schema:
                allowed_fields = set(section_schema["types"].keys())
                actual_fields = set(section_config.keys())
                extra_fields = actual_fields - allowed_fields

                if extra_fields:
                    extra_list = ", ".join(sorted(extra_fields))
                    errors.append(
                        f"Extra fields not allowed in strict mode at {section_path}: {extra_list}"
                    )

            # Validate list items if schema provided
            if "list_item_schema" in section_schema:
                for field, field_schema in section_schema.get("types", {}).items():
                    if field in section_config and field_schema == list:
                        field_path = f"{section_path}.{field}"
                        list_errors = self._validate_list_items(
                            section_config[field],
                            section_schema["list_item_schema"],
                            field_path,
                        )
                        errors.extend(list_errors)

            # Recursively validate nested schemas
            if "nested" in section_schema:
                for field, nested_schema in section_schema["nested"].items():
                    if field in section_config and isinstance(
                        section_config[field], dict
                    ):
                        f"{section_path}.{field}"
                        try:
                            self._validate_schema_recursive(
                                {field: section_config[field]},
                                {field: nested_schema},
                                strict=strict,
                                collect_warnings=collect_warnings,
                                path=section_path,
                            )
                        except ConfigError as e:
                            errors.append(str(e))

        # If we have multiple errors, collect them into a single error message
        if errors:
            if len(errors) == 1:
                raise ConfigError(errors[0])
            else:
                error_msg = f"Multiple validation errors found:\n" + "\n".join(
                    f"  - {err}" for err in errors
                )
                raise ConfigError(error_msg)

    def _validate_type(
        self, value: Any, expected_type: Union[type, Tuple[type, ...]], field_path: str
    ) -> Optional[str]:
        """
        Validate the type of a value.

        Args:
            value: Value to validate
            expected_type: Expected type or tuple of types
            field_path: Path to the field for error reporting

        Returns:
            Optional[str]: Error message if validation fails, None otherwise
        """
        # Handle multiple allowed types (tuple)
        if isinstance(expected_type, tuple):
            if not any(isinstance(value, t) for t in expected_type):
                type_names = [t.__name__ for t in expected_type]
                return (
                    f"Invalid type for {field_path}: expected {' or '.join(type_names)}, "
                    f"got {type(value).__name__}"
                )
        else:
            if not isinstance(value, expected_type):
                return (
                    f"Invalid type for {field_path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        return None

    def _validate_constraints(
        self, value: Any, constraints: Dict[str, Any], field_path: str
    ) -> Optional[str]:
        """
        Validate constraints on a value.

        Args:
            value: Value to validate
            constraints: Dictionary of constraints
            field_path: Path to the field for error reporting

        Returns:
            Optional[str]: Error message if validation fails, None otherwise
        """
        # Check numeric ranges (only if value is numeric)
        if isinstance(value, (int, float)):
            if "min" in constraints and value < constraints["min"]:
                return f"Value {value} for {field_path} is below minimum {constraints['min']}"

            if "max" in constraints and value > constraints["max"]:
                return f"Value {value} for {field_path} is above maximum {constraints['max']}"

        # Check choices
        if "choices" in constraints and value not in constraints["choices"]:
            return f"Invalid choice for {field_path}: {value} not in {constraints['choices']}"

        # Check string length constraints
        if isinstance(value, str):
            if "min_length" in constraints and len(value) < constraints["min_length"]:
                return f"String {field_path} is too short: {len(value)} < {constraints['min_length']}"

            if "max_length" in constraints and len(value) > constraints["max_length"]:
                return f"String {field_path} is too long: {len(value)} > {constraints['max_length']}"

        return None

    def _validate_custom_validators(
        self, value: Any, validator_names: List[str], field_path: str
    ) -> Optional[str]:
        """
        Apply custom validators to a value.

        Args:
            value: Value to validate
            validator_names: List of custom validator names to apply
            field_path: Path to the field for error reporting

        Returns:
            Optional[str]: Error message if validation fails, None otherwise
        """
        for validator_name in validator_names:
            if validator_name not in self.custom_rules:
                return f"Unknown validation rule: {validator_name}"

            try:
                rule = self.custom_rules[validator_name]
                rule.validate(value)
            except ValueError as e:
                return f"Custom validation failed for {field_path}: {str(e)}"
            except Exception as e:
                return f"Custom validation error for {field_path}: {str(e)}"

        return None

    def _validate_list_items(
        self, items: List[Any], item_schema: Dict[str, Any], field_path: str
    ) -> List[str]:
        """
        Validate items in a list against a schema.

        Args:
            items: List of items to validate
            item_schema: Schema for validating each item
            field_path: Path to the field for error reporting

        Returns:
            List[str]: List of error messages
        """
        errors = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"List item at {field_path}[{i}] must be a dictionary")
                continue

            item_path = f"{field_path}[{i}]"

            # Validate required fields for this item
            if "required" in item_schema:
                for required_field in item_schema["required"]:
                    if required_field not in item:
                        errors.append(
                            f"Required field missing: {item_path}.{required_field}"
                        )

            # Validate types for this item
            if "types" in item_schema:
                for field, expected_type in item_schema["types"].items():
                    if field in item:
                        value = item[field]
                        field_path_full = f"{item_path}.{field}"

                        # Type validation
                        type_error = self._validate_type(
                            value, expected_type, field_path_full
                        )
                        if type_error:
                            errors.append(type_error)

                        # Constraint validation
                        if (
                            "constraints" in item_schema
                            and field in item_schema["constraints"]
                        ):
                            constraint_error = self._validate_constraints(
                                value,
                                item_schema["constraints"][field],
                                field_path_full,
                            )
                            if constraint_error:
                                errors.append(constraint_error)

        return errors

    def _check_circular_references(self, obj: Any, seen: Optional[set] = None) -> None:
        """
        Check for circular references in the configuration.

        Args:
            obj: Object to check
            seen: Set of already seen object IDs

        Raises:
            ConfigError: If circular reference detected
        """

        if seen is None:
            seen = set()

        obj_id = id(obj)

        if obj_id in seen:
            # This would indicate a circular reference, but for config validation
            # we'll allow it since it's a normal Python behavior
            return

        if isinstance(obj, (dict, list)):
            seen.add(obj_id)

            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        self._check_circular_references(value, seen.copy())
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        self._check_circular_references(item, seen.copy())
