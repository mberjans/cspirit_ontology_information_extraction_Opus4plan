"""
Unit tests for config validator functionality.

This module contains comprehensive unit tests for the configuration validation system,
following Test-Driven Development (TDD) approach. These tests define the expected
interface and behavior of the ConfigValidator class and validation methods before implementation.

Test Coverage:
- Required field validation
- Type checking validation
- Schema structure validation
- Nested config validation
- Custom validation rules
- Invalid config handling
- Validation error reporting
- Schema compliance testing
- Edge cases and error conditions
- SchemaRegistry functionality
- ValidationReport functionality
- SchemaMigrator functionality
- External schema loading
- Schema versioning and migration
"""

import pytest
import json
import tempfile
from pathlib import Path

# Import the module to be tested (will fail initially in TDD)
try:
    from aim2_project.aim2_utils.config_manager import ConfigManager, ConfigError
    from aim2_project.aim2_utils.config_validator import (
        ConfigValidator,
        ValidationRule,
        SchemaRegistry,
        ValidationReport,
        SchemaMigrator,
        SchemaMigration,
    )
except ImportError:
    # Expected during TDD - tests define the interface
    pass


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    @pytest.fixture
    def config_validator(self):
        """Create a ConfigValidator instance for testing."""
        return ConfigValidator()

    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "test_user",
                "password": "test_pass",
                "ssl_enabled": True,
                "timeout": 30.0,
                "pool_size": 10,
            },
            "api": {
                "base_url": "https://api.example.com",
                "version": "v1",
                "timeout": 30,
                "retry_attempts": 3,
                "rate_limit": 1000,
                "endpoints": ["users", "data", "auth"],
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"],
                "file_path": "/var/log/app.log",
                "max_file_size": 10485760,  # 10MB in bytes
                "backup_count": 5,
            },
            "features": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "debug_mode": False,
                "max_workers": 4,
                "async_processing": True,
            },
        }

    @pytest.fixture
    def basic_schema(self):
        """Basic validation schema for testing."""
        return {
            "database": {
                "required": ["host", "port", "name", "username", "password"],
                "types": {
                    "host": str,
                    "port": int,
                    "name": str,
                    "username": str,
                    "password": str,
                    "ssl_enabled": bool,
                    "timeout": (int, float),
                    "pool_size": int,
                },
                "constraints": {
                    "port": {"min": 1, "max": 65535},
                    "timeout": {"min": 0.1},
                    "pool_size": {"min": 1, "max": 100},
                },
            },
            "api": {
                "required": ["base_url", "timeout"],
                "types": {
                    "base_url": str,
                    "version": str,
                    "timeout": int,
                    "retry_attempts": int,
                    "rate_limit": int,
                    "endpoints": list,
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
                    "file_path": str,
                    "max_file_size": int,
                    "backup_count": int,
                },
                "constraints": {
                    "level": {
                        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    },
                    "max_file_size": {"min": 1024},  # Minimum 1KB
                    "backup_count": {"min": 0, "max": 20},
                },
            },
        }

    def test_config_validator_initialization(self, config_validator):
        """Test ConfigValidator can be instantiated."""
        assert config_validator is not None
        assert hasattr(config_validator, "validate")
        assert hasattr(config_validator, "validate_schema")
        assert hasattr(config_validator, "add_validation_rule")

    # Required Field Validation Tests
    def test_validate_required_fields_success(
        self, config_validator, valid_config, basic_schema
    ):
        """Test that valid config with all required fields passes validation."""
        # Should not raise exception
        result = config_validator.validate_schema(valid_config, basic_schema)
        assert result is True

    def test_validate_required_fields_missing_top_level_section_raises_error(
        self, config_validator, basic_schema
    ):
        """Test that missing required top-level section raises ConfigError."""
        incomplete_config = {
            "api": {"base_url": "https://api.example.com", "timeout": 30}
            # Missing 'database' section which is required
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(incomplete_config, basic_schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg
        assert "database" in error_msg

    def test_validate_required_fields_missing_nested_field_raises_error(
        self, config_validator, basic_schema
    ):
        """Test that missing required nested field raises ConfigError."""
        incomplete_config = {
            "database": {
                "host": "localhost",
                "port": 5432
                # Missing required fields: name, username, password
            },
            "api": {"base_url": "https://api.example.com", "timeout": 30},
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(incomplete_config, basic_schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg
        assert any(field in error_msg for field in ["name", "username", "password"])

    def test_validate_required_fields_partial_config_success(self, config_validator):
        """Test validation of partial config with only required fields."""
        minimal_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "user",
                "password": "pass",
            }
        }

        minimal_schema = {
            "database": {
                "required": ["host", "port", "name", "username", "password"],
                "types": {
                    "host": str,
                    "port": int,
                    "name": str,
                    "username": str,
                    "password": str,
                },
            }
        }

        # Should not raise exception
        result = config_validator.validate_schema(minimal_config, minimal_schema)
        assert result is True

    # Type Checking Validation Tests
    def test_validate_types_success(self, config_validator, valid_config, basic_schema):
        """Test that config with correct types passes validation."""
        result = config_validator.validate_schema(valid_config, basic_schema)
        assert result is True

    def test_validate_types_string_instead_of_int_raises_error(
        self, config_validator, basic_schema
    ):
        """Test that wrong type (string instead of int) raises ConfigError."""
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number",  # Should be int
                "name": "test_db",
                "username": "user",
                "password": "pass",
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(invalid_config, basic_schema)

        error_msg = str(exc_info.value).lower()
        assert "type" in error_msg or "invalid" in error_msg
        assert "port" in error_msg

    def test_validate_types_int_instead_of_string_raises_error(
        self, config_validator, basic_schema
    ):
        """Test that wrong type (int instead of string) raises ConfigError."""
        invalid_config = {
            "database": {
                "host": 12345,  # Should be string
                "port": 5432,
                "name": "test_db",
                "username": "user",
                "password": "pass",
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(invalid_config, basic_schema)

        error_msg = str(exc_info.value).lower()
        assert "type" in error_msg or "invalid" in error_msg
        assert "host" in error_msg

    def test_validate_types_list_instead_of_string_raises_error(
        self, config_validator, basic_schema
    ):
        """Test that wrong type (list instead of string) raises ConfigError."""
        invalid_config = {
            "logging": {
                "level": ["INFO", "DEBUG"],  # Should be string
                "format": "%(asctime)s - %(message)s",
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(invalid_config, basic_schema)

        error_msg = str(exc_info.value).lower()
        assert "type" in error_msg or "invalid" in error_msg
        assert "level" in error_msg

    def test_validate_types_multiple_allowed_types_success(self, config_validator):
        """Test validation with multiple allowed types (union types)."""
        config = {"database": {"timeout": 30.5}}  # Should accept both int and float

        schema = {"database": {"types": {"timeout": (int, float)}}}

        result = config_validator.validate_schema(config, schema)
        assert result is True

        # Test with int value
        config["database"]["timeout"] = 30
        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_types_none_value_handling(self, config_validator):
        """Test handling of None values in validation."""
        config_with_none = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "password": None,  # None value
            }
        }

        schema = {
            "database": {
                "types": {
                    "host": str,
                    "port": int,
                    "password": (str, type(None)),  # Allow None
                }
            }
        }

        result = config_validator.validate_schema(config_with_none, schema)
        assert result is True

    # Schema Structure Validation Tests
    def test_validate_schema_structure_success(
        self, config_validator, valid_config, basic_schema
    ):
        """Test validation of correct schema structure."""
        result = config_validator.validate_schema(valid_config, basic_schema)
        assert result is True

    def test_validate_schema_structure_extra_fields_allowed(self, config_validator):
        """Test that extra fields not in schema are allowed by default."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "extra_field": "extra_value",  # Not in schema
            }
        }

        schema = {"database": {"types": {"host": str, "port": int}}}

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_schema_structure_strict_mode_extra_fields_rejected(
        self, config_validator
    ):
        """Test that extra fields are rejected in strict mode."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "extra_field": "extra_value",
            }
        }

        schema = {"database": {"types": {"host": str, "port": int}, "strict": True}}

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema, strict=True)

        error_msg = str(exc_info.value).lower()
        assert (
            "extra" in error_msg or "unknown" in error_msg or "unexpected" in error_msg
        )
        assert "extra_field" in error_msg

    def test_validate_schema_empty_config_section(self, config_validator):
        """Test validation of empty config sections."""
        empty_config = {"database": {}}

        schema = {"database": {"required": ["host", "port"]}}

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(empty_config, schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg

    # Constraint Validation Tests
    def test_validate_constraints_numeric_range_success(self, config_validator):
        """Test successful validation of numeric range constraints."""
        config = {"database": {"port": 5432, "timeout": 30.0, "pool_size": 10}}

        schema = {
            "database": {
                "types": {"port": int, "timeout": float, "pool_size": int},
                "constraints": {
                    "port": {"min": 1, "max": 65535},
                    "timeout": {"min": 0.1},
                    "pool_size": {"min": 1, "max": 100},
                },
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_constraints_numeric_range_below_min_raises_error(
        self, config_validator
    ):
        """Test that values below minimum constraint raise ConfigError."""
        config = {"database": {"port": 0}}  # Below minimum of 1

        schema = {
            "database": {
                "types": {"port": int},
                "constraints": {"port": {"min": 1, "max": 65535}},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "minimum" in error_msg or "min" in error_msg
        assert "port" in error_msg

    def test_validate_constraints_numeric_range_above_max_raises_error(
        self, config_validator
    ):
        """Test that values above maximum constraint raise ConfigError."""
        config = {"database": {"port": 70000}}  # Above maximum of 65535

        schema = {
            "database": {
                "types": {"port": int},
                "constraints": {"port": {"min": 1, "max": 65535}},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "maximum" in error_msg or "max" in error_msg
        assert "port" in error_msg

    def test_validate_constraints_choices_success(self, config_validator):
        """Test successful validation of choice constraints."""
        config = {"logging": {"level": "INFO"}}

        schema = {
            "logging": {
                "types": {"level": str},
                "constraints": {
                    "level": {
                        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    }
                },
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_constraints_choices_invalid_choice_raises_error(
        self, config_validator
    ):
        """Test that invalid choice constraint raises ConfigError."""
        config = {"logging": {"level": "INVALID"}}  # Not in allowed choices

        schema = {
            "logging": {
                "types": {"level": str},
                "constraints": {
                    "level": {
                        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    }
                },
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "choice" in error_msg or "allowed" in error_msg or "valid" in error_msg
        assert "level" in error_msg

    def test_validate_constraints_string_length_success(self, config_validator):
        """Test successful validation of string length constraints."""
        config = {"database": {"password": "secure_password123"}}

        schema = {
            "database": {
                "types": {"password": str},
                "constraints": {"password": {"min_length": 8, "max_length": 50}},
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_constraints_string_too_short_raises_error(self, config_validator):
        """Test that string shorter than minimum length raises ConfigError."""
        config = {"database": {"password": "123"}}  # Too short

        schema = {
            "database": {
                "types": {"password": str},
                "constraints": {"password": {"min_length": 8}},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "length" in error_msg or "short" in error_msg
        assert "password" in error_msg

    def test_validate_constraints_string_too_long_raises_error(self, config_validator):
        """Test that string longer than maximum length raises ConfigError."""
        config = {"database": {"password": "a" * 100}}  # Too long

        schema = {
            "database": {
                "types": {"password": str},
                "constraints": {"password": {"max_length": 50}},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "length" in error_msg or "long" in error_msg
        assert "password" in error_msg

    # Nested Configuration Validation Tests
    def test_validate_nested_config_success(self, config_validator):
        """Test validation of deeply nested configuration structures."""
        nested_config = {
            "app": {
                "services": {
                    "database": {
                        "primary": {"host": "db1.example.com", "port": 5432},
                        "replica": {"host": "db2.example.com", "port": 5432},
                    },
                    "cache": {
                        "redis": {
                            "host": "redis.example.com",
                            "port": 6379,
                            "clusters": ["cluster1", "cluster2"],
                        }
                    },
                }
            }
        }

        nested_schema = {
            "app": {
                "types": {"services": dict},
                "nested": {
                    "services": {
                        "types": {"database": dict, "cache": dict},
                        "nested": {
                            "database": {
                                "types": {"primary": dict, "replica": dict},
                                "nested": {
                                    "primary": {
                                        "required": ["host", "port"],
                                        "types": {"host": str, "port": int},
                                    },
                                    "replica": {
                                        "required": ["host", "port"],
                                        "types": {"host": str, "port": int},
                                    },
                                },
                            },
                            "cache": {
                                "types": {"redis": dict},
                                "nested": {
                                    "redis": {
                                        "required": ["host", "port"],
                                        "types": {
                                            "host": str,
                                            "port": int,
                                            "clusters": list,
                                        },
                                    }
                                },
                            },
                        },
                    }
                },
            }
        }

        result = config_validator.validate_schema(nested_config, nested_schema)
        assert result is True

    def test_validate_nested_config_missing_deep_field_raises_error(
        self, config_validator
    ):
        """Test that missing field in deep nested structure raises ConfigError."""
        nested_config = {
            "app": {
                "services": {
                    "database": {
                        "primary": {
                            "host": "db1.example.com"
                            # Missing required 'port'
                        }
                    }
                }
            }
        }

        nested_schema = {
            "app": {
                "nested": {
                    "services": {
                        "nested": {
                            "database": {
                                "nested": {
                                    "primary": {
                                        "required": ["host", "port"],
                                        "types": {"host": str, "port": int},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(nested_config, nested_schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg
        assert "port" in error_msg

    def test_validate_list_of_objects_success(self, config_validator):
        """Test validation of list containing objects with schema validation."""
        config = {
            "servers": [
                {"name": "server1", "host": "host1.example.com", "port": 8080},
                {"name": "server2", "host": "host2.example.com", "port": 8081},
            ]
        }

        schema = {
            "servers": {
                "types": {"servers": list},
                "list_item_schema": {
                    "required": ["name", "host", "port"],
                    "types": {"name": str, "host": str, "port": int},
                },
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_validate_list_of_objects_invalid_item_raises_error(self, config_validator):
        """Test that invalid item in list raises ConfigError."""
        config = {
            "servers": [
                {"name": "server1", "host": "host1.example.com", "port": 8080},
                {"name": "server2", "host": "host2.example.com"},  # Missing port
            ]
        }

        schema = {
            "servers": {
                "types": {"servers": list},
                "list_item_schema": {
                    "required": ["name", "host", "port"],
                    "types": {"name": str, "host": str, "port": int},
                },
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg
        assert "port" in error_msg

    # Custom Validation Rules Tests
    def test_add_custom_validation_rule_success(self, config_validator):
        """Test adding and using custom validation rules."""

        def validate_url(value):
            """Custom validator for URL format."""
            if not isinstance(value, str):
                raise ValueError("URL must be a string")
            if not value.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            return True

        config_validator.add_validation_rule("url_format", validate_url)

        config = {"api": {"base_url": "https://api.example.com"}}

        schema = {
            "api": {
                "types": {"base_url": str},
                "custom_validators": {"base_url": ["url_format"]},
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    def test_custom_validation_rule_failure_raises_error(self, config_validator):
        """Test that custom validation rule failure raises ConfigError."""

        def validate_url(value):
            if not value.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            return True

        config_validator.add_validation_rule("url_format", validate_url)

        config = {
            "api": {"base_url": "ftp://invalid.example.com"}  # Invalid URL scheme
        }

        schema = {
            "api": {
                "types": {"base_url": str},
                "custom_validators": {"base_url": ["url_format"]},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        assert "url" in error_msg or "http" in error_msg

    def test_multiple_custom_validation_rules(self, config_validator):
        """Test applying multiple custom validation rules to same field."""

        def validate_length(value):
            if len(value) < 8:
                raise ValueError("Must be at least 8 characters")
            return True

        def validate_complexity(value):
            if not any(c.isdigit() for c in value):
                raise ValueError("Must contain at least one digit")
            return True

        config_validator.add_validation_rule("min_length", validate_length)
        config_validator.add_validation_rule("complexity", validate_complexity)

        config = {"security": {"password": "secure123"}}

        schema = {
            "security": {
                "types": {"password": str},
                "custom_validators": {"password": ["min_length", "complexity"]},
            }
        }

        result = config_validator.validate_schema(config, schema)
        assert result is True

    # Error Reporting Tests
    def test_validation_error_detailed_path_reporting(self, config_validator):
        """Test that validation errors report detailed field paths."""
        config = {
            "app": {"database": {"connection": {"port": "invalid"}}}  # Should be int
        }

        schema = {
            "app": {
                "nested": {
                    "database": {"nested": {"connection": {"types": {"port": int}}}}
                }
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value)
        # Should contain the full path to the error
        assert "app.database.connection.port" in error_msg or all(
            part in error_msg for part in ["app", "database", "connection", "port"]
        )

    def test_validation_error_multiple_errors_collected(self, config_validator):
        """Test that multiple validation errors are collected and reported."""
        config = {
            "database": {
                "host": 12345,  # Should be string
                "port": "invalid",  # Should be int
                # Missing required field 'name'
            }
        }

        schema = {
            "database": {
                "required": ["host", "port", "name"],
                "types": {"host": str, "port": int, "name": str},
            }
        }

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(config, schema)

        error_msg = str(exc_info.value).lower()
        # Should mention multiple errors
        assert any(word in error_msg for word in ["multiple", "errors", "violations"])
        assert "host" in error_msg
        assert "port" in error_msg
        assert "name" in error_msg

    def test_validation_warning_collection(self, config_validator):
        """Test collection and reporting of validation warnings."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "deprecated_field": "value",  # Deprecated but not error
            }
        }

        schema = {
            "database": {
                "types": {"host": str, "port": int, "deprecated_field": str},
                "warnings": {
                    "deprecated_field": "This field is deprecated and will be removed"
                },
            }
        }

        result = config_validator.validate_schema(config, schema, collect_warnings=True)

        # Assume warnings are collected in validator instance
        assert result is True
        assert hasattr(config_validator, "warnings")
        assert len(config_validator.warnings) > 0
        assert "deprecated" in str(config_validator.warnings[0]).lower()

    # Edge Cases and Error Conditions Tests
    def test_validate_empty_config_with_schema(self, config_validator):
        """Test validation of completely empty config."""
        empty_config = {}

        schema = {"database": {"required": ["host", "port"]}}

        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_schema(empty_config, schema)

        error_msg = str(exc_info.value).lower()
        assert "required" in error_msg or "missing" in error_msg

    def test_validate_none_config_raises_error(self, config_validator):
        """Test that None config raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate(None)

        error_msg = str(exc_info.value).lower()
        assert "none" in error_msg or "null" in error_msg or "invalid" in error_msg

    def test_validate_non_dict_config_raises_error(self, config_validator):
        """Test that non-dictionary config raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate("not_a_dict")

        error_msg = str(exc_info.value).lower()
        assert "dictionary" in error_msg or "dict" in error_msg or "object" in error_msg

    def test_validate_circular_reference_handling(self, config_validator):
        """Test handling of circular references in config."""
        config = {"a": {"ref": None}, "b": {"ref": None}}
        # Create circular reference
        config["a"]["ref"] = config["b"]
        config["b"]["ref"] = config["a"]

        # Validator should handle this gracefully without infinite recursion
        result = config_validator.validate(config)
        assert result is True

    def test_validate_very_deep_nesting(self, config_validator):
        """Test validation of very deeply nested configuration."""
        # Create 20 levels deep nesting
        deep_config = {}
        current = deep_config

        for i in range(20):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        current["final_value"] = "test"

        # Should handle deep nesting without stack overflow
        result = config_validator.validate(deep_config)
        assert result is True

    def test_validate_large_config_performance(self, config_validator):
        """Test validation performance with large configuration."""
        import time

        # Create large config (1000 sections with 100 fields each)
        large_config = {}
        for i in range(100):  # Reduced for test performance
            section = {}
            for j in range(50):
                section[f"key_{j}"] = f"value_{i}_{j}"
            large_config[f"section_{i}"] = section

        start_time = time.time()
        result = config_validator.validate(large_config)
        validation_time = time.time() - start_time

        assert result is True
        assert validation_time < 1.0  # Should complete within 1 second

    def test_validate_unicode_and_special_characters(self, config_validator):
        """Test validation with Unicode and special characters."""
        unicode_config = {
            "app": {
                "title": "Приложение тест 测试 🚀",
                "description": "Spéciàl çhars & symbols: @#$%^&*()",
                "emoji_field": "🎉🔥💯",
                "multilingual": {
                    "english": "Hello World",
                    "chinese": "你好世界",
                    "arabic": "مرحبا بالعالم",
                    "russian": "Привет мир",
                },
            }
        }

        schema = {
            "app": {
                "types": {
                    "title": str,
                    "description": str,
                    "emoji_field": str,
                    "multilingual": dict,
                },
                "nested": {
                    "multilingual": {
                        "types": {
                            "english": str,
                            "chinese": str,
                            "arabic": str,
                            "russian": str,
                        }
                    }
                },
            }
        }

        result = config_validator.validate_schema(unicode_config, schema)
        assert result is True


class TestConfigValidatorIntegration:
    """Integration tests for config validator with ConfigManager."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance for integration testing."""
        return ConfigManager()

    def test_config_manager_with_validator_integration(self, config_manager):
        """Test ConfigManager integration with validator."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "user",
                "password": "pass",
            }
        }

        # ConfigManager should use validator internally
        config_manager.config = config
        result = config_manager.validate_config(config)
        assert result is True

    def test_config_manager_validation_with_schema(self, config_manager):
        """Test ConfigManager validation with custom schema."""
        config = {
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "endpoints": ["users", "data"],
            }
        }

        schema = {
            "api": {
                "required": ["base_url", "timeout"],
                "types": {"base_url": str, "timeout": int, "endpoints": list},
            }
        }

        config_manager.config = config
        result = config_manager.validate_config_schema(config, schema)
        assert result is True

    def test_config_manager_validation_error_propagation(self, config_manager):
        """Test that validation errors are properly propagated through ConfigManager."""
        invalid_config = {"database": {"port": "not_a_number"}}

        with pytest.raises(ConfigError) as exc_info:
            config_manager.validate_config(invalid_config)

        assert (
            "type" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )


class TestValidationRule:
    """Test suite for ValidationRule class."""

    def test_validation_rule_creation(self):
        """Test ValidationRule can be created with function."""

        def test_validator(value):
            return len(value) > 5

        rule = ValidationRule(
            "test_rule", test_validator, "Value must be longer than 5 characters"
        )
        assert rule.name == "test_rule"
        assert rule.description == "Value must be longer than 5 characters"
        assert callable(rule.validator)

    def test_validation_rule_execution_success(self):
        """Test ValidationRule execution with valid value."""

        def length_validator(value):
            if len(value) <= 5:
                raise ValueError("Too short")
            return True

        rule = ValidationRule("length_check", length_validator)

        # Should not raise exception
        result = rule.validate("long_enough_string")
        assert result is True

    def test_validation_rule_execution_failure(self):
        """Test ValidationRule execution with invalid value."""

        def length_validator(value):
            if len(value) <= 5:
                raise ValueError("Too short")
            return True

        rule = ValidationRule("length_check", length_validator)

        with pytest.raises(ValueError) as exc_info:
            rule.validate("short")

        assert "Too short" in str(exc_info.value)


class TestSchemaRegistry:
    """Test suite for SchemaRegistry class."""

    @pytest.fixture
    def schema_registry(self):
        """Create a SchemaRegistry instance for testing."""
        return SchemaRegistry()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_schema(self):
        """Sample schema for testing."""
        return {
            "database": {
                "required": ["host", "port"],
                "types": {"host": "str", "port": "int", "name": "str"},
                "constraints": {"port": {"min": 1, "max": 65535}},
            },
            "api": {"types": {"base_url": "str", "timeout": "int"}},
        }

    def test_schema_registry_initialization(self, schema_registry):
        """Test SchemaRegistry can be instantiated."""
        assert schema_registry is not None
        assert hasattr(schema_registry, "schemas")
        assert hasattr(schema_registry, "schema_versions")
        assert isinstance(schema_registry.schemas, dict)
        assert isinstance(schema_registry.schema_versions, dict)

    def test_schema_registry_has_built_in_schemas(self, schema_registry):
        """Test that built-in schemas are loaded during initialization."""
        # Check that AIM2 project schema is available
        assert "aim2_project" in schema_registry.schemas
        assert "database_only" in schema_registry.schemas
        assert "api_only" in schema_registry.schemas

        # Check that built-in schemas have versions
        assert "aim2_project" in schema_registry.schema_versions
        assert schema_registry.schema_versions["aim2_project"] == "1.0.0"

    def test_register_schema_success(self, schema_registry, sample_schema):
        """Test successful schema registration."""
        schema_name = "test_schema"
        version = "1.2.0"

        schema_registry.register_schema(schema_name, sample_schema, version)

        assert schema_name in schema_registry.schemas
        assert schema_registry.schemas[schema_name] == sample_schema
        assert schema_registry.schema_versions[schema_name] == version

    def test_register_schema_overwrites_existing(self, schema_registry, sample_schema):
        """Test that registering schema overwrites existing one."""
        schema_name = "test_schema"

        # Register first schema
        schema_registry.register_schema(schema_name, sample_schema, "1.0.0")

        # Register new schema with same name
        new_schema = {"different": {"types": {"field": "str"}}}
        schema_registry.register_schema(schema_name, new_schema, "2.0.0")

        assert schema_registry.schemas[schema_name] == new_schema
        assert schema_registry.schema_versions[schema_name] == "2.0.0"

    def test_get_schema_exists(self, schema_registry, sample_schema):
        """Test getting an existing schema."""
        schema_name = "test_schema"
        schema_registry.register_schema(schema_name, sample_schema)

        retrieved_schema = schema_registry.get_schema(schema_name)
        assert retrieved_schema == sample_schema

    def test_get_schema_not_exists(self, schema_registry):
        """Test getting a non-existent schema returns None."""
        retrieved_schema = schema_registry.get_schema("nonexistent_schema")
        assert retrieved_schema is None

    def test_get_schema_built_in_aim2_project(self, schema_registry):
        """Test getting the built-in AIM2 project schema."""
        schema = schema_registry.get_schema("aim2_project")

        assert schema is not None
        assert "database" in schema
        assert "api" in schema
        assert "logging" in schema
        assert "nlp" in schema
        assert "features" in schema

    def test_list_schemas_includes_built_in(self, schema_registry):
        """Test that list_schemas includes built-in schemas."""
        schemas = schema_registry.list_schemas()

        assert "aim2_project" in schemas
        assert "database_only" in schemas
        assert "api_only" in schemas
        assert isinstance(schemas, list)

    def test_list_schemas_includes_registered(self, schema_registry, sample_schema):
        """Test that list_schemas includes registered schemas."""
        schema_name = "custom_schema"
        schema_registry.register_schema(schema_name, sample_schema)

        schemas = schema_registry.list_schemas()
        assert schema_name in schemas

    def test_load_schema_from_json_file(self, schema_registry, temp_dir, sample_schema):
        """Test loading schema from JSON file."""
        schema_file = temp_dir / "test_schema.json"
        schema_file.write_text(json.dumps(sample_schema, indent=2))

        schema_registry.load_schema_from_file(str(schema_file))

        # Should be registered with filename as name
        assert "test_schema" in schema_registry.schemas
        loaded_schema = schema_registry.get_schema("test_schema")

        # Types should be normalized to Python types
        assert loaded_schema["database"]["types"]["host"] == str
        assert loaded_schema["database"]["types"]["port"] == int

    def test_load_schema_from_yaml_file(self, schema_registry, temp_dir):
        """Test loading schema from YAML file."""
        schema_yaml = """
database:
  required:
    - host
    - port
  types:
    host: str
    port: int
    ssl_enabled: bool
  constraints:
    port:
      min: 1
      max: 65535
"""
        schema_file = temp_dir / "test_schema.yaml"
        schema_file.write_text(schema_yaml)

        schema_registry.load_schema_from_file(str(schema_file))

        assert "test_schema" in schema_registry.schemas
        loaded_schema = schema_registry.get_schema("test_schema")

        # Check normalized types
        assert loaded_schema["database"]["types"]["host"] == str
        assert loaded_schema["database"]["types"]["port"] == int
        assert loaded_schema["database"]["types"]["ssl_enabled"] == bool

    def test_load_schema_with_custom_name(
        self, schema_registry, temp_dir, sample_schema
    ):
        """Test loading schema with custom name."""
        schema_file = temp_dir / "config.json"
        schema_file.write_text(json.dumps(sample_schema, indent=2))

        custom_name = "my_custom_schema"
        schema_registry.load_schema_from_file(str(schema_file), custom_name)

        assert custom_name in schema_registry.schemas
        assert "config" not in schema_registry.schemas  # Original filename not used

    def test_load_schema_nonexistent_file_raises_error(self, schema_registry, temp_dir):
        """Test that loading non-existent schema file raises error."""
        nonexistent_file = temp_dir / "nonexistent.json"

        with pytest.raises(ValueError) as exc_info:
            schema_registry.load_schema_from_file(str(nonexistent_file))

        assert "Schema file not found" in str(exc_info.value)

    def test_load_schema_unsupported_format_raises_error(
        self, schema_registry, temp_dir
    ):
        """Test that loading unsupported file format raises error."""
        unsupported_file = temp_dir / "schema.txt"
        unsupported_file.write_text("some content")

        with pytest.raises(ValueError) as exc_info:
            schema_registry.load_schema_from_file(str(unsupported_file))

        assert "Unsupported schema file format" in str(exc_info.value)

    def test_load_schema_invalid_json_raises_error(self, schema_registry, temp_dir):
        """Test that loading invalid JSON raises error."""
        invalid_json_file = temp_dir / "invalid.json"
        invalid_json_file.write_text('{"invalid": json}')  # Invalid JSON

        with pytest.raises(ValueError) as exc_info:
            schema_registry.load_schema_from_file(str(invalid_json_file))

        assert "Failed to load schema" in str(exc_info.value)

    def test_normalize_schema_types_string_types(self, schema_registry):
        """Test normalization of string type names to Python types."""
        schema = {
            "section": {
                "types": {
                    "str_field": "str",
                    "int_field": "int",
                    "float_field": "float",
                    "bool_field": "bool",
                    "list_field": "list",
                    "dict_field": "dict",
                    "none_field": "none",
                }
            }
        }

        normalized = schema_registry._normalize_schema_types(schema)

        types = normalized["section"]["types"]
        assert types["str_field"] == str
        assert types["int_field"] == int
        assert types["float_field"] == float
        assert types["bool_field"] == bool
        assert types["list_field"] == list
        assert types["dict_field"] == dict
        assert types["none_field"] == type(None)

    def test_normalize_schema_types_union_types(self, schema_registry):
        """Test normalization of union types (list of types)."""
        schema = {
            "section": {
                "types": {
                    "union_field": ["str", "int"],
                    "nullable_field": ["str", "null"],
                }
            }
        }

        normalized = schema_registry._normalize_schema_types(schema)

        types = normalized["section"]["types"]
        assert types["union_field"] == (str, int)
        assert types["nullable_field"] == (str, type(None))

    def test_normalize_schema_types_preserves_non_type_fields(self, schema_registry):
        """Test that normalization preserves non-type fields."""
        schema = {
            "section": {
                "required": ["field1", "field2"],
                "constraints": {"field1": {"min": 1, "max": 100}},
                "types": {"field1": "int", "field2": "str"},
            }
        }

        normalized = schema_registry._normalize_schema_types(schema)

        assert normalized["section"]["required"] == ["field1", "field2"]
        assert normalized["section"]["constraints"] == {
            "field1": {"min": 1, "max": 100}
        }
        assert normalized["section"]["types"]["field1"] == int
        assert normalized["section"]["types"]["field2"] == str

    def test_built_in_aim2_schema_structure(self, schema_registry):
        """Test that built-in AIM2 schema has expected structure."""
        schema = schema_registry.get_schema("aim2_project")

        # Check main sections exist
        expected_sections = [
            "database",
            "api",
            "logging",
            "nlp",
            "ontology",
            "features",
        ]
        for section in expected_sections:
            assert section in schema, f"Section {section} missing from AIM2 schema"

        # Check database section structure
        db_schema = schema["database"]
        assert "required" in db_schema
        assert "types" in db_schema
        assert "constraints" in db_schema

        # Check required database fields
        assert "host" in db_schema["required"]
        assert "port" in db_schema["required"]
        assert "name" in db_schema["required"]
        assert "username" in db_schema["required"]

        # Check type definitions
        assert db_schema["types"]["host"] == str
        assert db_schema["types"]["port"] == int

        # Check constraints
        assert "port" in db_schema["constraints"]
        assert "min" in db_schema["constraints"]["port"]
        assert "max" in db_schema["constraints"]["port"]

    def test_built_in_database_only_schema(self, schema_registry):
        """Test that database-only schema contains only database section."""
        schema = schema_registry.get_schema("database_only")

        assert "database" in schema
        assert len(schema) == 1  # Only database section

        # Should have same structure as database section in full schema
        full_schema = schema_registry.get_schema("aim2_project")
        assert schema["database"] == full_schema["database"]

    def test_built_in_api_only_schema(self, schema_registry):
        """Test that API-only schema contains only API section."""
        schema = schema_registry.get_schema("api_only")

        assert "api" in schema
        assert len(schema) == 1  # Only API section

        # Should have same structure as API section in full schema
        full_schema = schema_registry.get_schema("aim2_project")
        assert schema["api"] == full_schema["api"]


class TestValidationReport:
    """Test suite for ValidationReport class."""

    @pytest.fixture
    def validation_report(self):
        """Create a ValidationReport instance for testing."""
        return ValidationReport()

    def test_validation_report_initialization(self, validation_report):
        """Test ValidationReport can be instantiated with correct defaults."""
        assert validation_report is not None
        assert validation_report.is_valid is True
        assert isinstance(validation_report.errors, list)
        assert len(validation_report.errors) == 0
        assert isinstance(validation_report.warnings, list)
        assert len(validation_report.warnings) == 0
        assert validation_report.schema_name is None
        assert validation_report.schema_version is None
        assert isinstance(validation_report.validated_fields, list)
        assert isinstance(validation_report.missing_fields, list)
        assert isinstance(validation_report.extra_fields, list)

    def test_add_error_changes_validity(self, validation_report):
        """Test that adding an error changes is_valid to False."""
        assert validation_report.is_valid is True

        validation_report.add_error("Test error message")

        assert validation_report.is_valid is False
        assert len(validation_report.errors) == 1
        assert validation_report.errors[0] == "Test error message"

    def test_add_multiple_errors(self, validation_report):
        """Test adding multiple errors."""
        errors = ["Error 1", "Error 2", "Error 3"]

        for error in errors:
            validation_report.add_error(error)

        assert validation_report.is_valid is False
        assert len(validation_report.errors) == 3
        assert validation_report.errors == errors

    def test_add_warning_preserves_validity(self, validation_report):
        """Test that adding warnings doesn't change validity."""
        assert validation_report.is_valid is True

        validation_report.add_warning("Test warning message")

        assert validation_report.is_valid is True  # Should remain valid
        assert len(validation_report.warnings) == 1
        assert validation_report.warnings[0] == "Test warning message"

    def test_add_multiple_warnings(self, validation_report):
        """Test adding multiple warnings."""
        warnings = ["Warning 1", "Warning 2", "Warning 3"]

        for warning in warnings:
            validation_report.add_warning(warning)

        assert validation_report.is_valid is True
        assert len(validation_report.warnings) == 3
        assert validation_report.warnings == warnings

    def test_add_errors_and_warnings(self, validation_report):
        """Test adding both errors and warnings."""
        validation_report.add_error("An error occurred")
        validation_report.add_warning("A warning was issued")
        validation_report.add_error("Another error")

        assert validation_report.is_valid is False
        assert len(validation_report.errors) == 2
        assert len(validation_report.warnings) == 1
        assert "An error occurred" in validation_report.errors
        assert "Another error" in validation_report.errors
        assert "A warning was issued" in validation_report.warnings

    def test_to_dict_conversion(self, validation_report):
        """Test conversion of ValidationReport to dictionary."""
        # Set up test data
        validation_report.schema_name = "test_schema"
        validation_report.schema_version = "1.2.0"
        validation_report.add_error("Test error")
        validation_report.add_warning("Test warning")
        validation_report.validated_fields = ["field1", "field2"]
        validation_report.missing_fields = ["missing_field"]
        validation_report.extra_fields = ["extra_field"]

        result_dict = validation_report.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is False
        assert result_dict["errors"] == ["Test error"]
        assert result_dict["warnings"] == ["Test warning"]
        assert result_dict["schema_name"] == "test_schema"
        assert result_dict["schema_version"] == "1.2.0"
        assert result_dict["validated_fields"] == ["field1", "field2"]
        assert result_dict["missing_fields"] == ["missing_field"]
        assert result_dict["extra_fields"] == ["extra_field"]

    def test_to_dict_empty_report(self, validation_report):
        """Test dictionary conversion of empty report."""
        result_dict = validation_report.to_dict()

        assert result_dict["is_valid"] is True
        assert result_dict["errors"] == []
        assert result_dict["warnings"] == []
        assert result_dict["schema_name"] is None
        assert result_dict["schema_version"] is None
        assert result_dict["validated_fields"] == []
        assert result_dict["missing_fields"] == []
        assert result_dict["extra_fields"] == []

    def test_string_representation_valid_report(self, validation_report):
        """Test string representation of valid report."""
        validation_report.schema_name = "test_schema"
        validation_report.add_warning("Minor issue")

        str_repr = str(validation_report)

        assert "Validation Report (Schema: test_schema)" in str_repr
        assert "Valid: True" in str_repr
        assert "Warnings (1):" in str_repr
        assert "Minor issue" in str_repr

    def test_string_representation_invalid_report(self, validation_report):
        """Test string representation of invalid report."""
        validation_report.schema_name = "test_schema"
        validation_report.add_error("Critical error")
        validation_report.add_error("Another error")
        validation_report.add_warning("Warning message")
        validation_report.missing_fields = ["required_field"]
        validation_report.extra_fields = ["unexpected_field"]

        str_repr = str(validation_report)

        assert "Validation Report (Schema: test_schema)" in str_repr
        assert "Valid: False" in str_repr
        assert "Errors (2):" in str_repr
        assert "Critical error" in str_repr
        assert "Another error" in str_repr
        assert "Warnings (1):" in str_repr
        assert "Warning message" in str_repr
        assert "Missing fields: required_field" in str_repr
        assert "Extra fields: unexpected_field" in str_repr

    def test_string_representation_unknown_schema(self, validation_report):
        """Test string representation with unknown schema."""
        validation_report.add_error("Some error")

        str_repr = str(validation_report)

        assert "Validation Report (Schema: unknown)" in str_repr
        assert "Valid: False" in str_repr

    def test_string_representation_multiple_missing_and_extra_fields(
        self, validation_report
    ):
        """Test string representation with multiple missing and extra fields."""
        validation_report.missing_fields = ["field1", "field2", "field3"]
        validation_report.extra_fields = ["extra1", "extra2"]

        str_repr = str(validation_report)

        assert "Missing fields: field1, field2, field3" in str_repr
        assert "Extra fields: extra1, extra2" in str_repr

    def test_error_invalidates_report_immediately(self, validation_report):
        """Test that adding an error immediately invalidates the report."""
        # Start with valid report
        assert validation_report.is_valid is True

        # Add error and check immediately
        validation_report.add_error("Immediate error")
        assert validation_report.is_valid is False

        # Adding more errors should keep it invalid
        validation_report.add_error("Another error")
        assert validation_report.is_valid is False

        # Adding warnings should not change validity
        validation_report.add_warning("A warning")
        assert validation_report.is_valid is False

    def test_field_tracking(self, validation_report):
        """Test tracking of validated, missing, and extra fields."""
        # Test validated fields
        validation_report.validated_fields.extend(
            ["database.host", "database.port", "api.timeout"]
        )
        assert len(validation_report.validated_fields) == 3
        assert "database.host" in validation_report.validated_fields

        # Test missing fields
        validation_report.missing_fields.extend(["database.password", "logging.level"])
        assert len(validation_report.missing_fields) == 2
        assert "database.password" in validation_report.missing_fields

        # Test extra fields
        validation_report.extra_fields.extend(["unknown.field"])
        assert len(validation_report.extra_fields) == 1
        assert "unknown.field" in validation_report.extra_fields

    def test_schema_metadata(self, validation_report):
        """Test setting schema metadata."""
        validation_report.schema_name = "aim2_project"
        validation_report.schema_version = "2.1.0"

        assert validation_report.schema_name == "aim2_project"
        assert validation_report.schema_version == "2.1.0"

        # Should be reflected in dictionary conversion
        result_dict = validation_report.to_dict()
        assert result_dict["schema_name"] == "aim2_project"
        assert result_dict["schema_version"] == "2.1.0"


class TestSchemaMigration:
    """Test suite for SchemaMigration class."""

    @pytest.fixture
    def sample_migration_function(self):
        """Sample migration function for testing."""

        def migrate_config(config):
            # Example migration: rename 'old_field' to 'new_field'
            migrated = config.copy()
            if "database" in migrated and "old_field" in migrated["database"]:
                migrated["database"]["new_field"] = migrated["database"].pop(
                    "old_field"
                )
            return migrated

        return migrate_config

    def test_schema_migration_initialization(self, sample_migration_function):
        """Test SchemaMigration can be instantiated."""
        migration = SchemaMigration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_function=sample_migration_function,
            description="Rename old_field to new_field",
        )

        assert migration is not None
        assert migration.from_version == "1.0.0"
        assert migration.to_version == "1.1.0"
        assert migration.migration_function == sample_migration_function
        assert migration.description == "Rename old_field to new_field"

    def test_schema_migration_initialization_without_description(
        self, sample_migration_function
    ):
        """Test SchemaMigration initialization without description."""
        migration = SchemaMigration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_function=sample_migration_function,
        )

        assert migration.description == ""

    def test_schema_migration_apply(self, sample_migration_function):
        """Test applying a schema migration."""
        migration = SchemaMigration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_function=sample_migration_function,
        )

        original_config = {"database": {"host": "localhost", "old_field": "old_value"}}

        migrated_config = migration.apply(original_config)

        # Should have renamed the field
        assert "old_field" not in migrated_config["database"]
        assert "new_field" in migrated_config["database"]
        assert migrated_config["database"]["new_field"] == "old_value"
        assert migrated_config["database"]["host"] == "localhost"

    def test_schema_migration_apply_no_matching_field(self, sample_migration_function):
        """Test applying migration when target field doesn't exist."""
        migration = SchemaMigration(
            from_version="1.0.0",
            to_version="1.1.0",
            migration_function=sample_migration_function,
        )

        original_config = {"database": {"host": "localhost", "port": 5432}}

        migrated_config = migration.apply(original_config)

        # Should remain unchanged
        assert migrated_config == original_config


class TestSchemaMigrator:
    """Test suite for SchemaMigrator class."""

    @pytest.fixture
    def schema_migrator(self):
        """Create a SchemaMigrator instance for testing."""
        return SchemaMigrator()

    @pytest.fixture
    def sample_migration_1_to_2(self):
        """Sample migration from version 1.0.0 to 2.0.0."""

        def migrate(config):
            migrated = config.copy()
            # Add new security section
            if "security" not in migrated:
                migrated["security"] = {"enabled": False}
            return migrated

        return SchemaMigration(
            from_version="1.0.0",
            to_version="2.0.0",
            migration_function=migrate,
            description="Add security section",
        )

    @pytest.fixture
    def sample_migration_2_to_3(self):
        """Sample migration from version 2.0.0 to 3.0.0."""

        def migrate(config):
            migrated = config.copy()
            # Rename logging.log_level to logging.level
            if "logging" in migrated and "log_level" in migrated["logging"]:
                migrated["logging"]["level"] = migrated["logging"].pop("log_level")
            return migrated

        return SchemaMigration(
            from_version="2.0.0",
            to_version="3.0.0",
            migration_function=migrate,
            description="Rename log_level to level",
        )

    def test_schema_migrator_initialization(self, schema_migrator):
        """Test SchemaMigrator can be instantiated."""
        assert schema_migrator is not None
        assert hasattr(schema_migrator, "migrations")
        assert isinstance(schema_migrator.migrations, dict)

    def test_schema_migrator_has_built_in_migrations(self, schema_migrator):
        """Test that built-in migrations are loaded during initialization."""
        # Should have built-in migrations for aim2_project
        assert "aim2_project" in schema_migrator.migrations
        assert len(schema_migrator.migrations["aim2_project"]) > 0

        # Check the built-in migration structure
        aim2_migrations = schema_migrator.migrations["aim2_project"]
        migration = aim2_migrations[0]
        assert migration.from_version == "1.0.0"
        assert migration.to_version == "1.1.0"

    def test_register_migration(self, schema_migrator, sample_migration_1_to_2):
        """Test registering a new migration."""
        schema_name = "test_schema"

        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)

        assert schema_name in schema_migrator.migrations
        assert len(schema_migrator.migrations[schema_name]) == 1
        assert schema_migrator.migrations[schema_name][0] == sample_migration_1_to_2

    def test_register_multiple_migrations(
        self, schema_migrator, sample_migration_1_to_2, sample_migration_2_to_3
    ):
        """Test registering multiple migrations for same schema."""
        schema_name = "test_schema"

        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)
        schema_migrator.register_migration(schema_name, sample_migration_2_to_3)

        assert len(schema_migrator.migrations[schema_name]) == 2

        # Should be sorted by version
        migrations = schema_migrator.migrations[schema_name]
        assert migrations[0].from_version == "1.0.0"
        assert migrations[1].from_version == "2.0.0"

    def test_migrate_config_same_version(self, schema_migrator):
        """Test migration when from and to versions are the same."""
        config = {"database": {"host": "localhost"}}

        result = schema_migrator.migrate_config(config, "test_schema", "1.0.0", "1.0.0")

        # Should return config unchanged
        assert result == config

    def test_migrate_config_single_step(self, schema_migrator, sample_migration_1_to_2):
        """Test migration with single step."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)

        config = {"database": {"host": "localhost"}}

        result = schema_migrator.migrate_config(config, schema_name, "1.0.0", "2.0.0")

        # Should have applied migration
        assert "security" in result
        assert result["security"]["enabled"] is False
        assert result["database"]["host"] == "localhost"

    def test_migrate_config_multi_step(
        self, schema_migrator, sample_migration_1_to_2, sample_migration_2_to_3
    ):
        """Test migration with multiple steps."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)
        schema_migrator.register_migration(schema_name, sample_migration_2_to_3)

        config = {"database": {"host": "localhost"}, "logging": {"log_level": "INFO"}}

        result = schema_migrator.migrate_config(config, schema_name, "1.0.0", "3.0.0")

        # Should have applied both migrations
        assert "security" in result  # From 1.0.0 -> 2.0.0
        assert result["security"]["enabled"] is False
        assert "log_level" not in result["logging"]  # From 2.0.0 -> 3.0.0
        assert result["logging"]["level"] == "INFO"

    def test_migrate_config_no_schema_raises_error(self, schema_migrator):
        """Test migration with non-existent schema raises error."""
        config = {"database": {"host": "localhost"}}

        with pytest.raises(ValueError) as exc_info:
            schema_migrator.migrate_config(
                config, "nonexistent_schema", "1.0.0", "2.0.0"
            )

        assert "No migrations available" in str(exc_info.value)

    def test_migrate_config_no_path_raises_error(
        self, schema_migrator, sample_migration_1_to_2
    ):
        """Test migration with no available path raises error."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)

        config = {"database": {"host": "localhost"}}

        # Try to migrate from 2.0.0 to 3.0.0 but no such migration exists
        with pytest.raises(ValueError) as exc_info:
            schema_migrator.migrate_config(config, schema_name, "2.0.0", "3.0.0")

        assert "No migration path found" in str(exc_info.value)

    def test_get_latest_version_exists(
        self, schema_migrator, sample_migration_1_to_2, sample_migration_2_to_3
    ):
        """Test getting latest version when migrations exist."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)
        schema_migrator.register_migration(schema_name, sample_migration_2_to_3)

        latest_version = schema_migrator.get_latest_version(schema_name)

        assert latest_version == "3.0.0"

    def test_get_latest_version_no_schema(self, schema_migrator):
        """Test getting latest version for non-existent schema."""
        latest_version = schema_migrator.get_latest_version("nonexistent_schema")

        assert latest_version is None

    def test_get_latest_version_no_migrations(self, schema_migrator):
        """Test getting latest version when schema has no migrations."""
        schema_name = "empty_schema"
        schema_migrator.migrations[schema_name] = []

        latest_version = schema_migrator.get_latest_version(schema_name)

        assert latest_version is None

    def test_is_migration_needed_different_versions(self, schema_migrator):
        """Test checking if migration is needed between different versions."""
        result = schema_migrator.is_migration_needed("test_schema", "1.0.0", "2.0.0")
        assert result is True

    def test_is_migration_needed_same_versions(self, schema_migrator):
        """Test checking if migration is needed for same versions."""
        result = schema_migrator.is_migration_needed("test_schema", "1.0.0", "1.0.0")
        assert result is False

    def test_is_migration_needed_semantic_versions(self, schema_migrator):
        """Test migration needed check with semantic versioning."""
        # Different patch versions
        result = schema_migrator.is_migration_needed("test_schema", "1.0.0", "1.0.1")
        assert result is True

        # Different minor versions
        result = schema_migrator.is_migration_needed("test_schema", "1.0.0", "1.1.0")
        assert result is True

        # Different major versions
        result = schema_migrator.is_migration_needed("test_schema", "1.0.0", "2.0.0")
        assert result is True

    def test_find_migration_path_direct(self, schema_migrator, sample_migration_1_to_2):
        """Test finding direct migration path."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)

        path = schema_migrator._find_migration_path(schema_name, "1.0.0", "2.0.0")

        assert path is not None
        assert len(path) == 1
        assert path[0] == sample_migration_1_to_2

    def test_find_migration_path_multi_step(
        self, schema_migrator, sample_migration_1_to_2, sample_migration_2_to_3
    ):
        """Test finding multi-step migration path."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)
        schema_migrator.register_migration(schema_name, sample_migration_2_to_3)

        path = schema_migrator._find_migration_path(schema_name, "1.0.0", "3.0.0")

        assert path is not None
        assert len(path) == 2
        assert path[0] == sample_migration_1_to_2
        assert path[1] == sample_migration_2_to_3

    def test_find_migration_path_no_path(
        self, schema_migrator, sample_migration_1_to_2
    ):
        """Test finding migration path when no path exists."""
        schema_name = "test_schema"
        schema_migrator.register_migration(schema_name, sample_migration_1_to_2)

        # Try to find path from 3.0.0 to 4.0.0 - no such migration exists
        path = schema_migrator._find_migration_path(schema_name, "3.0.0", "4.0.0")

        assert path is None

    def test_find_migration_path_no_schema(self, schema_migrator):
        """Test finding migration path for non-existent schema."""
        path = schema_migrator._find_migration_path(
            "nonexistent_schema", "1.0.0", "2.0.0"
        )

        assert path is None

    def test_built_in_aim2_migration(self, schema_migrator):
        """Test the built-in AIM2 project migration."""
        # Get the built-in migration
        aim2_migrations = schema_migrator.migrations["aim2_project"]
        migration = aim2_migrations[0]

        # Test the migration function
        config = {
            "database": {"host": "localhost"},
            "logging": {"log_level": "INFO"},  # Old field name
        }

        migrated_config = migration.apply(config)

        # Should add security section
        assert "security" in migrated_config
        assert migrated_config["security"]["encrypt_config"] is False
        assert migrated_config["security"]["allowed_hosts"] == []
        assert migrated_config["security"]["cors_enabled"] is False

        # Should rename log_level to level
        assert "log_level" not in migrated_config["logging"]
        assert migrated_config["logging"]["level"] == "INFO"

    def test_built_in_migration_preserves_existing_fields(self, schema_migrator):
        """Test that built-in migration preserves existing fields."""
        aim2_migrations = schema_migrator.migrations["aim2_project"]
        migration = aim2_migrations[0]

        config = {
            "database": {"host": "localhost", "port": 5432},
            "logging": {"level": "DEBUG", "format": "%(message)s"},
            "security": {"encrypt_config": True},  # Already exists
        }

        migrated_config = migration.apply(config)

        # Should preserve existing database config
        assert migrated_config["database"]["host"] == "localhost"
        assert migrated_config["database"]["port"] == 5432

        # Should preserve existing logging config
        assert migrated_config["logging"]["level"] == "DEBUG"
        assert migrated_config["logging"]["format"] == "%(message)s"

        # Should preserve existing security config
        assert migrated_config["security"]["encrypt_config"] is True


class TestEnhancedConfigValidator:
    """Test suite for enhanced ConfigValidator functionality."""

    @pytest.fixture
    def config_validator(self):
        """Create a ConfigValidator instance for testing."""
        return ConfigValidator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "project": {
                "name": "Test Project",
                "version": "1.0.0",
                "description": "Test configuration",
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "test_user",
                "password": "secure_pass",
                "ssl_mode": "require",
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "retry_attempts": 3,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(message)s",
                "handlers": ["console"],
            },
            "nlp": {
                "models": {"ner": "bert-base-uncased", "relationship": "roberta-large"},
                "max_sequence_length": 512,
                "batch_size": 32,
            },
            "features": {"enable_caching": True, "debug_mode": False, "max_workers": 4},
        }

    def test_validate_with_schema_success(self, config_validator, sample_config):
        """Test successful validation using named schema."""
        result = config_validator.validate_with_schema(sample_config, "aim2_project")
        assert result is True

    def test_validate_with_schema_nonexistent_schema_raises_error(
        self, config_validator, sample_config
    ):
        """Test validation with non-existent schema raises error."""
        with pytest.raises(ValueError) as exc_info:
            config_validator.validate_with_schema(sample_config, "nonexistent_schema")

        assert "Schema not found" in str(exc_info.value)

    def test_validate_with_schema_return_report(self, config_validator, sample_config):
        """Test validation with return report option."""
        report = config_validator.validate_with_schema(
            sample_config, "aim2_project", return_report=True
        )

        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.schema_name == "aim2_project"
        assert len(report.validated_fields) > 0

    def test_validate_schema_with_report_success(self, config_validator, sample_config):
        """Test schema validation with detailed report."""
        schema = config_validator.schema_registry.get_schema("aim2_project")

        report = config_validator.validate_schema_with_report(
            sample_config, schema, "aim2_project"
        )

        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.schema_name == "aim2_project"
        assert len(report.errors) == 0
        assert len(report.validated_fields) > 0

    def test_validate_schema_with_report_validation_errors(self, config_validator):
        """Test schema validation with report capturing errors."""
        schema = config_validator.schema_registry.get_schema("aim2_project")

        # Invalid config - missing required fields and wrong types
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number"  # Wrong type
                # Missing required fields
            }
        }

        report = config_validator.validate_schema_with_report(
            invalid_config, schema, "aim2_project"
        )

        assert isinstance(report, ValidationReport)
        assert report.is_valid is False
        assert len(report.errors) > 0
        assert len(report.missing_fields) > 0

        # Check for specific errors
        error_text = " ".join(report.errors)
        assert "missing" in error_text.lower() or "required" in error_text.lower()

    def test_validate_aim2_config_success(self, config_validator, sample_config):
        """Test validation against built-in AIM2 schema."""
        result = config_validator.validate_aim2_config(sample_config)
        assert result is True

    def test_validate_aim2_config_with_report(self, config_validator, sample_config):
        """Test AIM2 validation with detailed report."""
        report = config_validator.validate_aim2_config(
            sample_config, return_report=True
        )

        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert report.schema_name == "aim2_project"

    def test_load_and_register_schema_json(self, config_validator, temp_dir):
        """Test loading and registering schema from JSON file."""
        schema = {
            "test_section": {
                "required": ["field1"],
                "types": {"field1": "str", "field2": "int"},
            }
        }

        schema_file = temp_dir / "test_schema.json"
        schema_file.write_text(json.dumps(schema))

        config_validator.load_and_register_schema(
            str(schema_file), "custom_schema", "1.5.0"
        )

        # Should be available in registry
        assert "custom_schema" in config_validator.get_available_schemas()
        assert (
            config_validator.schema_registry.schema_versions["custom_schema"] == "1.5.0"
        )

        # Should be usable for validation
        test_config = {"test_section": {"field1": "value"}}
        result = config_validator.validate_with_schema(test_config, "custom_schema")
        assert result is True

    def test_load_and_register_schema_yaml(self, config_validator, temp_dir):
        """Test loading and registering schema from YAML file."""
        schema_yaml = """
test_section:
  required:
    - field1
  types:
    field1: str
    field2: int
"""
        schema_file = temp_dir / "test_schema.yaml"
        schema_file.write_text(schema_yaml)

        config_validator.load_and_register_schema(str(schema_file))

        # Should be available (using filename as name)
        assert "test_schema" in config_validator.get_available_schemas()

    def test_get_available_schemas_includes_built_in_and_custom(
        self, config_validator, temp_dir
    ):
        """Test that available schemas includes both built-in and custom schemas."""
        # Add a custom schema
        schema = {"section": {"types": {"field": "str"}}}
        schema_file = temp_dir / "custom.json"
        schema_file.write_text(json.dumps(schema))
        config_validator.load_and_register_schema(str(schema_file))

        schemas = config_validator.get_available_schemas()

        # Should include built-in schemas
        assert "aim2_project" in schemas
        assert "database_only" in schemas

        # Should include custom schema
        assert "custom" in schemas

    def test_migrate_config_success(self, config_validator):
        """Test configuration migration."""

        # Register a test migration
        def migrate_func(config):
            migrated = config.copy()
            if "old_section" in migrated:
                migrated["new_section"] = migrated.pop("old_section")
            return migrated

        migration = SchemaMigration(
            from_version="1.0.0", to_version="2.0.0", migration_function=migrate_func
        )
        config_validator.register_schema_migration("test_schema", migration)

        config = {"old_section": {"field": "value"}}

        migrated = config_validator.migrate_config(
            config, "test_schema", "1.0.0", "2.0.0"
        )

        assert "old_section" not in migrated
        assert "new_section" in migrated
        assert migrated["new_section"]["field"] == "value"

    def test_register_schema_migration(self, config_validator):
        """Test registering a schema migration."""

        def dummy_migrate(config):
            return config

        migration = SchemaMigration(
            from_version="1.0.0", to_version="2.0.0", migration_function=dummy_migrate
        )

        config_validator.register_schema_migration("test_schema", migration)

        # Should be registered in migrator
        assert "test_schema" in config_validator.migrator.migrations
        assert len(config_validator.migrator.migrations["test_schema"]) == 1

    def test_get_latest_schema_version(self, config_validator):
        """Test getting latest schema version."""
        # Should return version for built-in schema
        version = config_validator.get_latest_schema_version("aim2_project")
        assert version == "1.1.0"  # Based on built-in migration

        # Should return None for non-existent schema
        version = config_validator.get_latest_schema_version("nonexistent")
        assert version is None

    def test_validate_and_migrate_no_migration_needed(
        self, config_validator, sample_config
    ):
        """Test validate and migrate when no migration is needed."""
        result = config_validator.validate_and_migrate(
            sample_config,
            "aim2_project",
            config_version="1.1.0",
            target_version="1.1.0",
        )

        # Should return validation result only (no migration occurred)
        assert result is True

    def test_validate_and_migrate_with_migration(self, config_validator):
        """Test validate and migrate when migration is needed."""
        # Register test schema and migration
        schema = {"section": {"types": {"new_field": str}}}
        config_validator.schema_registry.register_schema("test_schema", schema, "2.0.0")

        def migrate_func(config):
            migrated = config.copy()
            if "section" in migrated and "old_field" in migrated["section"]:
                migrated["section"]["new_field"] = migrated["section"].pop("old_field")
            return migrated

        migration = SchemaMigration(
            from_version="1.0.0", to_version="2.0.0", migration_function=migrate_func
        )
        config_validator.register_schema_migration("test_schema", migration)

        config = {"section": {"old_field": "value"}}

        result = config_validator.validate_and_migrate(
            config, "test_schema", config_version="1.0.0", target_version="2.0.0"
        )

        # Should return tuple of (migrated_config, validation_result)
        assert isinstance(result, tuple)
        migrated_config, validation_result = result

        assert "old_field" not in migrated_config["section"]
        assert "new_field" in migrated_config["section"]
        assert migrated_config["section"]["new_field"] == "value"
        assert validation_result is True

    def test_validate_and_migrate_with_report(self, config_validator, sample_config):
        """Test validate and migrate with detailed report."""
        result = config_validator.validate_and_migrate(
            sample_config, "aim2_project", return_report=True
        )

        # If migration occurred, result is a tuple (migrated_config, ValidationReport)
        # If no migration, result is just the ValidationReport
        if isinstance(result, tuple):
            migrated_config, validation_report = result
            assert isinstance(validation_report, ValidationReport)
            assert validation_report.is_valid is True
        else:
            assert isinstance(result, ValidationReport)
            assert result.is_valid is True

    def test_validate_and_migrate_auto_detect_version(self, config_validator):
        """Test validate and migrate with auto-detected config version."""
        config = {
            "project": {"name": "Test Project", "version": "1.0.0"},
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "db",
                "username": "user",
            },
        }

        # Should detect version from config
        result = config_validator.validate_and_migrate(config, "aim2_project")

        # Should work without explicit version
        # If migration occurred, result is a tuple (migrated_config, validation_result)
        # If no migration, result is just the validation result
        if isinstance(result, tuple):
            migrated_config, validation_result = result
            assert validation_result is True
            # Verify security section was added by migration
            assert "security" in migrated_config
        else:
            assert result is True

    def test_detect_config_version_from_project_section(self, config_validator):
        """Test config version detection from project section."""
        config = {"project": {"version": "2.1.0"}, "database": {"host": "localhost"}}

        version = config_validator._detect_config_version(config, "aim2_project")
        assert version == "2.1.0"

    def test_detect_config_version_from_top_level(self, config_validator):
        """Test config version detection from top-level version field."""
        config = {"version": "1.5.0", "database": {"host": "localhost"}}

        version = config_validator._detect_config_version(config, "aim2_project")
        assert version == "1.5.0"

    def test_detect_config_version_from_schema_version(self, config_validator):
        """Test config version detection from schema_version field."""
        config = {"schema_version": "3.0.0", "database": {"host": "localhost"}}

        version = config_validator._detect_config_version(config, "aim2_project")
        assert version == "3.0.0"

    def test_detect_config_version_default(self, config_validator):
        """Test config version detection defaults to 1.0.0."""
        config = {"database": {"host": "localhost"}}

        version = config_validator._detect_config_version(config, "aim2_project")
        assert version == "1.0.0"

    def test_enhanced_validator_with_registry_integration(
        self, config_validator, sample_config
    ):
        """Test that enhanced validator properly integrates with schema registry."""
        # Should be able to validate against built-in schemas
        result = config_validator.validate_with_schema(sample_config, "aim2_project")
        assert result is True

        result = config_validator.validate_with_schema(
            {"database": sample_config["database"]}, "database_only"
        )
        assert result is True

        result = config_validator.validate_with_schema(
            {"api": sample_config["api"]}, "api_only"
        )
        assert result is True

    def test_enhanced_validator_strict_mode_integration(self, config_validator):
        """Test enhanced validator with strict mode validation."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "username": "user",
                "password": "pass",
                "extra_field": "not_allowed",  # Extra field
            }
        }

        # Should pass in non-strict mode
        result = config_validator.validate_with_schema(
            config, "database_only", strict=False
        )
        assert result is True

        # Should fail in strict mode
        with pytest.raises(ConfigError) as exc_info:
            config_validator.validate_with_schema(config, "database_only", strict=True)

        assert (
            "extra" in str(exc_info.value).lower()
            or "strict" in str(exc_info.value).lower()
        )

    def test_enhanced_validator_list_item_validation(self, config_validator, temp_dir):
        """Test enhanced validator with list item schema validation."""
        schema = {
            "servers": {
                "types": {"servers": "list"},
                "list_item_schema": {
                    "required": ["name", "host"],
                    "types": {"name": "str", "host": "str", "port": "int"},
                },
            }
        }

        schema_file = temp_dir / "list_schema.json"
        schema_file.write_text(json.dumps(schema))
        config_validator.load_and_register_schema(str(schema_file), "list_schema")

        # Valid config
        valid_config = {
            "servers": [
                {"name": "server1", "host": "host1.com", "port": 8080},
                {"name": "server2", "host": "host2.com", "port": 8081},
            ]
        }

        result = config_validator.validate_with_schema(valid_config, "list_schema")
        assert result is True

        # Invalid config - missing required field
        invalid_config = {
            "servers": [
                {"name": "server1", "host": "host1.com"},
                {"name": "server2"},  # Missing host
            ]
        }

        with pytest.raises(ConfigError):
            config_validator.validate_with_schema(invalid_config, "list_schema")

    def test_enhanced_validator_nested_schema_validation(
        self, config_validator, temp_dir
    ):
        """Test enhanced validator with nested schema validation."""
        schema = {
            "app": {
                "types": {"services": "dict"},
                "nested": {
                    "services": {
                        "types": {"database": "dict", "cache": "dict"},
                        "nested": {
                            "database": {
                                "required": ["host", "port"],
                                "types": {"host": "str", "port": "int"},
                            },
                            "cache": {
                                "required": ["type"],
                                "types": {"type": "str", "ttl": "int"},
                            },
                        },
                    }
                },
            }
        }

        schema_file = temp_dir / "nested_schema.json"
        schema_file.write_text(json.dumps(schema))
        config_validator.load_and_register_schema(str(schema_file), "nested_schema")

        # Valid nested config
        valid_config = {
            "app": {
                "services": {
                    "database": {"host": "db.example.com", "port": 5432},
                    "cache": {"type": "redis", "ttl": 3600},
                }
            }
        }

        result = config_validator.validate_with_schema(valid_config, "nested_schema")
        assert result is True
