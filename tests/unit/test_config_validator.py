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
"""

import pytest

# Import the module to be tested (will fail initially in TDD)
try:
    from aim2_project.aim2_utils.config_manager import ConfigManager, ConfigError
    from aim2_project.aim2_utils.config_validator import ConfigValidator, ValidationRule
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
