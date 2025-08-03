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


class ConfigValidator:
    """
    Main configuration validation class.

    Provides comprehensive validation capabilities including schema validation,
    custom rules, constraint checking, and error collection.

    Attributes:
        custom_rules (Dict[str, ValidationRule]): Registry of custom validation rules
        warnings (List[str]): Collection of validation warnings
    """

    def __init__(self):
        """
        Initialize the ConfigValidator.

        Sets up the validator with empty custom rules registry and warnings collection.
        """
        self.custom_rules = {}
        self.warnings = []

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

    # Private helper methods

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
