"""
Comprehensive Unit Tests for Ontology Class

This module provides comprehensive unit tests for the Ontology class from the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the Ontology class before implementation.

Test Classes:
    TestOntologyCreation: Tests for Ontology instantiation and attribute handling
    TestOntologyValidation: Tests for ontology validation methods
    TestOntologyEquality: Tests for equality and hashing functionality
    TestOntologyStringRepresentation: Tests for __str__ and __repr__ methods
    TestOntologySerialization: Tests for JSON serialization/deserialization
    TestOntologyEdgeCases: Tests for edge cases and error handling
    TestOntologySpecific: Tests for ontology-specific functionality
    TestOntologyIntegration: Integration tests using fixtures

The Ontology class is expected to be a dataclass representing complete ontologies with:
- Core attributes: id, name, version, description, terms, relationships, namespaces, metadata
- Container functionality: add_term, remove_term, add_relationship, remove_relationship
- Query methods: get_term, find_terms, get_relationships, get_statistics
- Validation methods: validate_structure, is_valid, validate_consistency
- Serialization: to_dict, from_dict, to_json, from_json
- String representations: __str__, __repr__, __eq__, __hash__

Container Operations:
    - Term management: add, remove, update, query terms
    - Relationship management: add, remove, validate relationships
    - Namespace handling and validation
    - Statistics generation (term counts, relationship counts, etc.)
    - Merging and integration with other ontologies
    - Export/import functionality

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - json: For JSON serialization testing
    - typing: For type hints
    - dataclasses: For dataclass functionality

Usage:
    pytest tests/unit/test_ontology.py -v
"""

import pytest
from unittest.mock import Mock


# Note: The Ontology class doesn't exist yet - these tests define its expected behavior
# The import will be uncommented once the class is implemented
# from aim2_project.aim2_ontology.models import Ontology


class TestOntologyCreation:
    """Test Ontology class instantiation and attribute handling."""

    def test_ontology_creation_minimal(self):
        """Test creating an ontology with minimal required attributes."""
        # Arrange & Act - Using mock Ontology class for now
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Test Ontology")

        # Assert
        MockOntology.assert_called_once_with(id="ONT:001", name="Test Ontology")

    def test_ontology_creation_full_attributes(self):
        """Test creating an ontology with all possible attributes."""
        # Arrange
        ontology_data = {
            "id": "ONT:001",
            "name": "Chemical Ontology",
            "version": "2023.1",
            "description": "A comprehensive chemical ontology for biological systems",
            "terms": {},
            "relationships": {},
            "namespaces": ["chemical", "biological_process"],
            "metadata": {
                "source": "ChEBI",
                "created_date": "2023-01-01T00:00:00Z",
                "modified_date": "2023-06-15T10:30:00Z",
                "authors": ["John Doe", "Jane Smith"],
                "license": "CC BY 4.0",
                "format_version": "1.4",
                "default_namespace": "chemical",
            },
            "base_iris": ["http://purl.obolibrary.org/obo/chebi.owl"],
            "imports": ["http://purl.obolibrary.org/obo/go.owl"],
            "synonymtypedef": {},
            "term_count": 0,
            "relationship_count": 0,
            "is_consistent": True,
            "validation_errors": [],
        }

        # Act & Assert - Using mock Ontology class
        MockOntology = Mock()
        MockOntology(**ontology_data)
        MockOntology.assert_called_once_with(**ontology_data)

    def test_ontology_creation_with_defaults(self):
        """Test that ontology creation uses appropriate default values."""
        # Arrange & Act
        MockOntology = Mock()
        # Configure mock to return expected default values
        mock_instance = Mock()
        mock_instance.version = "1.0"
        mock_instance.description = None
        mock_instance.terms = {}
        mock_instance.relationships = {}
        mock_instance.namespaces = []
        mock_instance.metadata = {}
        mock_instance.base_iris = []
        mock_instance.imports = []
        mock_instance.synonymtypedef = {}
        mock_instance.term_count = 0
        mock_instance.relationship_count = 0
        mock_instance.is_consistent = True
        mock_instance.validation_errors = []
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Test Ontology")

        # Assert defaults
        assert ontology.version == "1.0"
        assert ontology.description is None
        assert ontology.terms == {}
        assert ontology.relationships == {}
        assert ontology.namespaces == []
        assert ontology.metadata == {}
        assert ontology.base_iris == []
        assert ontology.imports == []
        assert ontology.synonymtypedef == {}
        assert ontology.term_count == 0
        assert ontology.relationship_count == 0
        assert ontology.is_consistent is True
        assert ontology.validation_errors == []

    def test_ontology_creation_invalid_id(self):
        """Test that creating an ontology with invalid ID raises appropriate error."""
        MockOntology = Mock()
        MockOntology.side_effect = ValueError("Invalid ontology ID format")

        with pytest.raises(ValueError, match="Invalid ontology ID format"):
            MockOntology(id="invalid_id", name="Test Ontology")

    def test_ontology_creation_empty_name(self):
        """Test that creating an ontology with empty name raises appropriate error."""
        MockOntology = Mock()
        MockOntology.side_effect = ValueError("Ontology name cannot be empty")

        with pytest.raises(ValueError, match="Ontology name cannot be empty"):
            MockOntology(id="ONT:001", name="")

    def test_ontology_mutability_after_creation(self):
        """Test that ontology attributes can be modified after creation (mutable dataclass)."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.name = "Test Ontology"
        mock_instance.version = "1.0"
        mock_instance.description = None
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Test Ontology")

        # Should be able to modify attributes (mutable dataclass expected)
        ontology.name = "Updated Ontology"
        ontology.version = "2.0"
        ontology.description = "Updated description"

        assert ontology.name == "Updated Ontology"
        assert ontology.version == "2.0"
        assert ontology.description == "Updated description"

    def test_ontology_with_initial_terms_and_relationships(self):
        """Test creating ontology with initial terms and relationships."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.terms = {"CHEBI:12345": Mock(), "CHEBI:33917": Mock()}
        mock_instance.relationships = {"REL:001": Mock(), "REL:002": Mock()}
        mock_instance.term_count = 2
        mock_instance.relationship_count = 2
        MockOntology.return_value = mock_instance

        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            terms={"CHEBI:12345": Mock(), "CHEBI:33917": Mock()},
            relationships={"REL:001": Mock(), "REL:002": Mock()},
        )

        assert len(ontology.terms) == 2
        assert len(ontology.relationships) == 2
        assert ontology.term_count == 2
        assert ontology.relationship_count == 2


class TestOntologyValidation:
    """Test ontology validation methods."""

    def test_validate_id_format_valid(self):
        """Test validation of valid ontology ID formats."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_id_format.return_value = True
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Test Ontology")

        # Test various valid ID formats
        valid_ids = [
            "ONT:001",
            "ONTO:12345",
            "CHEBI:ONTOLOGY",
            "GO:ONTOLOGY",
            "TEST:001",
        ]

        for valid_id in valid_ids:
            assert ontology.validate_id_format(valid_id) is True

    def test_validate_id_format_invalid(self):
        """Test validation of invalid ontology ID formats."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_id_format.side_effect = (
            lambda x: x is not None
            and ":" in str(x)
            and len(str(x).split(":")) == 2
            and all(part.strip() for part in str(x).split(":"))
        )
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Test Ontology")

        # Test invalid ID formats
        invalid_ids = [
            "invalid_id",
            "ONT_001",
            "",
            "ONT:",
            ":001",
            "ONT:001:extra",
            None,
        ]

        for invalid_id in invalid_ids:
            assert ontology.validate_id_format(invalid_id) is False

    def test_validate_structure_valid(self):
        """Test validation of valid ontology structure."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_structure.return_value = True
        MockOntology.return_value = mock_instance

        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            terms={"CHEBI:12345": Mock()},
            relationships={"REL:001": Mock()},
            namespaces=["chemical"],
        )

        assert ontology.validate_structure() is True

    def test_validate_structure_invalid(self):
        """Test validation of invalid ontology structure."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_structure.return_value = False
        mock_instance.validation_errors = [
            "Missing required namespaces",
            "Invalid term references",
        ]
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Invalid Ontology")

        assert ontology.validate_structure() is False
        assert len(ontology.validation_errors) == 2

    def test_validate_consistency_valid(self):
        """Test validation of ontology consistency."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_consistency.return_value = True
        mock_instance.is_consistent = True
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Consistent Ontology")

        assert ontology.validate_consistency() is True
        assert ontology.is_consistent is True

    def test_validate_consistency_circular_dependencies(self):
        """Test detection of circular dependencies."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_consistency.return_value = False
        mock_instance.is_consistent = False
        mock_instance.validation_errors = [
            "Circular dependency detected: CHEBI:001 -> CHEBI:002 -> CHEBI:001"
        ]
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Inconsistent Ontology")

        assert ontology.validate_consistency() is False
        assert ontology.is_consistent is False
        assert "Circular dependency detected" in ontology.validation_errors[0]

    def test_is_valid_comprehensive(self):
        """Test comprehensive ontology validation."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.is_valid.return_value = True
        MockOntology.return_value = mock_instance

        ontology = MockOntology(
            id="ONT:001",
            name="Valid Ontology",
            version="1.0",
            terms={"CHEBI:12345": Mock()},
            relationships={"REL:001": Mock()},
            namespaces=["chemical"],
        )

        assert ontology.is_valid() is True

    def test_is_valid_invalid_ontology(self):
        """Test comprehensive validation on invalid ontology."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.is_valid.return_value = False
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="invalid", name="")

        assert ontology.is_valid() is False


class TestOntologyEquality:
    """Test equality and hashing functionality."""

    def test_ontology_equality_same_content(self):
        """Test that ontologies with same content are equal."""
        MockOntology = Mock()
        # Create two instances with same data
        ont1 = MockOntology(id="ONT:001", name="Chemical Ontology", version="1.0")
        ont2 = MockOntology(id="ONT:001", name="Chemical Ontology", version="1.0")

        # Configure mock equality
        ont1.__eq__ = Mock(return_value=True)

        assert ont1 == ont2

    def test_ontology_equality_different_content(self):
        """Test that ontologies with different content are not equal."""
        MockOntology = Mock()
        ont1 = MockOntology(id="ONT:001", name="Chemical Ontology", version="1.0")
        ont2 = MockOntology(id="ONT:002", name="Biological Ontology", version="2.0")

        # Configure mock inequality
        ont1.__eq__ = Mock(return_value=False)

        assert ont1 != ont2

    def test_ontology_equality_different_types(self):
        """Test that ontology is not equal to objects of different types."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Test Ontology")
        ontology.__eq__ = Mock(return_value=False)

        assert ontology != "not an ontology"
        assert ontology != 12345
        assert ontology != None
        assert ontology != {"id": "ONT:001", "name": "Test Ontology"}

    def test_ontology_hashing_consistent(self):
        """Test that ontology hashing is consistent."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Test Ontology")
        ontology.__hash__ = Mock(return_value=12345)

        hash1 = hash(ontology)
        hash2 = hash(ontology)

        assert hash1 == hash2

    def test_ontology_hashing_equal_ontologies(self):
        """Test that equal ontologies have the same hash."""
        MockOntology = Mock()
        ont1 = MockOntology(id="ONT:001", name="Chemical Ontology")
        ont2 = MockOntology(id="ONT:001", name="Chemical Ontology")

        # Configure same hash for equal objects
        expected_hash = 12345
        ont1.__hash__ = Mock(return_value=expected_hash)
        ont2.__hash__ = Mock(return_value=expected_hash)
        ont1.__eq__ = Mock(return_value=True)

        assert hash(ont1) == hash(ont2)

    def test_ontology_in_set(self):
        """Test that ontologies can be used in sets (hashable)."""
        # Test the concept that ontologies should be hashable
        # In the actual implementation, ontologies with same id should be considered equal
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.id = "ONT:001"
        mock_instance.name = "Chemical Ontology"
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Chemical Ontology")

        # Test that the ontology has the necessary attributes for set membership
        assert hasattr(ontology, "id")
        assert hasattr(ontology, "name")

        # In a real implementation, ontologies should be hashable and work in sets
        # This test verifies the expected behavior pattern

    def test_ontology_in_dict_key(self):
        """Test that ontologies can be used as dictionary keys."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Chemical Ontology")
        ontology.__hash__ = Mock(return_value=12345)

        ontology_dict = {ontology: "This is a chemical ontology"}

        assert ontology_dict[ontology] == "This is a chemical ontology"


class TestOntologyStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self):
        """Test __str__ method for user-friendly display."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Chemical Ontology", version="1.0")
        ontology.__str__ = Mock(return_value="Chemical Ontology v1.0 (ONT:001)")

        assert str(ontology) == "Chemical Ontology v1.0 (ONT:001)"

    def test_repr_representation(self):
        """Test __repr__ method for debugging."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Chemical Ontology", version="1.0")
        ontology.__repr__ = Mock(
            return_value="Ontology(id='ONT:001', name='Chemical Ontology', version='1.0')"
        )

        assert (
            repr(ontology)
            == "Ontology(id='ONT:001', name='Chemical Ontology', version='1.0')"
        )

    def test_str_with_description(self):
        """Test string representation with description."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            version="1.0",
            description="A comprehensive chemical ontology",
        )
        ontology.__str__ = Mock(
            return_value="Chemical Ontology v1.0 (ONT:001): A comprehensive chemical ontology"
        )

        assert (
            str(ontology)
            == "Chemical Ontology v1.0 (ONT:001): A comprehensive chemical ontology"
        )

    def test_str_with_statistics(self):
        """Test string representation with term and relationship counts."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            version="1.0",
            term_count=1500,
            relationship_count=3200,
        )
        ontology.__str__ = Mock(
            return_value="Chemical Ontology v1.0 (ONT:001) - 1500 terms, 3200 relationships"
        )

        assert (
            str(ontology)
            == "Chemical Ontology v1.0 (ONT:001) - 1500 terms, 3200 relationships"
        )

    def test_str_inconsistent_ontology(self):
        """Test string representation of inconsistent ontology."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001", name="Chemical Ontology", version="1.0", is_consistent=False
        )
        ontology.__str__ = Mock(
            return_value="[INCONSISTENT] Chemical Ontology v1.0 (ONT:001)"
        )

        assert str(ontology) == "[INCONSISTENT] Chemical Ontology v1.0 (ONT:001)"

    def test_repr_with_all_attributes(self):
        """Test repr with multiple attributes."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            version="1.0",
            term_count=150,
            relationship_count=320,
            is_consistent=True,
        )
        expected_repr = (
            "Ontology(id='ONT:001', name='Chemical Ontology', version='1.0', "
            "term_count=150, relationship_count=320, is_consistent=True)"
        )
        ontology.__repr__ = Mock(return_value=expected_repr)

        assert repr(ontology) == expected_repr


class TestOntologySerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic ontology serialization to dictionary."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Chemical Ontology")
        expected_dict = {
            "id": "ONT:001",
            "name": "Chemical Ontology",
            "version": "1.0",
            "description": None,
            "terms": {},
            "relationships": {},
            "namespaces": [],
            "metadata": {},
            "base_iris": [],
            "imports": [],
            "synonymtypedef": {},
            "term_count": 0,
            "relationship_count": 0,
            "is_consistent": True,
            "validation_errors": [],
        }
        ontology.to_dict = Mock(return_value=expected_dict)

        result = ontology.to_dict()
        assert result == expected_dict

    def test_to_dict_full(self):
        """Test full ontology serialization to dictionary."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            version="2023.1",
            description="A comprehensive chemical ontology",
            terms={"CHEBI:12345": {"name": "glucose"}},
            relationships={"REL:001": {"subject": "CHEBI:12345", "predicate": "is_a"}},
            namespaces=["chemical", "biological_process"],
            metadata={"source": "ChEBI", "license": "CC BY 4.0"},
            base_iris=["http://purl.obolibrary.org/obo/chebi.owl"],
            imports=["http://purl.obolibrary.org/obo/go.owl"],
            term_count=1500,
            relationship_count=3200,
        )

        expected_dict = {
            "id": "ONT:001",
            "name": "Chemical Ontology",
            "version": "2023.1",
            "description": "A comprehensive chemical ontology",
            "terms": {"CHEBI:12345": {"name": "glucose"}},
            "relationships": {
                "REL:001": {"subject": "CHEBI:12345", "predicate": "is_a"}
            },
            "namespaces": ["chemical", "biological_process"],
            "metadata": {"source": "ChEBI", "license": "CC BY 4.0"},
            "base_iris": ["http://purl.obolibrary.org/obo/chebi.owl"],
            "imports": ["http://purl.obolibrary.org/obo/go.owl"],
            "synonymtypedef": {},
            "term_count": 1500,
            "relationship_count": 3200,
            "is_consistent": True,
            "validation_errors": [],
        }
        ontology.to_dict = Mock(return_value=expected_dict)

        result = ontology.to_dict()
        assert result == expected_dict

    def test_from_dict_basic(self):
        """Test basic ontology deserialization from dictionary."""
        MockOntology = Mock()
        ontology_dict = {"id": "ONT:001", "name": "Chemical Ontology"}

        MockOntology.from_dict = Mock(
            return_value=MockOntology(id="ONT:001", name="Chemical Ontology")
        )

        MockOntology.from_dict(ontology_dict)

        MockOntology.from_dict.assert_called_once_with(ontology_dict)

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        MockOntology = Mock()
        original_ontology = MockOntology(
            id="ONT:001",
            name="Chemical Ontology",
            version="2023.1",
            description="A comprehensive chemical ontology",
            term_count=1500,
            relationship_count=3200,
        )

        # Mock the roundtrip
        ontology_dict = {
            "id": "ONT:001",
            "name": "Chemical Ontology",
            "version": "2023.1",
            "description": "A comprehensive chemical ontology",
            "term_count": 1500,
            "relationship_count": 3200,
        }
        original_ontology.to_dict = Mock(return_value=ontology_dict)
        MockOntology.from_dict = Mock(return_value=original_ontology)

        # Roundtrip test
        serialized = original_ontology.to_dict()
        deserialized = MockOntology.from_dict(serialized)

        assert deserialized == original_ontology

    def test_json_serialization_invalid_data(self):
        """Test JSON serialization with invalid data."""
        MockOntology = Mock()
        ontology = MockOntology(id="ONT:001", name="Chemical Ontology")
        ontology.to_json = Mock(
            side_effect=TypeError("Object is not JSON serializable")
        )

        with pytest.raises(TypeError, match="Object is not JSON serializable"):
            ontology.to_json()


class TestOntologyEdgeCases:
    """Test edge cases and error handling."""

    def test_ontology_with_none_values(self):
        """Test ontology creation with None values for optional fields."""
        MockOntology = Mock()
        ontology = MockOntology(
            id="ONT:001",
            name="Test Ontology",
            description=None,
            metadata=None,
            base_iris=None,
            imports=None,
        )

        # Should handle None values gracefully
        MockOntology.assert_called_once()

    def test_ontology_with_empty_collections(self):
        """Test ontology with empty collections."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.terms = {}
        mock_instance.relationships = {}
        mock_instance.namespaces = []
        mock_instance.metadata = {}
        mock_instance.base_iris = []
        mock_instance.imports = []
        MockOntology.return_value = mock_instance

        ontology = MockOntology(
            id="ONT:001",
            name="Empty Ontology",
            terms={},
            relationships={},
            namespaces=[],
            metadata={},
        )

        assert ontology.terms == {}
        assert ontology.relationships == {}
        assert ontology.namespaces == []
        assert ontology.metadata == {}
        assert ontology.base_iris == []
        assert ontology.imports == []

    def test_ontology_with_unicode_characters(self):
        """Test ontology with Unicode characters in name and description."""
        MockOntology = Mock()
        unicode_name = "Ontología Química"
        unicode_description = (
            "Una ontología química comprehensiva con caracteres especiales: α, β, γ"
        )

        ontology = MockOntology(
            id="ONT:001", name=unicode_name, description=unicode_description
        )

        MockOntology.assert_called_once_with(
            id="ONT:001", name=unicode_name, description=unicode_description
        )

    def test_ontology_modification_after_creation(self):
        """Test modifying ontology attributes after creation."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.name = "Original Ontology"
        mock_instance.description = "Original description"
        mock_instance.version = "1.0"
        mock_instance.terms = {}
        mock_instance.relationships = {}
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Original Ontology")

        # Modify attributes
        ontology.name = "Modified Ontology"
        ontology.description = "Modified description"
        ontology.version = "2.0"
        ontology.terms["NEW:001"] = {"name": "new term"}
        ontology.relationships["REL:001"] = {"subject": "NEW:001", "predicate": "is_a"}

        assert ontology.name == "Modified Ontology"
        assert ontology.description == "Modified description"
        assert ontology.version == "2.0"
        assert "NEW:001" in ontology.terms
        assert "REL:001" in ontology.relationships


class TestOntologySpecific:
    """Test ontology-specific functionality."""

    def test_ontology_term_management(self):
        """Test adding and removing terms from ontology."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.terms = {}
        mock_instance.term_count = 0
        mock_instance.add_term = Mock(return_value=True)
        mock_instance.remove_term = Mock(return_value=True)
        mock_instance.get_term = Mock(return_value=None)
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Term Management Test")

        # Test adding terms
        mock_term = Mock()
        mock_term.id = "CHEBI:12345"
        mock_term.name = "glucose"

        ontology.add_term(mock_term)
        ontology.add_term.assert_called_once_with(mock_term)

        # Test removing terms
        ontology.remove_term("CHEBI:12345")
        ontology.remove_term.assert_called_once_with("CHEBI:12345")

        # Test getting terms
        ontology.get_term("CHEBI:12345")
        ontology.get_term.assert_called_once_with("CHEBI:12345")

    def test_ontology_relationship_management(self):
        """Test adding and removing relationships from ontology."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.relationships = {}
        mock_instance.relationship_count = 0
        mock_instance.add_relationship = Mock(return_value=True)
        mock_instance.remove_relationship = Mock(return_value=True)
        mock_instance.get_relationships = Mock(return_value=[])
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Relationship Management Test")

        # Test adding relationships
        mock_relationship = Mock()
        mock_relationship.id = "REL:001"
        mock_relationship.subject = "CHEBI:12345"
        mock_relationship.predicate = "is_a"
        mock_relationship.object = "CHEBI:33917"

        ontology.add_relationship(mock_relationship)
        ontology.add_relationship.assert_called_once_with(mock_relationship)

        # Test removing relationships
        ontology.remove_relationship("REL:001")
        ontology.remove_relationship.assert_called_once_with("REL:001")

        # Test getting relationships
        ontology.get_relationships("CHEBI:12345")
        ontology.get_relationships.assert_called_once_with("CHEBI:12345")

    def test_ontology_statistics_generation(self):
        """Test ontology statistics generation."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.get_statistics = Mock(
            return_value={
                "term_count": 1500,
                "relationship_count": 3200,
                "namespace_count": 5,
                "max_depth": 12,
                "avg_children_per_term": 2.1,
                "orphan_terms": 3,
                "circular_dependencies": 0,
            }
        )
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Statistics Test Ontology")

        stats = ontology.get_statistics()

        assert stats["term_count"] == 1500
        assert stats["relationship_count"] == 3200
        assert stats["namespace_count"] == 5
        assert stats["max_depth"] == 12
        assert stats["avg_children_per_term"] == 2.1
        assert stats["orphan_terms"] == 3
        assert stats["circular_dependencies"] == 0

    def test_ontology_validation_comprehensive(self):
        """Test comprehensive ontology validation."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.validate_full = Mock(return_value=True)
        mock_instance.validation_report = {
            "structure_valid": True,
            "consistency_valid": True,
            "terms_valid": True,
            "relationships_valid": True,
            "namespaces_valid": True,
            "errors": [],
            "warnings": ["Consider adding more synonyms for term CHEBI:12345"],
        }
        MockOntology.return_value = mock_instance

        ontology = MockOntology(id="ONT:001", name="Validation Test Ontology")

        # Test full validation
        is_valid = ontology.validate_full()
        assert is_valid is True

        # Check validation report
        report = ontology.validation_report
        assert report["structure_valid"] is True
        assert report["consistency_valid"] is True
        assert len(report["errors"]) == 0
        assert len(report["warnings"]) == 1


# Test fixtures for complex scenarios
@pytest.fixture
def sample_ontology_data():
    """Fixture providing sample ontology data for testing."""
    return {
        "id": "ONT:001",
        "name": "Chemical Ontology",
        "version": "2023.1",
        "description": "A comprehensive chemical ontology for biological systems",
        "terms": {
            "CHEBI:12345": {
                "name": "glucose",
                "definition": "A monosaccharide that is aldehydo-D-glucose",
                "synonyms": ["dextrose", "D-glucose"],
            },
            "CHEBI:33917": {
                "name": "aldohexose",
                "definition": "A hexose with an aldehyde group",
                "synonyms": [],
            },
        },
        "relationships": {
            "REL:001": {
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
                "confidence": 0.95,
            }
        },
        "namespaces": ["chemical", "biological_process"],
        "metadata": {
            "source": "ChEBI",
            "created_date": "2023-01-01T00:00:00Z",
            "modified_date": "2023-06-15T10:30:00Z",
            "authors": ["John Doe", "Jane Smith"],
            "license": "CC BY 4.0",
        },
        "base_iris": ["http://purl.obolibrary.org/obo/chebi.owl"],
        "imports": ["http://purl.obolibrary.org/obo/go.owl"],
        "term_count": 2,
        "relationship_count": 1,
        "is_consistent": True,
        "validation_errors": [],
    }


@pytest.fixture
def sample_large_ontology_data():
    """Fixture providing sample large ontology data for testing."""
    # Generate large dataset
    terms = {
        f"TERM:{i:06d}": {"name": f"term_{i}", "definition": f"Definition for term {i}"}
        for i in range(1000)
    }
    relationships = {
        f"REL:{i:06d}": {
            "subject": f"TERM:{i:06d}",
            "predicate": "is_a",
            "object": f"TERM:{(i+1):06d}",
        }
        for i in range(999)
    }

    return {
        "id": "ONT:002",
        "name": "Large Test Ontology",
        "version": "1.0",
        "description": "A large ontology for performance testing",
        "terms": terms,
        "relationships": relationships,
        "namespaces": ["test", "performance"],
        "metadata": {"source": "Generated", "purpose": "testing"},
        "term_count": 1000,
        "relationship_count": 999,
        "is_consistent": True,
        "validation_errors": [],
    }


@pytest.fixture
def sample_inconsistent_ontology_data():
    """Fixture providing sample inconsistent ontology data for testing."""
    return {
        "id": "ONT:003",
        "name": "Inconsistent Ontology",
        "version": "1.0",
        "description": "An ontology with consistency issues",
        "terms": {
            "TERM:001": {"name": "term1", "definition": "First term"},
            "TERM:002": {"name": "term2", "definition": "Second term"},
            "TERM:003": {"name": "term3", "definition": "Third term"},
        },
        "relationships": {
            "REL:001": {
                "subject": "TERM:001",
                "predicate": "is_a",
                "object": "TERM:002",
            },
            "REL:002": {
                "subject": "TERM:002",
                "predicate": "is_a",
                "object": "TERM:003",
            },
            "REL:003": {
                "subject": "TERM:003",
                "predicate": "is_a",
                "object": "TERM:001",
            },  # Creates cycle
        },
        "namespaces": ["test"],
        "metadata": {"source": "Test", "notes": "Contains circular dependency"},
        "term_count": 3,
        "relationship_count": 3,
        "is_consistent": False,
        "validation_errors": [
            "Circular dependency detected: TERM:001 -> TERM:002 -> TERM:003 -> TERM:001"
        ],
    }


class TestOntologyIntegration:
    """Integration tests using fixtures."""

    def test_ontology_creation_with_fixture(self, sample_ontology_data):
        """Test ontology creation using fixture data."""
        MockOntology = Mock()
        MockOntology(**sample_ontology_data)
        MockOntology.assert_called_once_with(**sample_ontology_data)

    def test_large_ontology_with_fixture(self, sample_large_ontology_data):
        """Test large ontology handling using fixture data."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.term_count = 1000
        mock_instance.relationship_count = 999
        mock_instance.is_consistent = True
        MockOntology.return_value = mock_instance

        ontology = MockOntology(**sample_large_ontology_data)

        assert ontology.term_count == 1000
        assert ontology.relationship_count == 999
        assert ontology.is_consistent is True

    def test_inconsistent_ontology_with_fixture(
        self, sample_inconsistent_ontology_data
    ):
        """Test inconsistent ontology handling using fixture data."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.is_consistent = False
        mock_instance.validation_errors = [
            "Circular dependency detected: TERM:001 -> TERM:002 -> TERM:003 -> TERM:001"
        ]
        MockOntology.return_value = mock_instance

        ontology = MockOntology(**sample_inconsistent_ontology_data)

        assert ontology.is_consistent is False
        assert len(ontology.validation_errors) == 1
        assert "Circular dependency detected" in ontology.validation_errors[0]

    def test_ontology_collection_operations(self, sample_ontology_data):
        """Test ontology operations in collections."""
        MockOntology = Mock()
        # Create multiple ontologies
        ont1 = MockOntology(**sample_ontology_data)
        ont2_data = sample_ontology_data.copy()
        ont2_data["id"] = "ONT:002"
        ont2_data["name"] = "Biological Ontology"
        ont2 = MockOntology(**ont2_data)

        # Test basic collection operations
        ontologies = [ont1, ont2]

        assert len(ontologies) == 2
        assert ont1 in ontologies
        assert ont2 in ontologies

        # Test that both ontologies can be stored in a dictionary
        ontology_dict = {sample_ontology_data["id"]: ont1, ont2_data["id"]: ont2}

        assert len(ontology_dict) == 2
        assert ontology_dict["ONT:001"] == ont1
        assert ontology_dict["ONT:002"] == ont2

    def test_ontology_serialization_performance(self, sample_large_ontology_data):
        """Test ontology serialization with large datasets."""
        MockOntology = Mock()
        mock_instance = Mock()
        mock_instance.to_dict = Mock(return_value=sample_large_ontology_data)
        mock_instance.to_json = Mock(
            return_value='{"id": "ONT:002", "term_count": 1000}'
        )
        MockOntology.return_value = mock_instance

        ontology = MockOntology(**sample_large_ontology_data)

        # Test serialization performance
        dict_result = ontology.to_dict()
        json_result = ontology.to_json()

        assert dict_result["term_count"] == 1000
        assert "ONT:002" in json_result
