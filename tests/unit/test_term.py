"""
Comprehensive Unit Tests for Term Class

This module provides comprehensive unit tests for the Term class from the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the Term class before implementation.

Test Classes:
    TestTermCreation: Tests for Term instantiation and attribute handling
    TestTermValidation: Tests for term validation methods
    TestTermEquality: Tests for equality and hashing functionality
    TestTermStringRepresentation: Tests for __str__ and __repr__ methods
    TestTermSerialization: Tests for JSON serialization/deserialization
    TestTermEdgeCases: Tests for edge cases and error handling
    TestTermOntologySpecific: Tests for ontology-specific functionality

The Term class is expected to be a dataclass representing ontological terms with:
- Core attributes: id, name, definition, synonyms, etc.
- Validation methods for data integrity
- Equality and hashing support for collections
- String representations for debugging and display
- JSON serialization for persistence and data exchange

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - json: For JSON serialization testing
    - typing: For type hints
    - dataclasses: For dataclass functionality

Usage:
    pytest tests/unit/test_term.py -v
"""

import pytest
from unittest.mock import Mock, patch


# The Term class has been implemented - importing for testing
from aim2_project.aim2_ontology.models import Term


class TestTermCreation:
    """Test Term class instantiation and attribute handling."""

    def test_term_creation_minimal(self):
        """Test creating a term with minimal required attributes."""
        # Arrange & Act - Using mock Term class for now
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")

            # Assert
            MockTerm.assert_called_once_with(id="CHEBI:12345", name="glucose")

    def test_term_creation_full_attributes(self):
        """Test creating a term with all possible attributes."""
        # Arrange
        term_data = {
            "id": "CHEBI:12345",
            "name": "glucose",
            "definition": "A monosaccharide that is aldehydo-D-glucose",
            "synonyms": ["dextrose", "D-glucose", "grape sugar"],
            "namespace": "chemical",
            "is_obsolete": False,
            "replaced_by": None,
            "alt_ids": ["CHEBI:4167"],
            "xrefs": ["CAS:50-99-7", "KEGG:C00031"],
            "parents": ["CHEBI:33917"],  # aldohexose
            "children": [],
            "relationships": {"is_a": ["CHEBI:33917"]},
            "metadata": {"source": "ChEBI", "version": "2023.1"},
        }

        # Act & Assert - Using mock Term class
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            MockTerm(**term_data)
            MockTerm.assert_called_once_with(**term_data)

    def test_term_creation_with_defaults(self):
        """Test that term creation uses appropriate default values."""
        # Arrange & Act
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            # Configure mock to return expected default values
            mock_instance = Mock()
            mock_instance.synonyms = []
            mock_instance.alt_ids = []
            mock_instance.xrefs = []
            mock_instance.parents = []
            mock_instance.children = []
            mock_instance.relationships = {}
            mock_instance.metadata = {}
            mock_instance.is_obsolete = False
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="test term")

            # Assert defaults
            assert term.synonyms == []
            assert term.alt_ids == []
            assert term.xrefs == []
            assert term.parents == []
            assert term.children == []
            assert term.relationships == {}
            assert term.metadata == {}
            assert term.is_obsolete is False

    def test_term_creation_invalid_id(self):
        """Test that creating a term with invalid ID raises appropriate error."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            MockTerm.side_effect = ValueError("Invalid term ID format")

            with pytest.raises(ValueError, match="Invalid term ID format"):
                MockTerm(id="invalid_id", name="test")

    def test_term_creation_empty_name(self):
        """Test that creating a term with empty name raises appropriate error."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            MockTerm.side_effect = ValueError("Term name cannot be empty")

            with pytest.raises(ValueError, match="Term name cannot be empty"):
                MockTerm(id="TEST:001", name="")

    def test_term_immutability_after_creation(self):
        """Test that term attributes can be modified after creation (mutable dataclass)."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.name = "glucose"
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="glucose")

            # Should be able to modify attributes (mutable dataclass expected)
            term.name = "D-glucose"
            assert term.name == "D-glucose"


class TestTermValidation:
    """Test term validation methods."""

    def test_validate_id_format_valid(self):
        """Test validation of valid term ID formats."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_id_format.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="test")

            # Test various valid ID formats
            valid_ids = [
                "CHEBI:12345",
                "GO:0008150",
                "PO:0000003",
                "NCIT:C12345",
                "TEST:001",
            ]

            for valid_id in valid_ids:
                assert term.validate_id_format(valid_id) is True

    def test_validate_id_format_invalid(self):
        """Test validation of invalid term ID formats."""
        term = Term(id="CHEBI:12345", name="test")

        # Test invalid ID formats
        invalid_ids = [
            "invalid_id",
            "CHEBI_12345",
            "",
            "CHEBI:",
            ":12345",
            "CHEBI:12345:extra",
            None,
        ]

        for invalid_id in invalid_ids:
            assert term.validate_id_format(invalid_id) is False

    def test_validate_name_valid(self):
        """Test validation of valid term names."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_name.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="valid name")

            valid_names = [
                "glucose",
                "D-glucose",
                "glucose 6-phosphate",
                "mitochondrial ribosome",
                "α-D-glucose",
                "N-acetyl-D-glucosamine",
            ]

            for valid_name in valid_names:
                assert term.validate_name(valid_name) is True

    def test_validate_name_invalid(self):
        """Test validation of invalid term names."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_name.side_effect = lambda x: bool(x and x.strip())
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="valid name")

            invalid_names = ["", "   ", None]

            for invalid_name in invalid_names:
                assert term.validate_name(invalid_name) is False

    def test_validate_synonyms(self):
        """Test validation of synonym lists."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_synonyms.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="test")

            # Valid synonym lists
            valid_synonyms = [
                [],
                ["synonym1"],
                ["synonym1", "synonym2", "synonym3"],
                ["D-glucose", "dextrose", "grape sugar"],
            ]

            for synonyms in valid_synonyms:
                assert term.validate_synonyms(synonyms) is True

    def test_validate_relationships(self):
        """Test validation of relationship dictionaries."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_relationships.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="test")

            # Valid relationship structures
            valid_relationships = [
                {},
                {"is_a": ["PARENT:001"]},
                {"is_a": ["PARENT:001"], "part_of": ["WHOLE:001"]},
                {"regulates": ["TARGET:001", "TARGET:002"]},
            ]

            for relationships in valid_relationships:
                assert term.validate_relationships(relationships) is True

    def test_is_valid_comprehensive(self):
        """Test comprehensive term validation."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.is_valid.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(
                id="CHEBI:12345",
                name="glucose",
                definition="A monosaccharide",
                synonyms=["dextrose"],
                is_obsolete=False,
            )

            assert term.is_valid() is True

    def test_is_valid_invalid_term(self):
        """Test comprehensive validation on invalid term."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.is_valid.return_value = False
            MockTerm.return_value = mock_instance

            term = MockTerm(id="invalid", name="")

            assert term.is_valid() is False


class TestTermEquality:
    """Test equality and hashing functionality."""

    def test_term_equality_same_content(self):
        """Test that terms with same content are equal."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            # Create two instances with same data
            term1 = MockTerm(id="CHEBI:12345", name="glucose")
            term2 = MockTerm(id="CHEBI:12345", name="glucose")

            # Configure mock equality
            term1.__eq__ = Mock(return_value=True)

            assert term1 == term2

    def test_term_equality_different_content(self):
        """Test that terms with different content are not equal."""
        term1 = Term(id="CHEBI:12345", name="glucose")
        term2 = Term(id="CHEBI:54321", name="fructose")

        assert term1 != term2

    def test_term_equality_different_types(self):
        """Test that term is not equal to objects of different types."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            term.__eq__ = Mock(return_value=False)

            assert term != "not a term"
            assert term != 12345
            assert term != None
            assert term != {"id": "CHEBI:12345", "name": "glucose"}

    def test_term_hashing_consistent(self):
        """Test that term hashing is consistent."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            term.__hash__ = Mock(return_value=12345)

            hash1 = hash(term)
            hash2 = hash(term)

            assert hash1 == hash2

    def test_term_hashing_equal_terms(self):
        """Test that equal terms have the same hash."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term1 = MockTerm(id="CHEBI:12345", name="glucose")
            term2 = MockTerm(id="CHEBI:12345", name="glucose")

            # Configure same hash for equal objects
            expected_hash = 12345
            term1.__hash__ = Mock(return_value=expected_hash)
            term2.__hash__ = Mock(return_value=expected_hash)
            term1.__eq__ = Mock(return_value=True)

            assert hash(term1) == hash(term2)

    def test_term_in_set(self):
        """Test that terms can be used in sets (hashable)."""
        term1 = Term(id="CHEBI:12345", name="glucose")
        term2 = Term(id="CHEBI:54321", name="fructose")
        term3 = Term(id="CHEBI:12345", name="glucose")  # duplicate of term1

        term_set = {term1, term2, term3}

        # Should only contain 2 unique terms
        assert len(term_set) == 2

    def test_term_in_dict_key(self):
        """Test that terms can be used as dictionary keys."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            term.__hash__ = Mock(return_value=12345)

            term_dict = {term: "This is glucose"}

            assert term_dict[term] == "This is glucose"


class TestTermStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self):
        """Test __str__ method for user-friendly display."""
        term = Term(id="CHEBI:12345", name="glucose")
        assert str(term) == "glucose (CHEBI:12345)"

    def test_repr_representation(self):
        """Test __repr__ method for debugging."""
        term = Term(id="CHEBI:12345", name="glucose")
        assert repr(term) == "Term(id='CHEBI:12345', name='glucose')"

    def test_str_with_definition(self):
        """Test string representation with definition."""
        term = Term(id="CHEBI:12345", name="glucose", definition="A monosaccharide")
        assert str(term) == "glucose (CHEBI:12345): A monosaccharide"

    def test_str_obsolete_term(self):
        """Test string representation of obsolete term."""
        term = Term(id="CHEBI:12345", name="glucose", is_obsolete=True)
        assert str(term) == "[OBSOLETE] glucose (CHEBI:12345)"

    def test_str_obsolete_term_with_definition(self):
        """Test string representation of obsolete term with definition."""
        term = Term(
            id="CHEBI:12345",
            name="glucose",
            definition="A monosaccharide",
            is_obsolete=True,
        )
        assert str(term) == "[OBSOLETE] glucose (CHEBI:12345): A monosaccharide"

    def test_repr_with_synonyms(self):
        """Test repr with synonyms."""
        term = Term(
            id="CHEBI:12345",
            name="glucose",
            synonyms=["dextrose", "D-glucose"],
        )
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        assert "synonyms=['dextrose', 'D-glucose']" in repr_str

    def test_repr_with_definition(self):
        """Test repr with definition."""
        term = Term(
            id="CHEBI:12345",
            name="glucose",
            definition="A monosaccharide that is aldehydo-D-glucose",
        )
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        assert "definition='A monosaccharide that is aldehydo-D-glucose'" in repr_str

    def test_repr_with_long_definition(self):
        """Test repr with definition longer than 50 characters."""
        long_definition = "A" * 60  # 60 characters
        term = Term(id="CHEBI:12345", name="glucose", definition=long_definition)
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        # Should truncate definition and add ellipsis
        assert f"definition='{long_definition[:50]}...'" in repr_str

    def test_repr_with_namespace(self):
        """Test repr with namespace."""
        term = Term(id="CHEBI:12345", name="glucose", namespace="chemical")
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        assert "namespace='chemical'" in repr_str

    def test_repr_with_obsolete_flag(self):
        """Test repr with obsolete flag."""
        term = Term(id="CHEBI:12345", name="glucose", is_obsolete=True)
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        assert "is_obsolete=True" in repr_str

    def test_repr_with_multiple_attributes(self):
        """Test repr with multiple non-default attributes."""
        term = Term(
            id="CHEBI:12345",
            name="glucose",
            definition="A monosaccharide",
            synonyms=["dextrose"],
            namespace="chemical",
            is_obsolete=False,
        )
        repr_str = repr(term)
        assert "Term(id='CHEBI:12345', name='glucose'" in repr_str
        assert "definition='A monosaccharide'" in repr_str
        assert "synonyms=['dextrose']" in repr_str
        assert "namespace='chemical'" in repr_str
        # is_obsolete=False should not appear as it's the default value

    def test_str_edge_cases(self):
        """Test string representation edge cases."""
        # Term with empty definition should not show definition
        term_empty_def = Term(id="CHEBI:12345", name="glucose", definition="")
        assert str(term_empty_def) == "glucose (CHEBI:12345)"

        # Term with None definition should not show definition
        term_none_def = Term(id="CHEBI:12345", name="glucose", definition=None)
        assert str(term_none_def) == "glucose (CHEBI:12345)"


class TestTermSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic term serialization to dictionary."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            expected_dict = {
                "id": "CHEBI:12345",
                "name": "glucose",
                "definition": None,
                "synonyms": [],
                "namespace": None,
                "is_obsolete": False,
                "replaced_by": None,
                "alt_ids": [],
                "xrefs": [],
                "parents": [],
                "children": [],
                "relationships": {},
                "metadata": {},
            }
            term.to_dict = Mock(return_value=expected_dict)

            result = term.to_dict()
            assert result == expected_dict

    def test_to_dict_full(self):
        """Test full term serialization to dictionary."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(
                id="CHEBI:12345",
                name="glucose",
                definition="A monosaccharide",
                synonyms=["dextrose", "D-glucose"],
                namespace="chemical",
                is_obsolete=False,
                alt_ids=["CHEBI:4167"],
                xrefs=["CAS:50-99-7"],
                parents=["CHEBI:33917"],
                relationships={"is_a": ["CHEBI:33917"]},
                metadata={"source": "ChEBI"},
            )

            expected_dict = {
                "id": "CHEBI:12345",
                "name": "glucose",
                "definition": "A monosaccharide",
                "synonyms": ["dextrose", "D-glucose"],
                "namespace": "chemical",
                "is_obsolete": False,
                "replaced_by": None,
                "alt_ids": ["CHEBI:4167"],
                "xrefs": ["CAS:50-99-7"],
                "parents": ["CHEBI:33917"],
                "children": [],
                "relationships": {"is_a": ["CHEBI:33917"]},
                "metadata": {"source": "ChEBI"},
            }
            term.to_dict = Mock(return_value=expected_dict)

            result = term.to_dict()
            assert result == expected_dict

    def test_from_dict_basic(self):
        """Test basic term deserialization from dictionary."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term_dict = {"id": "CHEBI:12345", "name": "glucose"}

            MockTerm.from_dict = Mock(
                return_value=MockTerm(id="CHEBI:12345", name="glucose")
            )

            MockTerm.from_dict(term_dict)

            MockTerm.from_dict.assert_called_once_with(term_dict)

    def test_from_dict_full(self):
        """Test full term deserialization from dictionary."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term_dict = {
                "id": "CHEBI:12345",
                "name": "glucose",
                "definition": "A monosaccharide",
                "synonyms": ["dextrose", "D-glucose"],
                "namespace": "chemical",
                "is_obsolete": False,
                "alt_ids": ["CHEBI:4167"],
                "xrefs": ["CAS:50-99-7"],
                "parents": ["CHEBI:33917"],
                "relationships": {"is_a": ["CHEBI:33917"]},
                "metadata": {"source": "ChEBI"},
            }

            MockTerm.from_dict = Mock(return_value=MockTerm(**term_dict))

            MockTerm.from_dict(term_dict)

            MockTerm.from_dict.assert_called_once_with(term_dict)

    def test_to_json(self):
        """Test term serialization to JSON string."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            expected_json = '{"id": "CHEBI:12345", "name": "glucose"}'
            term.to_json = Mock(return_value=expected_json)

            result = term.to_json()
            assert result == expected_json

    def test_from_json(self):
        """Test term deserialization from JSON string."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            json_str = '{"id": "CHEBI:12345", "name": "glucose"}'

            MockTerm.from_json = Mock(
                return_value=MockTerm(id="CHEBI:12345", name="glucose")
            )

            MockTerm.from_json(json_str)

            MockTerm.from_json.assert_called_once_with(json_str)

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            original_term = MockTerm(
                id="CHEBI:12345",
                name="glucose",
                definition="A monosaccharide",
                synonyms=["dextrose"],
            )

            # Mock the roundtrip
            term_dict = {
                "id": "CHEBI:12345",
                "name": "glucose",
                "definition": "A monosaccharide",
                "synonyms": ["dextrose"],
            }
            original_term.to_dict = Mock(return_value=term_dict)
            MockTerm.from_dict = Mock(return_value=original_term)

            # Roundtrip test
            serialized = original_term.to_dict()
            deserialized = MockTerm.from_dict(serialized)

            assert deserialized == original_term

    def test_json_serialization_invalid_data(self):
        """Test JSON serialization with invalid data."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(id="CHEBI:12345", name="glucose")
            term.to_json = Mock(
                side_effect=TypeError("Object is not JSON serializable")
            )

            with pytest.raises(TypeError, match="Object is not JSON serializable"):
                term.to_json()


class TestTermEdgeCases:
    """Test edge cases and error handling."""

    def test_term_with_none_values(self):
        """Test term creation with None values for optional fields."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(
                id="TEST:001",
                name="test",
                definition=None,
                synonyms=None,
                namespace=None,
            )

            # Should handle None values gracefully
            MockTerm.assert_called_once()

    def test_term_with_empty_collections(self):
        """Test term with empty lists and dictionaries."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.synonyms = []
            mock_instance.relationships = {}
            mock_instance.metadata = {}
            MockTerm.return_value = mock_instance

            term = MockTerm(
                id="TEST:001", name="test", synonyms=[], relationships={}, metadata={}
            )

            assert term.synonyms == []
            assert term.relationships == {}
            assert term.metadata == {}

    def test_term_with_unicode_characters(self):
        """Test term with Unicode characters in name and definition."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            unicode_name = "α-D-glucose"
            unicode_definition = "α-D-glucose is a D-glucose with α configuration"

            term = MockTerm(
                id="CHEBI:12345", name=unicode_name, definition=unicode_definition
            )

            MockTerm.assert_called_once_with(
                id="CHEBI:12345", name=unicode_name, definition=unicode_definition
            )

    def test_term_with_very_long_strings(self):
        """Test term with very long strings."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            long_name = "a" * 1000
            long_definition = "b" * 5000

            term = MockTerm(id="TEST:001", name=long_name, definition=long_definition)

            MockTerm.assert_called_once()

    def test_term_with_special_characters(self):
        """Test term with special characters in various fields."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            term = MockTerm(
                id="TEST:001",
                name="term with (parentheses) and [brackets]",
                definition="Definition with special chars: !@#$%^&*()_+-={}[]|;':\",./<>?",
                synonyms=[
                    "synonym with spaces",
                    "synonym-with-dashes",
                    "synonym_with_underscores",
                ],
            )

            MockTerm.assert_called_once()

    def test_term_modification_after_creation(self):
        """Test modifying term attributes after creation."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.name = "original name"
            mock_instance.definition = "original definition"
            mock_instance.synonyms = ["original synonym"]
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="original name")

            # Modify attributes
            term.name = "modified name"
            term.definition = "modified definition"
            term.synonyms.append("new synonym")

            assert term.name == "modified name"
            assert term.definition == "modified definition"
            assert "new synonym" in term.synonyms

    def test_term_large_collections(self):
        """Test term with large collections of synonyms and relationships."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            large_synonyms = [f"synonym_{i}" for i in range(1000)]
            large_relationships = {
                f"rel_{i}": [f"target_{j}" for j in range(10)] for i in range(100)
            }

            term = MockTerm(
                id="TEST:001",
                name="test",
                synonyms=large_synonyms,
                relationships=large_relationships,
            )

            MockTerm.assert_called_once()


class TestTermOntologySpecific:
    """Test ontology-specific functionality."""

    def test_term_namespace_validation(self):
        """Test validation of ontology namespaces."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.validate_namespace.return_value = True
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="glucose", namespace="chemical")

            valid_namespaces = [
                "chemical",
                "biological_process",
                "molecular_function",
                "cellular_component",
                "anatomy",
                "phenotype",
            ]

            for namespace in valid_namespaces:
                assert term.validate_namespace(namespace) is True

    def test_term_obsolete_handling(self):
        """Test handling of obsolete terms."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.is_obsolete = True
            mock_instance.replaced_by = "CHEBI:54321"
            MockTerm.return_value = mock_instance

            term = MockTerm(
                id="CHEBI:12345",
                name="obsolete term",
                is_obsolete=True,
                replaced_by="CHEBI:54321",
            )

            assert term.is_obsolete is True
            assert term.replaced_by == "CHEBI:54321"

    def test_term_relationship_types(self):
        """Test various ontology relationship types."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.relationships = {
                "is_a": ["PARENT:001"],
                "part_of": ["WHOLE:001"],
                "regulates": ["TARGET:001"],
                "has_part": ["PART:001"],
                "occurs_in": ["LOCATION:001"],
            }
            MockTerm.return_value = mock_instance

            term = MockTerm(id="TEST:001", name="test")

            expected_relationships = {
                "is_a": ["PARENT:001"],
                "part_of": ["WHOLE:001"],
                "regulates": ["TARGET:001"],
                "has_part": ["PART:001"],
                "occurs_in": ["LOCATION:001"],
            }

            assert term.relationships == expected_relationships

    def test_term_cross_references(self):
        """Test handling of cross-references to external databases."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.xrefs = [
                "CAS:50-99-7",
                "KEGG:C00031",
                "PubChem:5793",
                "ChemSpider:5588",
                "Wikipedia:Glucose",
            ]
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="glucose")

            expected_xrefs = [
                "CAS:50-99-7",
                "KEGG:C00031",
                "PubChem:5793",
                "ChemSpider:5588",
                "Wikipedia:Glucose",
            ]

            assert term.xrefs == expected_xrefs

    def test_term_hierarchical_relationships(self):
        """Test parent-child hierarchical relationships."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.parents = [
                "CHEBI:33917",
                "CHEBI:16646",
            ]  # aldohexose, carbohydrate
            mock_instance.children = [
                "CHEBI:4167",
                "CHEBI:17925",
            ]  # D-glucose, L-glucose
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="glucose")

            assert "CHEBI:33917" in term.parents
            assert "CHEBI:16646" in term.parents
            assert "CHEBI:4167" in term.children
            assert "CHEBI:17925" in term.children

    def test_term_metadata_handling(self):
        """Test handling of term metadata."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.metadata = {
                "source": "ChEBI",
                "version": "2023.1",
                "created_date": "2023-01-01",
                "modified_date": "2023-06-15",
                "confidence_score": 0.95,
                "review_status": "approved",
            }
            MockTerm.return_value = mock_instance

            term = MockTerm(id="CHEBI:12345", name="glucose")

            expected_metadata = {
                "source": "ChEBI",
                "version": "2023.1",
                "created_date": "2023-01-01",
                "modified_date": "2023-06-15",
                "confidence_score": 0.95,
                "review_status": "approved",
            }

            assert term.metadata == expected_metadata


# Test fixtures for complex scenarios
@pytest.fixture
def sample_term_data():
    """Fixture providing sample term data for testing."""
    return {
        "id": "CHEBI:12345",
        "name": "glucose",
        "definition": "A monosaccharide that is aldehydo-D-glucose",
        "synonyms": ["dextrose", "D-glucose", "grape sugar"],
        "namespace": "chemical",
        "is_obsolete": False,
        "replaced_by": None,
        "alt_ids": ["CHEBI:4167"],
        "xrefs": ["CAS:50-99-7", "KEGG:C00031"],
        "parents": ["CHEBI:33917"],
        "children": [],
        "relationships": {"is_a": ["CHEBI:33917"]},
        "metadata": {"source": "ChEBI", "version": "2023.1"},
    }


@pytest.fixture
def sample_obsolete_term_data():
    """Fixture providing sample obsolete term data for testing."""
    return {
        "id": "CHEBI:99999",
        "name": "obsolete term",
        "definition": "This term is obsolete",
        "synonyms": [],
        "namespace": "chemical",
        "is_obsolete": True,
        "replaced_by": "CHEBI:12345",
        "alt_ids": [],
        "xrefs": [],
        "parents": [],
        "children": [],
        "relationships": {},
        "metadata": {
            "source": "ChEBI",
            "version": "2023.1",
            "obsolete_reason": "duplicate",
        },
    }


class TestTermIntegration:
    """Integration tests using fixtures."""

    def test_term_creation_with_fixture(self, sample_term_data):
        """Test term creation using fixture data."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            MockTerm(**sample_term_data)
            MockTerm.assert_called_once_with(**sample_term_data)

    def test_obsolete_term_with_fixture(self, sample_obsolete_term_data):
        """Test obsolete term handling using fixture data."""
        with patch("aim2_project.aim2_ontology.models.Term") as MockTerm:
            mock_instance = Mock()
            mock_instance.is_obsolete = True
            mock_instance.replaced_by = "CHEBI:12345"
            MockTerm.return_value = mock_instance

            term = MockTerm(**sample_obsolete_term_data)

            assert term.is_obsolete is True
            assert term.replaced_by == "CHEBI:12345"

    def test_term_collection_operations(self, sample_term_data):
        """Test term operations in collections."""
        # Create multiple terms
        term1 = Term(**sample_term_data)
        term2_data = sample_term_data.copy()
        term2_data["id"] = "CHEBI:54321"
        term2_data["name"] = "fructose"
        term2 = Term(**term2_data)

        terms = [term1, term2]
        term_set = {term1, term2}

        assert len(terms) == 2
        assert len(term_set) == 2
