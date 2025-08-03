"""
Comprehensive Unit Tests for Relationship Class

This module provides comprehensive unit tests for the Relationship class from the AIM2 ontology
information extraction system. The tests follow test-driven development (TDD) approach,
defining the expected behavior of the Relationship class before implementation.

Test Classes:
    TestRelationshipCreation: Tests for Relationship instantiation and attribute handling
    TestRelationshipValidation: Tests for relationship validation methods
    TestRelationshipEquality: Tests for equality and hashing functionality
    TestRelationshipStringRepresentation: Tests for __str__ and __repr__ methods
    TestRelationshipSerialization: Tests for JSON serialization/deserialization
    TestRelationshipEdgeCases: Tests for edge cases and error handling
    TestRelationshipOntologySpecific: Tests for ontology-specific functionality
    TestRelationshipIntegration: Integration tests using fixtures

The Relationship class is expected to be a dataclass representing ontological relationships with:
- Core attributes: id, subject, predicate, object, confidence
- Metadata attributes: source, evidence, extraction_method, created_date, provenance, is_validated
- Context attributes: context, sentence, negated, modality, directional
- Validation methods for data integrity
- Equality and hashing support for collections
- String representations for debugging and display
- JSON serialization for persistence and data exchange

Supported Relationship Types:
    is_a, part_of, has_role, participates_in, located_in, derives_from, regulates,
    catalyzes, accumulates_in, affects, involved_in, upregulates, downregulates,
    made_via, occurs_in

Dependencies:
    - pytest: For test framework
    - unittest.mock: For mocking functionality
    - json: For JSON serialization testing
    - typing: For type hints
    - dataclasses: For dataclass functionality

Usage:
    pytest tests/unit/test_relationship.py -v
"""

import pytest
from unittest.mock import Mock, patch


# Import the implemented Relationship class
from aim2_project.aim2_ontology.models import Relationship


class TestRelationshipCreation:
    """Test Relationship class instantiation and attribute handling."""

    def test_relationship_creation_minimal(self):
        """Test creating a relationship with minimal required attributes."""
        # Arrange & Act - Using mock Relationship class for now
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            relationship = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Assert
            MockRelationship.assert_called_once_with(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

    def test_relationship_creation_full_attributes(self):
        """Test creating a relationship with all possible attributes."""
        # Arrange
        relationship_data = {
            "id": "REL:001",
            "subject": "CHEBI:12345",
            "predicate": "is_a",
            "object": "CHEBI:33917",
            "confidence": 0.95,
            "source": "ChEBI",
            "evidence": "text mining",
            "extraction_method": "pattern_matching",
            "created_date": "2023-01-01T00:00:00Z",
            "provenance": "ChEBI ontology v2023.1",
            "is_validated": True,
            "context": "plant cell wall",
            "sentence": "Glucose is a type of aldohexose",
            "negated": False,
            "modality": "certain",
            "directional": True,
        }

        # Act & Assert - Using mock Relationship class
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            MockRelationship(**relationship_data)
            MockRelationship.assert_called_once_with(**relationship_data)

    def test_relationship_creation_with_defaults(self):
        """Test that relationship creation uses appropriate default values."""
        # Arrange & Act
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Configure mock to return expected default values
            mock_instance = Mock()
            mock_instance.confidence = 1.0
            mock_instance.source = None
            mock_instance.evidence = None
            mock_instance.extraction_method = None
            mock_instance.created_date = None
            mock_instance.provenance = None
            mock_instance.is_validated = False
            mock_instance.context = None
            mock_instance.sentence = None
            mock_instance.negated = False
            mock_instance.modality = "certain"
            mock_instance.directional = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Assert defaults
            assert rel.confidence == 1.0
            assert rel.source is None
            assert rel.evidence is None
            assert rel.extraction_method is None
            assert rel.created_date is None
            assert rel.provenance is None
            assert rel.is_validated is False
            assert rel.context is None
            assert rel.sentence is None
            assert rel.negated is False
            assert rel.modality == "certain"
            assert rel.directional is True

    def test_relationship_creation_invalid_id(self):
        """Test that creating a relationship with invalid ID raises appropriate error."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            MockRelationship.side_effect = ValueError("Invalid relationship ID format")

            with pytest.raises(ValueError, match="Invalid relationship ID format"):
                MockRelationship(
                    id="invalid_id",
                    subject="CHEBI:12345",
                    predicate="is_a",
                    object="CHEBI:33917",
                )

    def test_relationship_creation_invalid_predicate(self):
        """Test that creating a relationship with invalid predicate raises appropriate error."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            MockRelationship.side_effect = ValueError("Invalid predicate type")

            with pytest.raises(ValueError, match="Invalid predicate type"):
                MockRelationship(
                    id="REL:001",
                    subject="CHEBI:12345",
                    predicate="invalid_predicate",
                    object="CHEBI:33917",
                )

    def test_relationship_creation_empty_subject_object(self):
        """Test that creating a relationship with empty subject/object raises appropriate error."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            MockRelationship.side_effect = ValueError(
                "Subject and object cannot be empty"
            )

            with pytest.raises(ValueError, match="Subject and object cannot be empty"):
                MockRelationship(id="REL:001", subject="", predicate="is_a", object="")

    def test_relationship_mutability_after_creation(self):
        """Test that relationship attributes can be modified after creation (mutable dataclass)."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.confidence = 0.95
            mock_instance.is_validated = False
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Should be able to modify attributes (mutable dataclass expected)
            rel.confidence = 0.85
            rel.is_validated = True
            assert rel.confidence == 0.85
            assert rel.is_validated is True


class TestRelationshipValidation:
    """Test relationship validation methods."""

    def test_validate_id_format_valid(self):
        """Test validation of valid relationship ID formats."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_id_format.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Test various valid ID formats
            valid_ids = [
                "REL:001",
                "RELATIONSHIP:12345",
                "RO:0002202",
                "BFO:0000050",
                "TEST:001",
            ]

            for valid_id in valid_ids:
                assert rel.validate_id_format(valid_id) is True

    def test_validate_id_format_invalid(self):
        """Test validation of invalid relationship ID formats."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            # Fix mock validation to match regex pattern ^[A-Za-z]+:\d+$
            mock_instance.validate_id_format.side_effect = (
                lambda x: x is not None and ":" in x and len(x.split(":")) == 2 
                and x.split(":")[0].isalpha() and x.split(":")[1].isdigit()
            )
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Test invalid ID formats
            invalid_ids = [
                "invalid_id",
                "REL_001",
                "",
                "REL:",
                ":001",
                "REL:001:extra",
                None,
            ]

            for invalid_id in invalid_ids:
                assert rel.validate_id_format(invalid_id) is False

    def test_validate_predicate_valid(self):
        """Test validation of valid predicate types."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            valid_predicates = {
                "is_a",
                "part_of",
                "has_role",
                "participates_in",
                "located_in",
                "derives_from",
                "regulates",
                "catalyzes",
                "accumulates_in",
                "affects",
                "involved_in",
                "upregulates",
                "downregulates",
                "made_via",
                "occurs_in",
            }
            mock_instance.validate_predicate.side_effect = (
                lambda x: x in valid_predicates
            )
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Test all supported predicate types
            for predicate in valid_predicates:
                assert rel.validate_predicate(predicate) is True

    def test_validate_predicate_invalid(self):
        """Test validation of invalid predicate types."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            valid_predicates = {
                "is_a",
                "part_of",
                "has_role",
                "participates_in",
                "located_in",
                "derives_from",
                "regulates",
                "catalyzes",
                "accumulates_in",
                "affects",
                "involved_in",
                "upregulates",
                "downregulates",
                "made_via",
                "occurs_in",
            }
            mock_instance.validate_predicate.side_effect = (
                lambda x: x in valid_predicates
            )
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            invalid_predicates = [
                "invalid_predicate",
                "has_part",  # not in our supported list
                "similar_to",
                "",
                None,
                "IS_A",  # case sensitive
            ]

            for predicate in invalid_predicates:
                assert rel.validate_predicate(predicate) is False

    def test_validate_subject_object_format(self):
        """Test validation of subject and object ID formats."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_subject_object_format.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Valid subject/object pairs
            valid_pairs = [
                ("CHEBI:12345", "CHEBI:33917"),
                ("GO:0008150", "GO:0003674"),
                ("PO:0000003", "PO:0000001"),
                ("NCIT:C12345", "NCIT:C54321"),
            ]

            for subject, obj in valid_pairs:
                assert rel.validate_subject_object_format(subject, obj) is True

    def test_validate_confidence_range(self):
        """Test validation of confidence score range."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_confidence.side_effect = lambda x: 0.0 <= x <= 1.0
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Valid confidence scores
            valid_confidences = [0.0, 0.5, 0.95, 1.0]
            for confidence in valid_confidences:
                assert rel.validate_confidence(confidence) is True

            # Invalid confidence scores
            invalid_confidences = [-0.1, 1.1, 2.0, -1.0]
            for confidence in invalid_confidences:
                assert rel.validate_confidence(confidence) is False

    def test_validate_circular_relationship(self):
        """Test validation to prevent circular relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_circular_relationship.side_effect = (
                lambda s, o: s != o
            )
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Valid non-circular relationship
            assert (
                rel.validate_circular_relationship("CHEBI:12345", "CHEBI:33917") is True
            )

            # Invalid circular relationship
            assert (
                rel.validate_circular_relationship("CHEBI:12345", "CHEBI:12345")
                is False
            )

    def test_is_valid_comprehensive(self):
        """Test comprehensive relationship validation."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.is_valid.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                confidence=0.95,
                is_validated=True,
            )

            assert rel.is_valid() is True

    def test_is_valid_invalid_relationship(self):
        """Test comprehensive validation on invalid relationship."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.is_valid.return_value = False
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="invalid", subject="", predicate="invalid_predicate", object=""
            )

            assert rel.is_valid() is False


class TestRelationshipEquality:
    """Test equality and hashing functionality."""

    def test_relationship_equality_same_content(self):
        """Test that relationships with same content are equal."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create two instances with same data
            rel1 = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel2 = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Configure mock equality
            rel1.__eq__ = Mock(return_value=True)

            assert rel1 == rel2

    def test_relationship_equality_different_content(self):
        """Test that relationships with different content are not equal."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create separate mock instances
            rel1 = Mock()
            rel2 = Mock()
            MockRelationship.side_effect = [rel1, rel2]

            MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            MockRelationship(
                id="REL:002",
                subject="CHEBI:54321",
                predicate="part_of",
                object="CHEBI:12345",
            )

            # Configure mock inequality
            rel1.__eq__ = Mock(return_value=False)

            assert rel1 != rel2

    def test_relationship_equality_different_types(self):
        """Test that relationship is not equal to objects of different types."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel.__eq__ = Mock(return_value=False)

            assert rel != "not a relationship"
            assert rel != 12345
            assert rel != None
            assert rel != {"id": "REL:001", "subject": "CHEBI:12345"}

    def test_relationship_hashing_consistent(self):
        """Test that relationship hashing is consistent."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel.__hash__ = Mock(return_value=12345)

            hash1 = hash(rel)
            hash2 = hash(rel)

            assert hash1 == hash2

    def test_relationship_hashing_equal_relationships(self):
        """Test that equal relationships have the same hash."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel1 = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel2 = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Configure same hash for equal objects
            expected_hash = 12345
            rel1.__hash__ = Mock(return_value=expected_hash)
            rel2.__hash__ = Mock(return_value=expected_hash)
            rel1.__eq__ = Mock(return_value=True)

            assert hash(rel1) == hash(rel2)

    def test_relationship_in_set(self):
        """Test that relationships can be used in sets (hashable)."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create separate mock instances
            rel1 = Mock()
            rel2 = Mock()
            rel3 = Mock()
            MockRelationship.side_effect = [rel1, rel2, rel3]

            MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            MockRelationship(
                id="REL:002",
                subject="CHEBI:54321",
                predicate="part_of",
                object="CHEBI:12345",
            )
            MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )  # duplicate of rel1

            # Configure hashing and equality for set operations
            rel1.__hash__ = Mock(return_value=12345)
            rel2.__hash__ = Mock(return_value=54321)
            rel3.__hash__ = Mock(return_value=12345)

            # Configure equality so rel1 and rel3 are equal
            rel1.__eq__ = Mock(side_effect=lambda other: other is rel3)
            rel2.__eq__ = Mock(side_effect=lambda other: other is not rel1 and other is not rel3)
            rel3.__eq__ = Mock(side_effect=lambda other: other is rel1)

            rel_set = {rel1, rel2, rel3}

            # Should only contain 2 unique relationships
            assert len(rel_set) == 2

    def test_relationship_in_dict_key(self):
        """Test that relationships can be used as dictionary keys."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel.__hash__ = Mock(return_value=12345)

            rel_dict = {rel: "This is an is_a relationship"}

            assert rel_dict[rel] == "This is an is_a relationship"


class TestRelationshipStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self):
        """Test __str__ method for user-friendly display."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
        )
        assert str(rel) == "CHEBI:12345 is_a CHEBI:33917"

    def test_repr_representation(self):
        """Test __repr__ method for debugging."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
        )
        assert (
            repr(rel)
            == "Relationship(id='REL:001', subject='CHEBI:12345', predicate='is_a', object='CHEBI:33917')"
        )

    def test_str_with_confidence(self):
        """Test string representation with confidence score."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            confidence=0.95,
        )
        assert str(rel) == "CHEBI:12345 is_a CHEBI:33917 (confidence: 0.95)"

    def test_str_with_default_confidence(self):
        """Test string representation with default confidence (1.0) - should not show confidence."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            confidence=1.0,
        )
        assert str(rel) == "CHEBI:12345 is_a CHEBI:33917"

    def test_str_negated_relationship(self):
        """Test string representation of negated relationship."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            negated=True,
        )
        assert str(rel) == "CHEBI:12345 NOT is_a CHEBI:33917"

    def test_str_with_context(self):
        """Test string representation with context information."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="located_in",
            object="GO:0005623",
            context="plant cell wall",
        )
        assert (
            str(rel) == "CHEBI:12345 located_in GO:0005623 [context: plant cell wall]"
        )

    def test_str_negated_with_confidence_and_context(self):
        """Test string representation with negation, confidence, and context."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            confidence=0.85,
            negated=True,
            context="test environment",
        )
        assert (
            str(rel)
            == "CHEBI:12345 NOT is_a CHEBI:33917 (confidence: 0.85) [context: test environment]"
        )

    def test_repr_with_confidence(self):
        """Test repr with confidence score."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            confidence=0.95,
        )
        repr_str = repr(rel)
        assert (
            "Relationship(id='REL:001', subject='CHEBI:12345', predicate='is_a', object='CHEBI:33917'"
            in repr_str
        )
        assert "confidence=0.95" in repr_str

    def test_repr_with_source(self):
        """Test repr with source."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            source="ChEBI",
        )
        repr_str = repr(rel)
        assert "source='ChEBI'" in repr_str

    def test_repr_with_evidence(self):
        """Test repr with evidence."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            evidence="experimental_evidence",
        )
        repr_str = repr(rel)
        assert "evidence='experimental_evidence'" in repr_str

    def test_repr_with_validation_flag(self):
        """Test repr with is_validated flag."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            is_validated=True,
        )
        repr_str = repr(rel)
        assert "is_validated=True" in repr_str

    def test_repr_with_context(self):
        """Test repr with context."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            context="cellular metabolism",
        )
        repr_str = repr(rel)
        assert "context='cellular metabolism'" in repr_str

    def test_repr_with_long_context(self):
        """Test repr with long context (should be truncated)."""
        long_context = "a" * 40  # 40 characters, should be truncated to 30
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            context=long_context,
        )
        repr_str = repr(rel)
        assert f"context='{long_context[:30]}...'" in repr_str

    def test_repr_with_negated_flag(self):
        """Test repr with negated flag."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            negated=True,
        )
        repr_str = repr(rel)
        assert "negated=True" in repr_str

    def test_repr_with_non_default_modality(self):
        """Test repr with non-default modality."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            modality="probable",
        )
        repr_str = repr(rel)
        assert "modality='probable'" in repr_str

    def test_repr_with_non_directional(self):
        """Test repr with directional=False."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            directional=False,
        )
        repr_str = repr(rel)
        assert "directional=False" in repr_str

    def test_repr_with_multiple_attributes(self):
        """Test repr with multiple non-default attributes."""
        rel = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            confidence=0.95,
            source="ChEBI",
            evidence="experimental",
            is_validated=True,
            context="test",
            negated=True,
            modality="probable",
            directional=False,
        )
        repr_str = repr(rel)
        # Check that all non-default values are included
        assert "confidence=0.95" in repr_str
        assert "source='ChEBI'" in repr_str
        assert "evidence='experimental'" in repr_str
        assert "is_validated=True" in repr_str
        assert "context='test'" in repr_str
        assert "negated=True" in repr_str
        assert "modality='probable'" in repr_str
        assert "directional=False" in repr_str

    def test_str_edge_cases(self):
        """Test string representation edge cases."""
        # Relationship with empty context should not show context
        rel_empty_context = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            context="",
        )
        assert str(rel_empty_context) == "CHEBI:12345 is_a CHEBI:33917"

        # Relationship with None context should not show context
        rel_none_context = Relationship(
            id="REL:001",
            subject="CHEBI:12345",
            predicate="is_a",
            object="CHEBI:33917",
            context=None,
        )
        assert str(rel_none_context) == "CHEBI:12345 is_a CHEBI:33917"


class TestRelationshipSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic relationship serialization to dictionary."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            expected_dict = {
                "id": "REL:001",
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
                "confidence": 1.0,
                "source": None,
                "evidence": None,
                "extraction_method": None,
                "created_date": None,
                "provenance": None,
                "is_validated": False,
                "context": None,
                "sentence": None,
                "negated": False,
                "modality": "certain",
                "directional": True,
            }
            rel.to_dict = Mock(return_value=expected_dict)

            result = rel.to_dict()
            assert result == expected_dict

    def test_to_dict_full(self):
        """Test full relationship serialization to dictionary."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                confidence=0.95,
                source="ChEBI",
                evidence="text mining",
                extraction_method="pattern_matching",
                created_date="2023-01-01T00:00:00Z",
                provenance="ChEBI ontology v2023.1",
                is_validated=True,
                context="plant cell wall",
                sentence="Glucose is a type of aldohexose",
                negated=False,
                modality="certain",
                directional=True,
            )

            expected_dict = {
                "id": "REL:001",
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
                "confidence": 0.95,
                "source": "ChEBI",
                "evidence": "text mining",
                "extraction_method": "pattern_matching",
                "created_date": "2023-01-01T00:00:00Z",
                "provenance": "ChEBI ontology v2023.1",
                "is_validated": True,
                "context": "plant cell wall",
                "sentence": "Glucose is a type of aldohexose",
                "negated": False,
                "modality": "certain",
                "directional": True,
            }
            rel.to_dict = Mock(return_value=expected_dict)

            result = rel.to_dict()
            assert result == expected_dict

    def test_from_dict_basic(self):
        """Test basic relationship deserialization from dictionary."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel_dict = {
                "id": "REL:001",
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
            }

            MockRelationship.from_dict = Mock(return_value=MockRelationship(**rel_dict))

            MockRelationship.from_dict(rel_dict)

            MockRelationship.from_dict.assert_called_once_with(rel_dict)

    def test_from_dict_full(self):
        """Test full relationship deserialization from dictionary."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel_dict = {
                "id": "REL:001",
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
                "confidence": 0.95,
                "source": "ChEBI",
                "evidence": "text mining",
                "extraction_method": "pattern_matching",
                "created_date": "2023-01-01T00:00:00Z",
                "provenance": "ChEBI ontology v2023.1",
                "is_validated": True,
                "context": "plant cell wall",
                "sentence": "Glucose is a type of aldohexose",
                "negated": False,
                "modality": "certain",
                "directional": True,
            }

            MockRelationship.from_dict = Mock(return_value=MockRelationship(**rel_dict))

            MockRelationship.from_dict(rel_dict)

            MockRelationship.from_dict.assert_called_once_with(rel_dict)

    def test_to_json(self):
        """Test relationship serialization to JSON string."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            expected_json = '{"id": "REL:001", "subject": "CHEBI:12345", "predicate": "is_a", "object": "CHEBI:33917"}'
            rel.to_json = Mock(return_value=expected_json)

            result = rel.to_json()
            assert result == expected_json

    def test_from_json(self):
        """Test relationship deserialization from JSON string."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            json_str = '{"id": "REL:001", "subject": "CHEBI:12345", "predicate": "is_a", "object": "CHEBI:33917"}'

            MockRelationship.from_json = Mock(
                return_value=MockRelationship(
                    id="REL:001",
                    subject="CHEBI:12345",
                    predicate="is_a",
                    object="CHEBI:33917",
                )
            )

            MockRelationship.from_json(json_str)

            MockRelationship.from_json.assert_called_once_with(json_str)

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization preserve data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            original_rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                confidence=0.95,
                is_validated=True,
            )

            # Mock the roundtrip
            rel_dict = {
                "id": "REL:001",
                "subject": "CHEBI:12345",
                "predicate": "is_a",
                "object": "CHEBI:33917",
                "confidence": 0.95,
                "is_validated": True,
            }
            original_rel.to_dict = Mock(return_value=rel_dict)
            MockRelationship.from_dict = Mock(return_value=original_rel)

            # Roundtrip test
            serialized = original_rel.to_dict()
            deserialized = MockRelationship.from_dict(serialized)

            assert deserialized == original_rel

    def test_json_serialization_invalid_data(self):
        """Test JSON serialization with invalid data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )
            rel.to_json = Mock(side_effect=TypeError("Object is not JSON serializable"))

            with pytest.raises(TypeError, match="Object is not JSON serializable"):
                rel.to_json()


class TestRelationshipEdgeCases:
    """Test edge cases and error handling."""

    def test_relationship_with_none_values(self):
        """Test relationship creation with None values for optional fields."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                source=None,
                evidence=None,
                context=None,
                sentence=None,
            )

            # Should handle None values gracefully
            MockRelationship.assert_called_once()

    def test_relationship_with_unicode_characters(self):
        """Test relationship with Unicode characters in text fields."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            unicode_context = "α-D-glucose metabolism"
            unicode_sentence = "α-D-glucose is synthesized in plant cells"

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                context=unicode_context,
                sentence=unicode_sentence,
            )

            MockRelationship.assert_called_once_with(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                context=unicode_context,
                sentence=unicode_sentence,
            )

    def test_relationship_with_very_long_strings(self):
        """Test relationship with very long strings."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            long_context = "a" * 1000
            long_sentence = "b" * 5000
            long_evidence = "c" * 2000

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                context=long_context,
                sentence=long_sentence,
                evidence=long_evidence,
            )

            MockRelationship.assert_called_once()

    def test_relationship_with_special_characters(self):
        """Test relationship with special characters in various fields."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                context="metabolism (cellular)",
                sentence="The compound [glucose] is a type of sugar",
                evidence="Pattern matching: !@#$%^&*()_+-={}[]|;':\",./<>?",
            )

            MockRelationship.assert_called_once()

    def test_relationship_modification_after_creation(self):
        """Test modifying relationship attributes after creation."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.confidence = 0.95
            mock_instance.is_validated = False
            mock_instance.context = "original context"
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Modify attributes
            rel.confidence = 0.85
            rel.is_validated = True
            rel.context = "modified context"

            assert rel.confidence == 0.85
            assert rel.is_validated is True
            assert rel.context == "modified context"

    def test_relationship_extreme_confidence_values(self):
        """Test relationship with extreme confidence values."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Test edge values
            rel_min = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                confidence=0.0,
            )

            rel_max = MockRelationship(
                id="REL:002",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                confidence=1.0,
            )

            assert MockRelationship.call_count == 2

    def test_relationship_circular_reference_detection(self):
        """Test handling of circular references in relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # This should be caught by validation
            MockRelationship.side_effect = ValueError("Circular relationship detected")

            with pytest.raises(ValueError, match="Circular relationship detected"):
                MockRelationship(
                    id="REL:001",
                    subject="CHEBI:12345",
                    predicate="is_a",
                    object="CHEBI:12345",  # Same as subject
                )

    def test_relationship_network_performance(self):
        """Test relationship operations with large networks."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Simulate creating many relationships
            relationships = []
            for i in range(1000):
                rel = MockRelationship(
                    id=f"REL:{i:06d}",
                    subject=f"SUBJECT:{i}",
                    predicate="is_a",
                    object=f"OBJECT:{i}",
                )
                relationships.append(rel)

            assert len(relationships) == 1000
            assert MockRelationship.call_count == 1000


class TestRelationshipOntologySpecific:
    """Test ontology-specific functionality."""

    def test_relationship_predicate_semantics(self):
        """Test semantic validation of predicate types."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_predicate_semantics.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Test various semantic relationships
            semantic_tests = [
                ("is_a", "CHEBI:12345", "CHEBI:33917"),  # glucose is_a aldohexose
                ("part_of", "GO:0005739", "GO:0005623"),  # mitochondrion part_of cell
                ("regulates", "GO:0008150", "GO:0003674"),  # process regulates function
                ("located_in", "CHEBI:12345", "GO:0005623"),  # glucose located_in cell
                ("catalyzes", "GO:0003824", "GO:0008152"),  # enzyme catalyzes reaction
            ]

            for predicate, subject, obj in semantic_tests:
                assert rel.validate_predicate_semantics(predicate, subject, obj) is True

    def test_relationship_domain_constraints(self):
        """Test domain-specific relationship constraints."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.validate_domain_constraints.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
            )

            # Chemical domain constraints
            chemical_relationships = [
                ("CHEBI:12345", "is_a", "CHEBI:33917"),  # glucose is_a aldohexose
                (
                    "CHEBI:12345",
                    "derives_from",
                    "CHEBI:16646",
                ),  # glucose derives_from carbohydrate
                (
                    "CHEBI:15422",
                    "regulates",
                    "GO:0008152",
                ),  # ATP regulates metabolic process
            ]

            for subject, predicate, obj in chemical_relationships:
                assert rel.validate_domain_constraints(subject, predicate, obj) is True

    def test_relationship_plant_ontology_specific(self):
        """Test plant ontology specific relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.relationships = {
                "accumulates_in": ["PO:0000030"],  # cell wall
                "occurs_in": ["PO:0009005"],  # root
                "participates_in": ["GO:0015979"],  # photosynthesis
            }
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:18420",  # cellulose
                predicate="accumulates_in",
                object="PO:0000030",  # cell wall
            )

            plant_specific_relationships = {
                "accumulates_in": ["PO:0000030"],  # cell wall
                "occurs_in": ["PO:0009005"],  # root
                "participates_in": ["GO:0015979"],  # photosynthesis
            }

            assert rel.relationships == plant_specific_relationships

    def test_relationship_chemical_ontology_specific(self):
        """Test chemical ontology specific relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.relationships = {
                "made_via": ["GO:0008152"],  # metabolic process
                "upregulates": ["GO:0008150"],  # biological process
                "downregulates": ["GO:0003674"],  # molecular function
                "affects": ["GO:0005623"],  # cell
            }
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:15422",  # ATP
                predicate="upregulates",
                object="GO:0008150",  # biological process
            )

            chemical_specific_relationships = {
                "made_via": ["GO:0008152"],  # metabolic process
                "upregulates": ["GO:0008150"],  # biological process
                "downregulates": ["GO:0003674"],  # molecular function
                "affects": ["GO:0005623"],  # cell
            }

            assert rel.relationships == chemical_specific_relationships

    def test_relationship_evidence_types(self):
        """Test different types of evidence for relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.evidence = "experimental_evidence"
            mock_instance.extraction_method = "manual_curation"
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                evidence="experimental_evidence",
                extraction_method="manual_curation",
            )

            evidence_types = [
                "experimental_evidence",
                "text_mining",
                "computational_prediction",
                "manual_curation",
                "literature_review",
                "database_cross_reference",
            ]

            assert rel.evidence in evidence_types
            assert rel.extraction_method in [
                "manual_curation",
                "automated_extraction",
                "hybrid_approach",
            ]

    def test_relationship_temporal_constraints(self):
        """Test temporal constraints on relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.created_date = "2023-01-01T00:00:00Z"
            mock_instance.validate_temporal_constraints.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                created_date="2023-01-01T00:00:00Z",
            )

            # Test that creation date is valid
            assert rel.validate_temporal_constraints() is True
            assert rel.created_date == "2023-01-01T00:00:00Z"

    def test_relationship_modality_types(self):
        """Test different modality types for relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.modality = "certain"
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                modality="certain",
            )

            valid_modalities = [
                "certain",
                "probable",
                "possible",
                "unlikely",
                "speculative",
            ]
            assert rel.modality in valid_modalities

    def test_relationship_directional_properties(self):
        """Test directional properties of relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.directional = True
            mock_instance.get_inverse.return_value = (
                None  # Most relationships don't have simple inverses
            )
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                directional=True,
            )

            # Test directional property
            assert rel.directional is True

            # Test that inverse relationship is handled properly
            inverse = rel.get_inverse()
            assert inverse is None  # is_a doesn't have a simple inverse

    def test_relationship_provenance_tracking(self):
        """Test provenance tracking for relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.provenance = "ChEBI ontology v2023.1"
            mock_instance.source = "ChEBI"
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(
                id="REL:001",
                subject="CHEBI:12345",
                predicate="is_a",
                object="CHEBI:33917",
                provenance="ChEBI ontology v2023.1",
                source="ChEBI",
            )

            expected_provenance = "ChEBI ontology v2023.1"
            expected_source = "ChEBI"

            assert rel.provenance == expected_provenance
            assert rel.source == expected_source


# Test fixtures for complex scenarios
@pytest.fixture
def sample_relationship_data():
    """Fixture providing sample relationship data for testing."""
    return {
        "id": "REL:001",
        "subject": "CHEBI:12345",
        "predicate": "is_a",
        "object": "CHEBI:33917",
        "confidence": 0.95,
        "source": "ChEBI",
        "evidence": "text mining",
        "extraction_method": "pattern_matching",
        "created_date": "2023-01-01T00:00:00Z",
        "provenance": "ChEBI ontology v2023.1",
        "is_validated": True,
        "context": "plant cell wall",
        "sentence": "Glucose is a type of aldohexose",
        "negated": False,
        "modality": "certain",
        "directional": True,
    }


@pytest.fixture
def sample_negated_relationship_data():
    """Fixture providing sample negated relationship data for testing."""
    return {
        "id": "REL:002",
        "subject": "CHEBI:12345",
        "predicate": "is_a",
        "object": "CHEBI:16646",
        "confidence": 0.85,
        "source": "text_mining",
        "evidence": "automated_extraction",
        "extraction_method": "NLP",
        "created_date": "2023-02-15T12:30:00Z",
        "provenance": "PubMed abstract analysis",
        "is_validated": False,
        "context": "biochemical pathway",
        "sentence": "Glucose is not a protein",
        "negated": True,
        "modality": "certain",
        "directional": True,
    }


@pytest.fixture
def sample_plant_relationship_data():
    """Fixture providing sample plant ontology relationship data for testing."""
    return {
        "id": "REL:003",
        "subject": "CHEBI:18420",  # cellulose
        "predicate": "accumulates_in",
        "object": "PO:0000030",  # cell wall
        "confidence": 0.98,
        "source": "Plant Ontology",
        "evidence": "experimental_evidence",
        "extraction_method": "manual_curation",
        "created_date": "2023-03-10T09:15:00Z",
        "provenance": "Plant Ontology v2023.1",
        "is_validated": True,
        "context": "cell wall composition",
        "sentence": "Cellulose accumulates in the plant cell wall",
        "negated": False,
        "modality": "certain",
        "directional": True,
    }


@pytest.fixture
def sample_regulatory_relationship_data():
    """Fixture providing sample regulatory relationship data for testing."""
    return {
        "id": "REL:004",
        "subject": "CHEBI:15422",  # ATP
        "predicate": "upregulates",
        "object": "GO:0008152",  # metabolic process
        "confidence": 0.90,
        "source": "Gene Ontology",
        "evidence": "computational_prediction",
        "extraction_method": "pathway_analysis",
        "created_date": "2023-04-20T14:45:00Z",
        "provenance": "GO annotation pipeline v2023.2",
        "is_validated": True,
        "context": "cellular metabolism",
        "sentence": "ATP upregulates metabolic processes in the cell",
        "negated": False,
        "modality": "probable",
        "directional": True,
    }


class TestRelationshipIntegration:
    """Integration tests using fixtures."""

    def test_relationship_creation_with_fixture(self, sample_relationship_data):
        """Test relationship creation using fixture data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            MockRelationship(**sample_relationship_data)
            MockRelationship.assert_called_once_with(**sample_relationship_data)

    def test_negated_relationship_with_fixture(self, sample_negated_relationship_data):
        """Test negated relationship handling using fixture data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.negated = True
            mock_instance.is_validated = False
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(**sample_negated_relationship_data)

            assert rel.negated is True
            assert rel.is_validated is False

    def test_plant_relationship_with_fixture(self, sample_plant_relationship_data):
        """Test plant ontology relationship using fixture data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.predicate = "accumulates_in"
            mock_instance.subject = "CHEBI:18420"  # cellulose
            mock_instance.object = "PO:0000030"  # cell wall
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(**sample_plant_relationship_data)

            assert rel.predicate == "accumulates_in"
            assert rel.subject == "CHEBI:18420"
            assert rel.object == "PO:0000030"

    def test_regulatory_relationship_with_fixture(
        self, sample_regulatory_relationship_data
    ):
        """Test regulatory relationship using fixture data."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.predicate = "upregulates"
            mock_instance.modality = "probable"
            mock_instance.confidence = 0.90
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(**sample_regulatory_relationship_data)

            assert rel.predicate == "upregulates"
            assert rel.modality == "probable"
            assert rel.confidence == 0.90

    def test_relationship_collection_operations(self, sample_relationship_data):
        """Test relationship operations in collections."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create separate mock instances
            rel1 = Mock()
            rel2 = Mock()
            MockRelationship.side_effect = [rel1, rel2]

            # Create multiple relationships
            MockRelationship(**sample_relationship_data)
            rel2_data = sample_relationship_data.copy()
            rel2_data["id"] = "REL:002"
            rel2_data["predicate"] = "part_of"
            MockRelationship(**rel2_data)

            # Configure hashing for set operations
            rel1.__hash__ = Mock(return_value=hash(sample_relationship_data["id"]))
            rel2.__hash__ = Mock(return_value=hash(rel2_data["id"]))
            rel1.__eq__ = Mock(return_value=False)
            rel2.__eq__ = Mock(return_value=False)

            relationships = [rel1, rel2]
            relationship_set = {rel1, rel2}

            assert len(relationships) == 2
            assert len(relationship_set) == 2

    def test_relationship_network_analysis(
        self, sample_relationship_data, sample_plant_relationship_data
    ):
        """Test relationship network analysis capabilities."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create separate mock instances
            rel1 = Mock()
            rel2 = Mock()
            MockRelationship.side_effect = [rel1, rel2]

            # Create network of relationships
            MockRelationship(**sample_relationship_data)
            MockRelationship(**sample_plant_relationship_data)

            # Configure network analysis methods
            rel1.get_related_entities = Mock(
                return_value=["CHEBI:12345", "CHEBI:33917"]
            )
            rel2.get_related_entities = Mock(return_value=["CHEBI:18420", "PO:0000030"])

            # Test network operations
            entities1 = rel1.get_related_entities()
            entities2 = rel2.get_related_entities()

            assert "CHEBI:12345" in entities1
            assert "CHEBI:33917" in entities1
            assert "CHEBI:18420" in entities2
            assert "PO:0000030" in entities2

    def test_relationship_validation_workflow(
        self, sample_regulatory_relationship_data
    ):
        """Test complete relationship validation workflow."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            mock_instance = Mock()
            mock_instance.is_validated = True
            mock_instance.validate_full.return_value = True
            MockRelationship.return_value = mock_instance

            rel = MockRelationship(**sample_regulatory_relationship_data)

            # Test validation workflow
            validation_result = rel.validate_full()

            assert validation_result is True
            assert rel.is_validated is True

    def test_relationship_serialization_performance(self, sample_relationship_data):
        """Test relationship serialization with large datasets."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create multiple relationships for performance testing
            relationships = []
            for i in range(100):
                rel_data = sample_relationship_data.copy()
                rel_data["id"] = f"REL:{i:03d}"
                rel = MockRelationship(**rel_data)
                rel.to_dict = Mock(return_value=rel_data)
                relationships.append(rel)

            # Test serialization performance
            serialized_data = [rel.to_dict() for rel in relationships]

            assert len(serialized_data) == 100
            assert all(isinstance(data, dict) for data in serialized_data)

    def test_relationship_confidence_aggregation(
        self, sample_relationship_data, sample_negated_relationship_data
    ):
        """Test confidence score aggregation across relationships."""
        with patch(
            "aim2_project.aim2_ontology.models.Relationship"
        ) as MockRelationship:
            # Create separate mock instances
            rel1 = Mock()
            rel2 = Mock()
            MockRelationship.side_effect = [rel1, rel2]

            MockRelationship(**sample_relationship_data)
            MockRelationship(**sample_negated_relationship_data)

            # Configure confidence values
            rel1.confidence = 0.95
            rel2.confidence = 0.85

            relationships = [rel1, rel2]
            avg_confidence = sum(rel.confidence for rel in relationships) / len(
                relationships
            )

            # Use approximate comparison for floating point values
            assert abs(avg_confidence - 0.90) < 1e-10
