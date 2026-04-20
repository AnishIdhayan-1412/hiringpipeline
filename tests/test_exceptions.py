# -*- coding: utf-8 -*-
"""Tests for exceptions.py — verifies the typed exception hierarchy."""
import pytest
from exceptions import (
    PipelineError, ConfigurationError, AnonymizationError,
    PIILeakError, ParsingError, RankingError, ModelLoadError,
    ExplainabilityError, AuditError, BiasAuditError,
    DashboardError, DatabaseError,
)


def test_pipeline_error_base():
    err = PipelineError("something failed", stage="module1", candidate_id="Candidate_01")
    assert "something failed" in str(err)
    assert "module1" in str(err)
    assert "Candidate_01" in str(err)


def test_pipeline_error_no_optional_fields():
    err = PipelineError("bare error")
    assert str(err) == "bare error"


def test_anonymization_error_carries_filepath():
    err = AnonymizationError("parse failed", candidate_id="Candidate_02", filepath="/tmp/cv.pdf")
    assert "module0" in str(err)
    assert "/tmp/cv.pdf" in str(err)


def test_pii_leak_error_is_anonymization_error():
    err = PIILeakError("leaked email", candidate_id="Candidate_03", leaks=[{"type": "email"}], confidence=70.0)
    assert isinstance(err, AnonymizationError)
    assert err.confidence == 70.0
    assert len(err.leaks) == 1


def test_model_load_error_is_ranking_error():
    err = ModelLoadError("all-MiniLM-L6-v2", "no module named sentence_transformers")
    assert isinstance(err, RankingError)
    assert "all-MiniLM-L6-v2" in str(err)


def test_database_error_carries_operation():
    err = DatabaseError("constraint violation", operation="INSERT")
    assert "INSERT" in str(err)
    assert "database" in str(err)


def test_all_exceptions_inherit_pipeline_error():
    for cls in (ConfigurationError, AnonymizationError, PIILeakError,
                ParsingError, RankingError, ModelLoadError,
                ExplainabilityError, AuditError, BiasAuditError,
                DashboardError, DatabaseError):
        assert issubclass(cls, PipelineError), f"{cls.__name__} must inherit PipelineError"
