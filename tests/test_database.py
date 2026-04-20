# -*- coding: utf-8 -*-
"""Tests for database.py — verifies schema, CRUD, and dashboard stats."""
import sqlite3
import pytest
from database import Database
from datetime import datetime


def _run_id():
    return "RUN_" + datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture
def db(tmp_path):
    return Database(db_path=str(tmp_path / "test.db"))


def test_schema_created(db):
    conn = sqlite3.connect(db._path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert {"pipeline_runs", "candidates", "ranking_results", "audit_events"}.issubset(tables)


def test_create_and_finish_run(db):
    rid = _run_id()
    db.create_run(rid, jd_file="jd.txt", jd_text="Data Scientist")
    db.finish_run(rid, exit_code=0, total_cvs=3)
    run = db.get_run(rid)
    assert run is not None
    assert run["status"] == "completed"


def test_get_latest_run_id_empty(db):
    assert db.get_latest_run_id() is None


def test_get_latest_run_id(db):
    rid = _run_id()
    db.create_run(rid)
    assert db.get_latest_run_id() == rid


def test_upsert_candidate(db):
    rid = _run_id()
    db.create_run(rid)
    vault = {
        "candidate_id": "Candidate_01",
        "original_filename": "cv.pdf",
        "pii_categories_found": {"emails": 1, "phones": 1},
        "validation_score": {"passed": True, "confidence_score": 95.0},
    }
    parsed = {
        "quality_score": 0.82,
        "sections": {
            "total_experience_months": 36,
            "skills": {"technical": ["Python", "SQL"], "soft": ["Leadership"]},
            "education": [],
        },
    }
    # API: upsert_candidate(run_id, candidate_id, vault_data, parsed_data)
    db.upsert_candidate(rid, "Candidate_01", vault, parsed)
    candidates = db.list_candidates(rid)
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "Candidate_01"
    assert candidates[0]["total_exp_months"] == 36


def test_save_ranking(db):
    rid = _run_id()
    db.create_run(rid)
    # API: save_ranking(run_id, ranking_details, explanation_map)
    ranking_details = {
        "Candidate_01": {
            "rank": 1, "final_score": 0.87,
            "semantic_score": 0.82, "keyword_score": 0.75, "quality_score": 0.9,
            "matched_skills": ["Python", "SQL"],
            "missing_skills": ["Tableau"], "jd_skill_count": 3,
        }
    }
    explanation_map = {
        "Candidate_01": {
            "verdict": "Strong Match",
            "explanation": {"recommendation": "Recommended for interview."},
        }
    }
    db.save_ranking(rid, ranking_details, explanation_map)
    rankings = db.get_rankings(rid)
    assert len(rankings) == 1
    assert rankings[0]["rank"] == 1
    assert rankings[0]["verdict"] == "Strong Match"


def test_dashboard_stats(db):
    rid = _run_id()
    db.create_run(rid)
    ranking_details = {
        f"Candidate_0{i}": {
            "rank": i, "final_score": score,
            "semantic_score": score, "keyword_score": score, "quality_score": 0.7,
            "matched_skills": [], "missing_skills": [], "jd_skill_count": 5,
        }
        for i, score in enumerate([0.88, 0.62, 0.35], start=1)
    }
    exp_map = {
        "Candidate_01": {"verdict": "Strong Match", "explanation": {"recommendation": "x"}},
        "Candidate_02": {"verdict": "Good Match", "explanation": {"recommendation": "x"}},
        "Candidate_03": {"verdict": "Weak Match", "explanation": {"recommendation": "x"}},
    }
    db.save_ranking(rid, ranking_details, exp_map)
    db.finish_run(rid, exit_code=0, total_cvs=3)
    stats = db.get_dashboard_stats(rid)
    assert stats["top_candidate"] == "Candidate_01"
    assert stats["top_score"] > 0.8


def test_idempotent_schema(tmp_path):
    path = str(tmp_path / "idempotent.db")
    db1 = Database(db_path=path)
    db2 = Database(db_path=path)
    assert db1._path == db2._path
    db1.close()
    db2.close()
