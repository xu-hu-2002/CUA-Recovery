from __future__ import annotations

import json

from derail.phase5.verifier import verify_changelog


def gold() -> dict:
    return {
        "writes_gold": [
            {
                "table": "mail.sent",
                "column": "subject",
                "entity_ref": "derived:n1:message",
                "value": "Move plan",
                "volatile": False,
            },
            {
                "table": "mail.sent",
                "column": "date",
                "entity_ref": "derived:n1:message",
                "value": "ignored",
                "volatile": True,
            },
        ]
    }


def steps(subject: str) -> list[dict]:
    return [
        {
            "delta": [
                {
                    "db": "/data/mail.sqlite",
                    "tbl": "sent",
                    "new_json": json.dumps({"id": 99, "subject": subject}),
                }
            ]
        }
    ]


def test_matches_expected_entity_without_requiring_frozen_row_id() -> None:
    result = verify_changelog(gold(), steps("Move plan"))
    assert result["passed"] is True
    assert result["missing"] == []


def test_reports_missing_expected_entity() -> None:
    result = verify_changelog(gold(), steps("Wrong subject"))
    assert result["passed"] is False
    assert result["missing"][0]["entity_ref"] == "derived:n1:message"
