import unittest

from derail.rollout.records import RolloutRecord


class RolloutRecordTests(unittest.TestCase):
    def test_records_source_benchmark_explicitly(self) -> None:
        record = RolloutRecord(
            rollout_id="rollout-1",
            source_benchmark="mypcbench",
            task_id="task-1",
            agent_id="agent-1",
            snapshot_sha256="vm-snapshot-sha256",
            trajectory_uri="artifacts/rollout-1.jsonl",
            trajectory_sha256="trajectory-sha256",
            task_success=False,
            step_count=73,
        )
        self.assertEqual(record.source_benchmark, "mypcbench")

    def test_rejects_unknown_source_benchmark(self) -> None:
        with self.assertRaises(ValueError):
            RolloutRecord(
                rollout_id="rollout-2",
                source_benchmark="osworld",
                task_id="task-2",
                agent_id="agent-1",
                snapshot_sha256="snapshot-sha256",
                trajectory_uri="artifacts/rollout-2.jsonl",
                trajectory_sha256="trajectory-sha256",
                task_success=False,
                step_count=10,
            )


if __name__ == "__main__":
    unittest.main()
