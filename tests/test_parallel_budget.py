"""Parallel node selection must respect draft and remaining stage budgets."""

import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import ParallelAgent


class ParallelBudgetTests(unittest.TestCase):
    def agent(self, num_drafts=1):
        agent = ParallelAgent.__new__(ParallelAgent)
        agent.num_workers = 3
        agent.stage_name = "1_initial_implementation_1_preliminary"
        agent.cfg = OmegaConf.create({"agent": {"search": {
            "num_drafts": num_drafts, "debug_prob": 0, "max_debug_depth": 3
        }}})
        agent.journal = Journal()
        return agent

    def test_pending_drafts_count_toward_initial_draft_limit(self):
        for drafts in (1, 2, 5):
            with self.subTest(drafts=drafts):
                agent = self.agent(drafts)
                self.assertEqual(agent._select_parallel_nodes(), [None] * min(drafts, 3))

    def test_only_remaining_drafts_are_created_alongside_improvements(self):
        agent = self.agent(2)
        parent = Node(is_buggy=False, is_buggy_plots=False)
        agent.journal.append(parent)
        with patch.object(agent.journal, "get_best_node", return_value=parent):
            selected = agent._select_parallel_nodes()
        self.assertEqual(selected, [None, parent, parent])

    def test_last_batch_is_bounded_by_remaining_stage_budget(self):
        agent = self.agent()
        parent = Node(is_buggy=False, is_buggy_plots=False)
        agent.journal.append(parent)
        with patch.object(agent.journal, "get_best_node", return_value=parent):
            self.assertEqual(agent._select_parallel_nodes(max_nodes=1), [parent])
            self.assertEqual(agent._select_parallel_nodes(max_nodes=2), [parent, parent])
            self.assertEqual(agent._select_parallel_nodes(max_nodes=10), [parent] * 3)

    def test_exhausted_budget_skips_summary_and_executor(self):
        agent = self.agent()
        agent.executor = Mock()
        with patch.object(agent.journal, "generate_summary") as summarize:
            agent.step(None, max_nodes=0)
        summarize.assert_not_called()
        agent.executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
