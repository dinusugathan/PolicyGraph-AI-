import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.graph import policy_graph_app

class TestPolicyGraphAI(unittest.TestCase):
    """Automated Golden Evaluation Suite for Akbar Travels PolicyGraph AI."""

    def test_b2c_cancellation_policy(self):
        """Verify B2C cancellation rules return INR 350 service fee and POL-CANCEL-001 citation."""
        result = policy_graph_app.invoke({
            "query": "What is the cancellation service fee for domestic flight B2C?",
            "channel": "B2C"
        })
        self.assertEqual(result["intent"], "policy_inquiry")
        self.assertIn("POL-CANCEL-001", result["citations"])
        self.assertIn("350", result["response"])
        self.assertFalse(result["escalated"])

    def test_b2b_cancellation_policy(self):
        """Verify B2B cancellation rules return INR 200 service fee and POL-CANCEL-002 citation."""
        result = policy_graph_app.invoke({
            "query": "What is the cancellation charge for B2B sub-agent bookings?",
            "channel": "B2B"
        })
        self.assertEqual(result["intent"], "policy_inquiry")
        self.assertIn("POL-CANCEL-002", result["citations"])
        self.assertIn("200", result["response"])
        self.assertFalse(result["escalated"])

    def test_refund_calculation_arithmetic(self):
        """Verify deterministic arithmetic in refund calculation node."""
        result_b2c = policy_graph_app.invoke({
            "query": "Calculate my net refund: paid 5000 and airline penalty is 1500",
            "channel": "B2C"
        })
        self.assertEqual(result_b2c["intent"], "refund_calculation")
        self.assertIsNotNone(result_b2c["refund_calculation"])
        # Paid: 5000, Penalty: 1500, B2C Fee: 350 -> Net = 3150
        self.assertEqual(result_b2c["refund_calculation"]["net_refund"], 3150)
        self.assertIn("POL-CANCEL-001", result_b2c["citations"])

        result_b2b = policy_graph_app.invoke({
            "query": "Calculate net refund: paid 5000 and airline penalty is 1500",
            "channel": "B2B"
        })
        # Paid: 5000, Penalty: 1500, B2B Fee: 200 -> Net = 3300
        self.assertEqual(result_b2b["refund_calculation"]["net_refund"], 3300)

    def test_human_escalation_grievance(self):
        """Verify complaint keywords trigger human supervisor escalation node."""
        result = policy_graph_app.invoke({
            "query": "I have an unresolved fraud complaint and will file in consumer court",
            "channel": "B2C"
        })
        self.assertEqual(result["intent"], "human_escalation")
        self.assertTrue(result["escalated"])
        self.assertIn("POL-COMPLAINT-001", result["citations"])
        self.assertIn("grievance@akbartravels.com", result["response"])

    def test_medical_emergency_waiver(self):
        """Verify medical emergency queries retrieve compassionate waiver policy."""
        result = policy_graph_app.invoke({
            "query": "My father was hospitalized, can I get a medical refund waiver?",
            "channel": "ALL"
        })
        self.assertEqual(result["intent"], "policy_inquiry")
        self.assertIn("POL-REFUND-002", result["citations"])
        self.assertIn("Compassionate Waiver", result["response"])

    def test_baggage_allowance_gulf(self):
        """Verify Gulf excess baggage policies are retrieved."""
        result = policy_graph_app.invoke({
            "query": "How much excess baggage rate for Gulf corridor flights?",
            "channel": "B2B"
        })
        self.assertIn("POL-BAGGAGE-002", result["citations"])

    def test_guardrails_pii_redaction(self):
        """Verify that 16-digit card numbers are redacted by the guardrail node."""
        result = policy_graph_app.invoke({
            "query": "Refund to my card 4111 2222 3333 4444 please, what is the policy?",
            "channel": "B2C"
        })
        self.assertTrue(result["guardrail_passed"])
        self.assertNotIn("4111 2222 3333 4444", result["response"])

    def test_graph_state_machine_trace(self):
        """Verify that the state graph records chronological execution spans for every node."""
        result = policy_graph_app.invoke({
            "query": "What are the Akbar Rewards points on international flights?",
            "channel": "B2C"
        })
        traces = result.get("trace_steps", [])
        self.assertGreaterEqual(len(traces), 3)
        node_names = [t["node"] for t in traces]
        self.assertIn("router_node", node_names)
        self.assertIn("retriever_node", node_names)
        self.assertIn("guardrail_node", node_names)

if __name__ == "__main__":
    unittest.main()
