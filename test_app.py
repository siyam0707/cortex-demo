import json
import unittest

import app


class DemoAdapterTests(unittest.TestCase):
    def test_removes_agent_prompt_tools_and_thinking(self):
        source = {
            "model": app.MODEL,
            "messages": [
                {"role": "system", "content": "Always call a tool."},
                {"role": "developer", "content": "Never answer."},
                {"role": "user", "content": "Hello!"},
            ],
            "tools": [{"type": "function", "function": {"name": "ask_clarification"}}],
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "stream": True,
            "max_tokens": 9000,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        cleaned = app.build_request(source)
        self.assertEqual(cleaned["messages"], [
            {"role": "system", "content": app.SYSTEM_PROMPT},
            {"role": "user", "content": "Hello!"},
        ])
        self.assertFalse(cleaned["stream"])
        self.assertFalse(cleaned["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(cleaned["max_tokens"], app.MAX_OUTPUT_TOKENS)
        self.assertNotIn("tools", cleaned)

    def test_rejects_tool_history(self):
        with self.assertRaisesRegex(ValueError, "Tool history"):
            app.build_request({"model": app.MODEL, "messages": [{"role": "tool", "content": "x"}]})

    def test_sse_has_content_usage_and_done(self):
        completion = {
            "id": "test", "created": 1, "model": app.MODEL,
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 4},
        }
        raw = app.sse(completion, include_usage=True).decode()
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        parts = [json.loads(line[6:]) for line in raw.splitlines()
                 if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertEqual(parts[1]["choices"][0]["delta"]["content"], "Hello!")
        self.assertEqual(parts[-1]["usage"]["total_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
