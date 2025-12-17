from utils.context import build_context
from model.hybrid_agent import HybridAgent

agent = HybridAgent()

conversation = [
    "Hi",
    "Are you busy right now?",
    "I have something important to share",
    "You have been selected for a free gift card",
    "Act now before it expires",
]

# ❌ single-message test
single = conversation[-1]
print("SINGLE MESSAGE:")
print(single)
print(agent.predict(single))

print("\n" + "=" * 60)

# ✅ context test
context = build_context(conversation)
print("CONTEXT MESSAGE:")
print(context)
print(agent.predict(context))
