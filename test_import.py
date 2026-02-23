import sys
import traceback

print("Testing imports...")

try:
    import research_agent.tools
    print("research_agent.tools: OK")
except Exception as e:
    print(f"research_agent.tools ERROR: {e}")
    traceback.print_exc()

try:
    import agent
    print("agent: OK")
except Exception as e:
    print(f"agent ERROR: {e}")
    traceback.print_exc()
