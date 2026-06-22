#!/usr/bin/env python3
"""Quick standalone connection check - run this first to see the REAL error."""
import os

client_id = "1110569990"
token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgyMTg2MzIwLCJpYXQiOjE3ODIwOTk5MjAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwNTY5OTkwIn0.LHn9fx3rowd5wq6TRxvA4wT4y-jqD_4_NtvIFbPR1rmARvFHP0EsTkRsATEihPAGm_xPcXjh09xLL73udEiH7Q"


print(f"DHAN_CLIENT_ID set: {bool(client_id)} (len={len(client_id)})")
print(f"DHAN_ACCESS_TOKEN set: {bool(token)} (len={len(token)})")

if not client_id or not token:
    print("\n--> One or both env vars are empty in THIS shell/process.")
    print("    Note: env vars set in one terminal session don't carry over to another.")
    print("    Make sure you export them in the SAME terminal/session you run the bot from,")
    print("    or put them in a .env file and load it, or hardcode them in Config for testing.")
    raise SystemExit(1)

try:
    from dhanhq import DhanContext, dhanhq
except ImportError as e:
    print(f"\n--> dhanhq package not installed or broken: {e}")
    print("    Run: pip install dhanhq")
    raise SystemExit(1)

try:
    ctx = DhanContext(client_id, token)
    dhan = dhanhq(ctx)
    print("\n--> DhanContext + dhanhq client created successfully.")
except Exception as e:
    print(f"\n--> Failed to construct DhanContext/dhanhq: {e}")
    raise SystemExit(1)

try:
    positions = dhan.get_positions()
    print(f"\n--> API call succeeded. get_positions() returned: {positions}")
    print("\nConnection is GOOD. The bot should be able to fetch data.")
except Exception as e:
    print(f"\n--> API call FAILED: {e}")
    print("    Common causes: expired/invalid access token, IP not whitelisted for your Dhan account,")
    print("    or client ID mismatch. Regenerate your access token from the Dhan developer console")
    print("    and double check IP whitelisting requirements for order/data APIs.")