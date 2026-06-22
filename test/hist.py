#!/usr/bin/env python3
"""Print the RAW response of historical_daily_data so we can see its real shape."""
import os
import json
from dhanhq import DhanContext, dhanhq

client_id = "1110569990"
token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgyMTg2MzIwLCJpYXQiOjE3ODIwOTk5MjAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwNTY5OTkwIn0.LHn9fx3rowd5wq6TRxvA4wT4y-jqD_4_NtvIFbPR1rmARvFHP0EsTkRsATEihPAGm_xPcXjh09xLL73udEiH7Q"


ctx = DhanContext(client_id, token)
dhan = dhanhq(ctx)

resp = dhan.historical_daily_data(
    security_id="13",              # NIFTY index
    exchange_segment="IDX_I",
    instrument_type="INDEX",
    from_date="2024-01-01",
    to_date="2024-06-01",
)

print("TYPE OF RESPONSE:", type(resp))
print()
if isinstance(resp, dict):
    print("TOP LEVEL KEYS:", list(resp.keys()))
    for k, v in resp.items():
        print(f"  key='{k}' -> type={type(v)}", end="")
        if isinstance(v, (list, dict)):
            print(f" len={len(v)}")
        else:
            print()
print()
print("FULL RESPONSE (first 2000 chars):")
print(json.dumps(resp, default=str)[:2000])