"""Register the Bridge MCP server with the local Claude Science daemon.

Inserts (or updates) a row in ``operon-cli.db`` so Claude Science's UI lists
'bridge' under Customize -> MCP Connectors. Idempotent — safe to run repeatedly.

Assumes the MCP server is (or will be) reachable at ``http://127.0.0.1:8765/mcp/``.
Run ``uv run python mcp_server.py --http 8765`` in a separate terminal, or use
``scripts/start_mcp_http.sh``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

CS_HOME = Path.home() / ".claude-science"
ACTIVE = CS_HOME / "active-org.json"
NAME = "bridge"
URL = "http://127.0.0.1:8765/mcp/"
DESCRIPTION = (
    "Bridge — drug/condition prediction. Tools for searching ingredients, "
    "reading Frank's condition features, invoking the 5-agent scoring swarm, "
    "and reading workflow KPIs."
)


def main() -> int:
    if not ACTIVE.is_file():
        print("Claude Science not initialized — run `claude-science serve` first")
        return 1

    active = json.loads(ACTIVE.read_text())
    org_uuid = active["org_uuid"]
    account_uuid = active["account_uuid"]

    db_path = CS_HOME / "orgs" / org_uuid / "operon-cli.db"
    if not db_path.is_file():
        print(f"DB not found: {db_path}")
        return 1

    now_ms = int(time.time() * 1000)
    con = sqlite3.connect(str(db_path))
    try:
        existing = con.execute(
            "SELECT id FROM custom_mcp_servers WHERE name = ? AND user_id = ?",
            (NAME, account_uuid),
        ).fetchone()
        if existing:
            con.execute(
                """
                UPDATE custom_mcp_servers
                SET url = ?, transport = ?, description = ?, updated_at = ?, source = 'custom'
                WHERE id = ?
                """,
                (URL, "streamable_http", DESCRIPTION, now_ms, existing[0]),
            )
            print(f"updated existing 'bridge' server (id={existing[0]})")
        else:
            new_id = str(uuid.uuid4())
            con.execute(
                """
                INSERT INTO custom_mcp_servers
                  (id, user_id, name, description, url, transport, created_at, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'custom')
                """,
                (
                    new_id,
                    account_uuid,
                    NAME,
                    DESCRIPTION,
                    URL,
                    "streamable_http",
                    now_ms,
                    now_ms,
                ),
            )
            print(f"registered 'bridge' MCP (id={new_id})")
        con.commit()
    finally:
        con.close()

    print(f"URL: {URL}")
    print("Next: start the HTTP MCP server:")
    print("  uv run python mcp_server.py --http 8765")
    print("Then reload Claude Science in the browser and enable 'bridge' under Customize -> MCP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
