"""
Mock vulnerable Flask application for SQLi agent training.

Intentionally vulnerable to SQL injection at multiple difficulty tiers.
Run standalone:  python -m src.environment.vulnerable_app
"""

import sqlite3
import os
import threading
from flask import Flask, request, jsonify, g

DB_PATH = os.environ.get("SQLI_DB_PATH", ":memory:")

# ---------------------------------------------------------------------------
# Vulnerability catalogue — ground truth used by reward function
# ---------------------------------------------------------------------------
VULNERABILITY_CATALOGUE = {
    "login_username": {
        "endpoint": "/api/login",
        "method": "POST",
        "param": "username",
        "type": "error_based",
        "difficulty": 1,
        "severity": "critical",
    },
    "search_query": {
        "endpoint": "/api/products/search",
        "method": "GET",
        "param": "q",
        "type": "union_based",
        "difficulty": 2,
        "severity": "high",
    },
    "product_id": {
        "endpoint": "/api/products/{id}",
        "method": "GET",
        "param": "id",
        "type": "error_based",
        "difficulty": 1,
        "severity": "high",
    },
    "order_filter": {
        "endpoint": "/api/orders",
        "method": "GET",
        "param": "status",
        "type": "blind_boolean",
        "difficulty": 3,
        "severity": "medium",
    },
    "user_profile": {
        "endpoint": "/api/users/{id}/profile",
        "method": "GET",
        "param": "fields",
        "type": "blind_time",
        "difficulty": 4,
        "severity": "medium",
    },
}

KNOWN_VULN_COUNT = len(VULNERABILITY_CATALOGUE)


def create_app(db_path: str = DB_PATH) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path
    app.config["TESTING"] = False

    # SQLite :memory: databases are connection-scoped — every new connection
    # gets a fresh empty DB. For in-memory use we must share ONE persistent
    # connection across all requests; for file-based DBs we init once and
    # open fresh per-request connections normally.
    if db_path == ":memory:":
        _persistent = sqlite3.connect(db_path, check_same_thread=False)
        _persistent.row_factory = sqlite3.Row
        _init_db_on_conn(_persistent)
        app.config["PERSISTENT_CONN"] = _persistent
    else:
        _init_db(db_path)
        app.config["PERSISTENT_CONN"] = None

    # -----------------------------------------------------------------------
    # Request/response logger — prints payload + response status to stdout
    # -----------------------------------------------------------------------

    import logging
    app.logger.setLevel(logging.DEBUG)

    @app.before_request
    def _log_request():
        body = {}
        params = {}
        try:
            if request.is_json:
                body = request.get_json(silent=True) or {}
            params = dict(request.args)
        except Exception:
            pass
        parts = [f"  → {request.method} {request.path}"]
        if params:
            parts.append(f"    QUERY_PARAMS : {params}")
        if body:
            parts.append(f"    REQUEST_BODY : {body}")
        print("\n".join(parts), flush=True)

    @app.after_request
    def _log_response(response):
        print(f"    RESPONSE_STATUS: {response.status_code}", flush=True)
        return response

    # -----------------------------------------------------------------------
    # DB helpers
    # -----------------------------------------------------------------------

    def get_db():
        persistent = app.config.get("PERSISTENT_CONN")
        if persistent is not None:
            return persistent
        if not hasattr(g, "_sqli_db"):
            g._sqli_db = sqlite3.connect(app.config["DB_PATH"])
            g._sqli_db.row_factory = sqlite3.Row
        return g._sqli_db

    @app.teardown_appcontext
    def close_db(error):
        if app.config.get("PERSISTENT_CONN") is not None:
            return
        db = getattr(g, "_sqli_db", None)
        if db is not None:
            db.close()

    # -----------------------------------------------------------------------
    # Vulnerable endpoints
    # -----------------------------------------------------------------------

    @app.route("/api/login", methods=["POST"])
    def login():
        """
        VULN-1 (error_based, difficulty=1):
        Username is injected directly into query without parameterization.
        ' OR '1'='1  → dumps all users
        """
        data = request.get_json(silent=True) or {}
        username = data.get("username", "")
        password = data.get("password", "")
        db = get_db()
        try:
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            rows = db.execute(query).fetchall()
            if rows:
                return jsonify({"status": "ok", "user": dict(rows[0])})
            return jsonify({"status": "fail", "message": "Invalid credentials"}), 401
        except sqlite3.OperationalError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/products/search", methods=["GET"])
    def product_search():
        """
        VULN-2 (union_based, difficulty=2):
        q parameter injectable.  ' UNION SELECT name,sql,null FROM sqlite_master--
        """
        q = request.args.get("q", "")
        db = get_db()
        try:
            query = f"SELECT id, name, price FROM products WHERE name LIKE '%{q}%'"
            rows = db.execute(query).fetchall()
            return jsonify({"results": [dict(r) for r in rows]})
        except sqlite3.OperationalError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/products/<product_id>", methods=["GET"])
    def product_detail(product_id):
        """
        VULN-3 (error_based, difficulty=1):
        Path parameter concatenated directly.
        /api/products/1' → triggers error
        """
        db = get_db()
        try:
            query = f"SELECT * FROM products WHERE id={product_id}"
            row = db.execute(query).fetchone()
            if row:
                return jsonify(dict(row))
            return jsonify({"message": "Not found"}), 404
        except sqlite3.OperationalError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/orders", methods=["GET"])
    def list_orders():
        """
        VULN-4 (blind_boolean, difficulty=3):
        status parameter injectable but no error messages returned.
        Boolean-based: status=pending' AND '1'='1  vs  AND '1'='2
        """
        status = request.args.get("status", "pending")
        db = get_db()
        try:
            query = f"SELECT id, product_id, quantity FROM orders WHERE status='{status}'"
            rows = db.execute(query).fetchall()
            return jsonify({"count": len(rows), "orders": [dict(r) for r in rows]})
        except sqlite3.OperationalError:
            return jsonify({"count": 0, "orders": []}), 200

    @app.route("/api/users/<user_id>/profile", methods=["GET"])
    def user_profile(user_id):
        """
        VULN-5 (blind_time, difficulty=4):
        fields parameter injectable.  SQLite doesn't support SLEEP() but
        we simulate it with a heavy recursive CTE for timing analysis.
        fields=username,(SELECT+CASE+WHEN+(1=1)+THEN+randomblob(100000000)+ELSE+1+END)
        """
        fields = request.args.get("fields", "username,email")
        db = get_db()
        try:
            query = f"SELECT {fields} FROM users WHERE id={user_id}"
            row = db.execute(query).fetchone()
            if row:
                return jsonify(dict(row))
            return jsonify({"message": "Not found"}), 404
        except sqlite3.OperationalError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    # -----------------------------------------------------------------------
    # Safe endpoints (agent should NOT report these as vulnerable)
    # -----------------------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "version": "1.0.0"})

    @app.route("/api/products", methods=["GET"])
    def list_products():
        """Safe — uses parameterized query."""
        db = get_db()
        rows = db.execute("SELECT id, name, price FROM products").fetchall()
        return jsonify({"results": [dict(r) for r in rows]})

    @app.route("/api/info", methods=["GET"])
    def app_info():
        """Safe — no user input reflected to DB."""
        return jsonify(
            {
                "app": "MockShop",
                "endpoints": [
                    "/api/login",
                    "/api/products/search",
                    "/api/products",
                    "/api/products/<id>",
                    "/api/orders",
                    "/api/users/<id>/profile",
                    "/api/health",
                ],
            }
        )

    return app


# ---------------------------------------------------------------------------
# DB initializer
# ---------------------------------------------------------------------------

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    email    TEXT,
    role     TEXT DEFAULT 'user'
);
CREATE TABLE IF NOT EXISTS products (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER DEFAULT 100
);
CREATE TABLE IF NOT EXISTS orders (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER,
    product_id INTEGER,
    quantity   INTEGER,
    status     TEXT DEFAULT 'pending'
);
INSERT OR IGNORE INTO users VALUES (1,'admin','sup3rS3cr3t!','admin@shop.local','admin');
INSERT OR IGNORE INTO users VALUES (2,'alice','alice123','alice@shop.local','user');
INSERT OR IGNORE INTO users VALUES (3,'bob','b0bpass','bob@shop.local','user');
INSERT OR IGNORE INTO products VALUES (1,'Laptop',999.99,50);
INSERT OR IGNORE INTO products VALUES (2,'Phone',499.99,200);
INSERT OR IGNORE INTO products VALUES (3,'Tablet',299.99,150);
INSERT OR IGNORE INTO orders VALUES (1,2,1,1,'pending');
INSERT OR IGNORE INTO orders VALUES (2,3,2,2,'shipped');
INSERT OR IGNORE INTO orders VALUES (3,2,3,1,'delivered');
"""

_db_lock = threading.Lock()
_initialized_paths: set = set()


def _init_db_on_conn(conn: sqlite3.Connection):
    """Initialize schema on an already-open connection (used for :memory: DBs)."""
    c = conn.cursor()
    c.executescript(_DB_SCHEMA)
    conn.commit()


def _init_db(db_path: str):
    """Initialize schema for a file-based DB (opens/closes its own connection)."""
    with _db_lock:
        if db_path in _initialized_paths:
            return
        conn = sqlite3.connect(db_path)
        _init_db_on_conn(conn)
        conn.close()
        _initialized_paths.add(db_path)


# ---------------------------------------------------------------------------
# Entry point for standalone use
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app = create_app(db_path="mock_shop.db")
    print(f"[VulnerableApp] Listening on http://127.0.0.1:{port}")
    print(f"[VulnerableApp] {KNOWN_VULN_COUNT} known injection points in catalogue")
    app.run(host="0.0.0.0", port=port, debug=False)
