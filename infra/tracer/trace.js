"use strict";

const fs = require("fs");
const path = require("path");
const http = require("http");
const Module = require("module");
const { AsyncLocalStorage } = require("async_hooks");

const TRACE_DIR = process.env.DERAIL_TRACE_DIR || "/data/_trace";
const CURSOR_FILE = process.env.DERAIL_CURSOR_FILE || "/data/_derail/cursor.json";
const APP_NAME = process.env.DATABASE_APP_NAME || process.env.DERAIL_APP_NAME || path.basename(process.cwd());
const MAX_PARAM_CHARS = parseInt(process.env.DERAIL_TRACE_MAX_PARAM_CHARS || "200", 10);
const MAX_ROWS = parseInt(process.env.DERAIL_TRACE_MAX_ROWS || "50", 10);
const MAX_ROW_CHARS = parseInt(process.env.DERAIL_TRACE_MAX_ROW_CHARS || "8000", 10);
const requestContext = new AsyncLocalStorage();

let warned = false;
function warnOnce(err) {
  if (!warned) {
    warned = true;
    process.stderr.write("[derail-trace] disabled after error: " + (err && err.stack || err) + "\n");
  }
}

let cursorCache = { mtimeMs: -1, value: -1 };
function currentActionIndex() {
  try {
    const stat = fs.statSync(CURSOR_FILE);
    if (stat.mtimeMs !== cursorCache.mtimeMs) {
      const parsed = JSON.parse(fs.readFileSync(CURSOR_FILE, "utf8"));
      cursorCache = { mtimeMs: stat.mtimeMs, value: Number.isInteger(parsed.action_index) ? parsed.action_index : -1 };
    }
    return cursorCache.value;
  } catch (err) {
    return -1;
  }
}

let stream = null;
function write(record) {
  try {
    if (stream === null) {
      fs.mkdirSync(TRACE_DIR, { recursive: true });
      stream = fs.createWriteStream(path.join(TRACE_DIR, APP_NAME + ".jsonl"), { flags: "a" });
    }
    stream.write(JSON.stringify(record) + "\n");
  } catch (err) {
    warnOnce(err);
  }
}

function truncateParams(params) {
  try {
    const text = JSON.stringify(params);
    if (text === undefined) return null;
    return text.length > MAX_PARAM_CHARS ? text.slice(0, MAX_PARAM_CHARS) + "…" : JSON.parse(text);
  } catch (err) {
    return null;
  }
}

function capturedRows(method, result) {
  try {
    let rows = null;
    if (method === "all" && Array.isArray(result)) rows = result.slice(0, MAX_ROWS);
    else if (method === "get" && result && typeof result === "object") rows = [result];
    if (rows === null) return null;
    const text = JSON.stringify(rows);
    if (text.length > MAX_ROW_CHARS) return { truncated: true, sample: JSON.parse(JSON.stringify(rows.slice(0, 3))) };
    return rows;
  } catch (err) {
    return null;
  }
}

function rowsReturned(result) {
  if (Array.isArray(result)) return result.length;
  if (result === undefined || result === null) return 0;
  if (typeof result === "object" && "changes" in result) return 0;
  return 1;
}

const originalEmit = http.Server.prototype.emit;
http.Server.prototype.emit = function (event, req, res) {
  if (event === "request" && req && typeof req.url === "string") {
    const ctx = { route: req.url.split("?")[0], query: req.url.split("?")[1] || "", method: req.method,
                  started: Date.now() / 1000, action_index: currentActionIndex() };
    const args = arguments;
    return requestContext.run(ctx, () => originalEmit.apply(this, args));
  }
  return originalEmit.apply(this, arguments);
};

function wrapDatabase(Database) {
  if (Database.__derailWrapped) return Database;
  const originalPrepare = Database.prototype.prepare;
  Database.prototype.prepare = function (sql) {
    const statement = originalPrepare.apply(this, arguments);
    const dbName = this.name ? path.basename(String(this.name), ".sqlite") : APP_NAME;
    for (const method of ["all", "get", "run", "iterate", "pluck", "raw"]) {
      const original = statement[method];
      if (typeof original !== "function" || method === "pluck" || method === "raw") continue;
      statement[method] = function () {
        const result = original.apply(this, arguments);
        try {
          const ctx = requestContext.getStore();
          write({
            ts: Date.now() / 1000,
            action_index: ctx ? ctx.action_index : currentActionIndex(),
            route: ctx ? ctx.route : null,
            method: ctx ? ctx.method : null,
            query: ctx ? ctx.query : null,
            db: dbName,
            sql: String(sql),
            rows_returned: method === "iterate" ? null : rowsReturned(result),
            rows: capturedRows(method, result),
            params: truncateParams(Array.from(arguments)),
          });
        } catch (err) {
          warnOnce(err);
        }
        return result;
      };
    }
    return statement;
  };
  Database.__derailWrapped = true;
  return Database;
}

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  const exported = originalLoad.apply(this, arguments);
  if (request === "better-sqlite3") {
    try {
      return wrapDatabase(exported);
    } catch (err) {
      warnOnce(err);
    }
  }
  return exported;
};
