-- name: cleanup :exec
PRAGMA optimize;

PRAGMA journal_mode = WAL;

PRAGMA synchronous = normal;

PRAGMA auto_vacuum = INCREMENTAL;

PRAGMA mmap_size = 30000000000;

PRAGMA page_size = 32768;