#!/usr/bin/env python3
"""Refill a pgvector collection with vectors from a different embedding model.

Fork-only runner script, not a migration: it moves data, so it never runs
automatically and defaults to a dry run.

Switching the embedding model invalidates every stored vector, because
distances are only comparable between vectors from the same model. Mixing them
silently degrades retrieval, so the switch is blue/green: build a second
collection with the new model, verify it, then point POSTGRES_COLLECTION_NAME
at it. The old collection stays untouched and is the rollback.

Memory ids and payloads are copied verbatim, so ids, timestamps, scope keys
and all metadata survive; only the vector is recomputed. get_memory,
update_memory, delete_memory and import provenance keep working, and the
history database is not involved.

Writes go through mem0's own PGVector store rather than hand-written SQL, so
the collection is created with the same schema and index the server expects.

Docker only, like the rest of server/:

    # measure first, write nothing
    docker compose exec mem0 python reembed_collection.py \
        --target-collection memories_gemma --model embeddinggemma --dims 768

    # then write
    docker compose exec mem0 python reembed_collection.py \
        --target-collection memories_gemma --model embeddinggemma --dims 768 --apply

Re-running with --apply skips ids already present in the target, so a second
pass picks up memories written while the first pass ran.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg
from mem0.vector_stores.pgvector import PGVector
from ollama import Client
from psycopg import sql


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-collection",
        default=os.environ.get("POSTGRES_COLLECTION_NAME", "memories"),
        help="Collection to read from (default: POSTGRES_COLLECTION_NAME).",
    )
    parser.add_argument("--target-collection", required=True, help="Collection to write into. Must not be the source.")
    parser.add_argument("--model", required=True, help="Ollama embedding model for the new vectors.")
    parser.add_argument("--dims", type=int, required=True, help="Vector dimension the new model emits.")
    parser.add_argument("--batch-size", type=int, default=16, help="Texts per Ollama call (default 16).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N memories. 0 means all.")
    parser.add_argument("--apply", action="store_true", help="Write. Without it nothing is modified.")
    return parser.parse_args()


def _pg_settings() -> dict[str, object]:
    return {
        "host": os.environ.get("POSTGRES_HOST", "postgres"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "postgres"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


def _source_rows(table: str, limit: int) -> list[tuple[str, dict]]:
    """Read id and payload only — the old vectors are of no use here."""
    query = sql.SQL("SELECT id, payload FROM {} ORDER BY id").format(sql.Identifier(table))
    if limit:
        query = sql.SQL("{} LIMIT {}").format(query, sql.Literal(limit))
    with psycopg.connect(**_pg_settings()) as conn, conn.cursor() as cur:
        cur.execute(query)
        return [(str(row[0]), row[1] or {}) for row in cur.fetchall()]


def _existing_ids(table: str) -> set[str]:
    """Ids already in the target, so --apply can be re-run for a delta pass."""
    with psycopg.connect(**_pg_settings()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (table,),
        )
        if not cur.fetchone()[0]:
            return set()
        cur.execute(sql.SQL("SELECT id FROM {}").format(sql.Identifier(table)))
        return {str(row[0]) for row in cur.fetchall()}


def _target_store(collection: str, dims: int) -> PGVector:
    return PGVector(
        collection_name=collection,
        embedding_model_dims=dims,
        diskann=False,
        hnsw=True,
        **_pg_settings(),
    )


def _verify_dims(client: Client, model: str, expected: int) -> None:
    """Embed one probe string so a wrong --dims fails before any write."""
    vectors = client.embed(model=model, input="dimension probe").get("embeddings") or []
    if not vectors:
        raise SystemExit(f"model '{model}' returned no embedding")
    if len(vectors[0]) != expected:
        raise SystemExit(f"model '{model}' emits {len(vectors[0])} dimensions, --dims says {expected}")


def main() -> int:
    args = _parse_args()
    if args.target_collection == args.source_collection:
        raise SystemExit("target collection must differ from the source collection")

    client = Client(host=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"))
    _verify_dims(client, args.model, args.dims)

    rows = _source_rows(args.source_collection, args.limit)
    empty = [memory_id for memory_id, payload in rows if not str(payload.get("data", "")).strip()]
    work = [(memory_id, payload) for memory_id, payload in rows if str(payload.get("data", "")).strip()]

    store = None
    already: set[str] = set()
    if args.apply:
        store = _target_store(args.target_collection, args.dims)
        already = _existing_ids(args.target_collection)
        work = [item for item in work if item[0] not in already]

    print(f"source   : {args.source_collection} ({len(rows)} memories)")
    print(f"target   : {args.target_collection} (already present: {len(already)})")
    print(f"model    : {args.model} @ {args.dims} dims")
    print(f"to embed : {len(work)}")
    if empty:
        print(f"skipped  : {len(empty)} with empty payload text")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")

    started = time.monotonic()
    done = 0
    for offset in range(0, len(work), args.batch_size):
        batch = work[offset : offset + args.batch_size]
        vectors = client.embed(model=args.model, input=[str(payload["data"]) for _, payload in batch]).get(
            "embeddings"
        ) or []
        if len(vectors) != len(batch):
            raise SystemExit(f"model returned {len(vectors)} embeddings for {len(batch)} texts")

        if store is not None:
            store.insert(
                vectors=[list(vector) for vector in vectors],
                payloads=[payload for _, payload in batch],
                ids=[memory_id for memory_id, _ in batch],
            )
        done += len(batch)
        print(f"  {done}/{len(work)} — {(time.monotonic() - started) / done:.2f}s per memory", flush=True)

    total = time.monotonic() - started
    if work:
        verb = "wrote" if args.apply else "embedded (discarded)"
        print(f"\n{verb} {done} in {total:.1f}s ({total / done:.2f}s per memory)")
    if args.apply:
        print(
            f"\nNext: set POSTGRES_COLLECTION_NAME={args.target_collection}, "
            f"MEM0_DEFAULT_EMBEDDER_MODEL={args.model}, MEM0_EMBEDDING_DIMS={args.dims} and restart the API.\n"
            f"Rollback is pointing POSTGRES_COLLECTION_NAME back at {args.source_collection}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
