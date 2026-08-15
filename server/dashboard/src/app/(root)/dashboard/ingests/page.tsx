"use client";

import { useState } from "react";
import { format, formatDistanceToNow } from "date-fns";
import { RefreshCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import { api } from "@/utils/api";
import { INGEST_ENDPOINTS } from "@/utils/api-endpoints";
import { useApiQuery } from "@/hooks/use-api-query";

type ApiMemoryIngest = {
  id: string;
  status: "pending" | "processing" | "failed";
  attempts: number;
  scope: string;
  project_id: string | null;
  last_error: string | null;
  request_id: string | null;
  created_at: string;
  next_attempt_at: string;
};

type MemoryIngest = {
  id: string;
  status: "pending" | "processing" | "failed";
  attempts: number;
  scope: string;
  projectId: string | null;
  lastError: string | null;
  createdAt: string;
  nextAttemptAt: string;
};

const PAGE_SIZE = 20;

const STATUS_CLASSES: Record<MemoryIngest["status"], string> = {
  pending:
    "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/40 dark:text-amber-300",
  processing:
    "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-900/40 dark:bg-sky-950/40 dark:text-sky-300",
  failed:
    "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/40 dark:bg-rose-950/40 dark:text-rose-300",
};

const normalizeIngest = (entry: ApiMemoryIngest): MemoryIngest => {
  return {
    id: entry.id,
    status: entry.status,
    attempts: entry.attempts,
    scope: entry.scope,
    projectId: entry.project_id,
    lastError: entry.last_error,
    createdAt: entry.created_at,
    nextAttemptAt: entry.next_attempt_at,
  };
};

export default function IngestsPage() {
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);

  const {
    data: ingests = [],
    isLoading,
    error,
    refetch,
  } = useApiQuery<MemoryIngest[]>(
    async () => {
      const res = await api.get<ApiMemoryIngest[]>(INGEST_ENDPOINTS.BASE);
      setLastUpdated(new Date().toISOString());
      return (res.data ?? []).map(normalizeIngest);
    },
    { errorToast: "Failed to load memory ingests", initialData: [] },
  );

  const retry = async (id: string) => {
    setRetryingId(id);
    setRetryError(null);
    try {
      await api.post(INGEST_ENDPOINTS.RETRY(id));
      await refetch();
    } catch {
      setRetryError("Retry failed — the ingest may already be gone.");
    } finally {
      setRetryingId(null);
    }
  };

  const pendingCount = ingests.filter((row) => row.status !== "failed").length;
  const failedCount = ingests.filter((row) => row.status === "failed").length;

  const columns = [
    {
      key: "createdAt" as keyof MemoryIngest,
      label: "Created",
      width: 140,
      render: (value: string) => (
        <span title={format(new Date(value), "PPpp")}>
          {formatDistanceToNow(new Date(value), { addSuffix: true })}
        </span>
      ),
    },
    {
      key: "scope" as keyof MemoryIngest,
      label: "Scope",
      width: 220,
      render: (value: string, row: MemoryIngest) => (
        <span className="font-mono text-xs break-all text-onSurface-default-primary">
          {row.projectId ? `${value}: ${row.projectId}` : value}
        </span>
      ),
    },
    {
      key: "status" as keyof MemoryIngest,
      label: "Status",
      width: 120,
      render: (value: MemoryIngest["status"]) => (
        <Badge variant="outline" className={STATUS_CLASSES[value]}>
          {value}
        </Badge>
      ),
    },
    {
      key: "attempts" as keyof MemoryIngest,
      label: "Attempts",
      width: 90,
    },
    {
      key: "lastError" as keyof MemoryIngest,
      label: "Error",
      width: 180,
      render: (value: string | null) => (
        <span className="font-mono text-xs">{value ?? "--"}</span>
      ),
    },
    {
      key: "id" as keyof MemoryIngest,
      label: "",
      width: 110,
      render: (value: string, row: MemoryIngest) =>
        row.status === "failed" ? (
          <Button
            variant="outline"
            size="sm"
            disabled={retryingId !== null}
            onClick={() => void retry(value)}
          >
            <RotateCcw className="size-3.5 mr-1.5" />
            Retry
          </Button>
        ) : null,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold font-fustat">Ingests</h1>
          <p className="text-sm text-onSurface-default-secondary">
            Deferred memory writes waiting for background processing. Processed
            ingests are deleted; failed ones stay retryable for 30 days.
          </p>
          {lastUpdated && (
            <p className="text-xs text-onSurface-default-tertiary">
              Last updated{" "}
              {formatDistanceToNow(new Date(lastUpdated), { addSuffix: true })}
            </p>
          )}
        </div>
        <Button
          variant="outline"
          onClick={() => {
            setPage(0);
            void refetch();
          }}
          disabled={isLoading}
        >
          <RefreshCw className="size-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[
          { label: "Queued", value: pendingCount },
          { label: "Failed", value: failedCount },
        ].map((card) => (
          <Card key={card.label} className="border-memBorder-primary">
            <CardContent className="p-5">
              <p className="text-xs text-onSurface-default-tertiary">
                {card.label}
              </p>
              <p className="mt-1 text-2xl font-semibold">{card.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {(error || retryError) && (
        <Card className="border-memBorder-primary">
          <CardContent className="p-4 text-sm text-onSurface-danger-primary">
            {error ?? retryError}
          </CardContent>
        </Card>
      )}

      {isLoading ? (
        <TableSkeleton rows={6} columns={6} />
      ) : ingests.length === 0 ? (
        <EmptyState
          title="No queued or failed memory ingests"
          description="Deferred writes appear here until they are processed."
          image="requests"
        />
      ) : (
        <>
          <Card className="border-memBorder-primary overflow-hidden">
            <DataTable
              data={ingests.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)}
              columns={columns}
              getRowKey={(row) => row.id}
            />
          </Card>
          {ingests.length > PAGE_SIZE && (
            <div className="flex items-center justify-between text-sm text-onSurface-default-tertiary">
              <span>
                {page * PAGE_SIZE + 1}–
                {Math.min((page + 1) * PAGE_SIZE, ingests.length)} of{" "}
                {ingests.length}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={(page + 1) * PAGE_SIZE >= ingests.length}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
