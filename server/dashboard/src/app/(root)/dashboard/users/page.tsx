"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import { api } from "@/utils/api";
import { USER_ENDPOINTS } from "@/utils/api-endpoints";
import { toast } from "@/components/ui/use-toast";
import { Plus } from "lucide-react";
import { format } from "date-fns";
import { getErrorMessage } from "@/lib/error-message";
import { useApiQuery } from "@/hooks/use-api-query";
import { useAuth } from "@/hooks/use-auth";
import { AccountUser } from "@/types/api";

const MIN_PASSWORD_LENGTH = 8;

export default function UsersPage() {
  const { isAdmin } = useAuth();
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");

  const {
    data: users = [],
    isLoading,
    refetch,
  } = useApiQuery<AccountUser[]>(
    async () => {
      if (!isAdmin) return [];
      const res = await api.get<AccountUser[]>(USER_ENDPOINTS.BASE);
      return res.data ?? [];
    },
    { errorToast: "Failed to load users", initialData: [] },
  );

  const canCreate =
    newName.trim().length > 0 &&
    newEmail.includes("@") &&
    newPassword.length >= MIN_PASSWORD_LENGTH;

  const handleDialogClose = (open: boolean) => {
    if (!open) {
      setNewName("");
      setNewEmail("");
      setNewPassword("");
    }
    setCreateOpen(open);
  };

  const handleCreate = async () => {
    try {
      await api.post(USER_ENDPOINTS.BASE, {
        name: newName.trim(),
        email: newEmail,
        password: newPassword,
      });
      toast({ title: "User created", variant: "success" });
      handleDialogClose(false);
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to create user",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const columns = [
    { key: "name" as keyof AccountUser, label: "Name", width: 160 },
    { key: "email" as keyof AccountUser, label: "Email", width: 220 },
    { key: "role" as keyof AccountUser, label: "Role", width: 80 },
    {
      key: "created_at" as keyof AccountUser,
      label: "Created",
      width: 120,
      render: (value: string) => format(new Date(value), "MMM d, yyyy"),
    },
  ];

  if (!isAdmin) {
    return (
      <EmptyState
        title="Admin access required"
        description="Only the administrator can manage user accounts."
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold font-fustat">Users</h1>
        <Dialog open={createOpen} onOpenChange={handleDialogClose}>
          <DialogTrigger asChild>
            <Button size="sm">
              <Plus className="size-4 mr-1" /> Create User
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create User</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 mt-2">
              <div className="space-y-2">
                <Label htmlFor="user-name">Name</Label>
                <Input
                  id="user-name"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Jane Doe"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="user-email">Email</Label>
                <Input
                  id="user-email"
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="jane@example.com"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="user-password">Initial password</Label>
                <Input
                  id="user-password"
                  autoComplete="off"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder={`At least ${MIN_PASSWORD_LENGTH} characters`}
                />
                <p className="text-xs text-onSurface-secondary">
                  Share this initial password with the user; they can change it
                  in Settings after their first login.
                </p>
              </div>
              <Button
                onClick={handleCreate}
                disabled={!canCreate}
                className="w-full"
              >
                Create
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading ? (
        <TableSkeleton rows={3} columns={4} />
      ) : users.length === 0 ? (
        <EmptyState
          title="No users yet"
          description="Create accounts for your team members so each gets a private memory space."
        />
      ) : (
        <Card className="border-memBorder-primary overflow-hidden">
          <DataTable
            data={users}
            columns={columns}
            getRowKey={(row) => row.id}
          />
        </Card>
      )}
    </div>
  );
}
