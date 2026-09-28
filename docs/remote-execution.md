# Remote execution

The controller uploads an experiment workspace, submits a job, polls its state,
and downloads results. The executor uses a fresh directory and subprocess for
each job. It is an execution service, not a sandbox: generated code has the
runtime user's filesystem and network permissions.

## Connection

The notebook prints a JSON object with `url` and `token`. Save it as
`remote_executor.json` at the repository root. Environment overrides are
`AI_SCIENTIST_REMOTE_URL`, `AI_SCIENTIST_REMOTE_TOKEN` and
`AI_SCIENTIST_REMOTE_CONFIG` (an alternate configuration file).

HTTPS is required except for localhost development. Redirects are rejected.
Authentication failures stop immediately. Connection failures, server errors
and capacity responses are retried for `exec.remote_wait_minutes`.
Settings are re-read between requests; partial/invalid JSON fails explicitly,
so replace the credentials file atomically when updating a running controller.

This version uses protocol 2. Rebuild and upload the notebook when updating the
server or workspace-transfer module. Do not mix an older executor with the new
controller and expect retry deduplication.

## Jobs and reconnection

The client assigns a job ID before submission. Retrying the same POST with the
same ID and payload returns the existing job. Reusing an ID with a different
payload returns HTTP 409. This prevents duplicate training after a lost POST
response while the server still retains the job record.

If a restarted runtime has lost the job, the client can resubmit it up to three
times from the last local workspace. This repeats the experiment from that
snapshot; it does not resume an in-memory optimizer or recover files lost with
the runtime. No exactly-once guarantee spans runtime loss.

Polling is bounded by the execution timeout plus the configured queue/wait
budget and a short grace period. Cancellation attempts a DELETE request.
If a disconnected runtime cannot receive cancellation, its own execution timeout
remains the fallback. Stopping the notebook server terminates its active job
process groups.

## Workspace transfer

Both directions allow only regular files and directories. Absolute paths,
parent traversal, links, devices, duplicate entries and existing symlink paths
are rejected. Limits are 128 MiB compressed, 512 MiB expanded and 10,000 entries
per workspace transfer; they are defined in `ai_scientist/remote/workspace.py`.

`exec.remote_max_file_mb` defaults to 100 MiB per file. Larger files are skipped.
Files deleted remotely are removed from the corresponding local snapshot only
after a successful download. Local files that were not uploaded are preserved.

Jobs with skipped result files are retained, and the controller logs their ID.
In Colab, inspect `/content/aisci_jobs/<job-id>/ws/` and retrieve checkpoints
before the runtime ends. Completed jobs expire approximately six hours after
completion. Retention is temporary, not durable storage. Fully transferred jobs
are deleted automatically. Retained jobs count toward the server's 64-job limit;
delete them after retrieval to free capacity.

## HTTP interface

All routes require `Authorization: Bearer <token>`.

| Method and route | Behavior |
|---|---|
| `GET /health` | Protocol version, GPU information and queue counts |
| `POST /jobs` | Submit or retrieve an identical existing submission |
| `GET /jobs/<id>` | Queued, running or done state and execution result |
| `GET /jobs/<id>/workspace` | Download a completed workspace archive |
| `DELETE /jobs/<id>` | Cancel and release a job |

Submission fields: `job_id`, `code`, `timeout`, `agent_file_name`, `env_vars`,
`max_file_mb`, and `workspace_tgz_b64`. Timeout is positive and capped at 24
hours. `agent_file_name` must be a plain filename. Request bodies and decoded
archives are bounded. The control token is removed from the experiment child's
environment; this is not a guarantee that untrusted code cannot access the
rest of its runtime.

`AISCI_MAX_CONCURRENT` controls execution concurrency. Start with one worker
and increase it only after checking GPU memory use. The notebook monitor reports
health and exits if its server or tunnel process stops. It does not defeat
Colab idle limits or extend the runtime's lifetime.
