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

## Colab CLI provisioning

`autoresearch/colab_runtime.py` replaces the manual notebook steps with the
[Colab CLI](https://pypi.org/project/google-colab-cli/) (checked with 0.7.4).
It uploads the same `colab_server.py` and `workspace.py`, starts the server and a
quick tunnel from a `colab exec` bootstrap, and writes the executor file
atomically with mode 0600. The HTTP protocol, job semantics and limits above are
unchanged. Authenticate the CLI first; `colab usage` must print a balance.

A run owns a pool: one state file and at most one VM per compute tier. Enable
it per project (a project `config.yaml` replaces `tiers` as a whole):

```yaml
exec:
  backend: colab
  colab:
    auto_provision: true
    selection: llm          # or fixed: every job on `gpu`
    gpu: T4                 # default tier (fixed mode: a fallback list such as [L4, T4])
    tiers:
      cpu:  {rate: 0.08, use: "no GPU: numpy/sklearn, tiny models"}
      T4:   {rate: 1.19, memory_gb: 15, use: "small models, fp16"}
      L4:   {rate: 1.71, memory_gb: 22, use: "mid-size models, bf16"}
      A100: {rate: 5.40, memory_gb: 40, idle_stop_minutes: 10, use: "large models"}
    max_sessions: 2
    max_replicas: 2
    max_upgrade_ratio: 1.5
    escalate_on_oom: true
    idle_stop_minutes: 30
    max_compute_units: 30
    min_balance: 5
```

### Model-selected compute

With `selection: llm`, the draft, debug, improve, hyperparameter and ablation
prompts include a "Compute selection" section. It lists each tier's memory, rate
and intended use, plus the remaining run budget. The code model declares its
choice on the first line of the script:

```python
# COMPUTE: L4  # ResNet-50 at batch 256 exceeds a T4's 15 GB
```

The declaration is a comment and changes nothing when run locally. Routing:

| Situation | Tier used |
|---|---|
| Declared tier on the menu | That tier's VM, created on first use |
| No or unknown declaration | The node's previous tier (metric parsing, plotting), else `gpu` |
| Seed evaluations | The parent's code, so the parent's declaration |
| Tier refused by Colab | Pricier tiers within `max_upgrade_ratio` of its rate (T4 → L4 at 1.44×), then cheaper GPU tiers; never cpu for a GPU request |
| No tier can be allocated | Retried every 60 s within `remote_wait_minutes`, then `NoTierAvailable` stops the run |
| Tier's VM overloaded | Least-loaded VM of the tier with room. If every one is overloaded, another VM of the same tier starts (see "Scale-out") |
| CUDA out of memory | Re-run once from the same uploaded snapshot. If the VM was shared with other jobs, alone on the same tier (an idle or new VM); otherwise on the next larger tier. A second OOM goes to the model as a normal failure |

Each job's output starts with a line such as
`[compute] Requested L4 (reason); ran on L4 (NVIDIA L4, 23034 MiB) in 312s, ~0.148 compute units.`
The feedback and debug prompts therefore see what the choice cost. Every job is
also recorded in the run's `compute.jsonl`: declared tier, reason, tier used,
GPU name, time, estimated units, escalation and fallback notes.

### Scale-out

The model chooses a tier; the harness may run that choice on more than one VM.
A VM is overloaded when all of its job slots (`max_concurrent`, default
`agent.num_workers`) are running or queued, or when its GPU memory is at least
85% used while a job runs. When every VM of the requested tier is overloaded, the
job goes to a new replica (`T4.2`, `T4.3`, ...). This happens only if all of the
following hold:

- fewer than `max_replicas` VMs of that tier are up;
- `max_sessions` leaves room, or another tier's VM is idle and can be stopped;
- the remaining budget covers at least 30 minutes at the tier's rate;
- the executor is not a standalone `up` executor file.

Otherwise the job waits on the least-loaded VM, and its `[compute]` line says
why. Placement is serialized per tier across workers, so simultaneous jobs
start at most one extra VM. Replicas scale in through the normal per-VM idle
stop. `compute.jsonl` records the replica and whether the VM was shared.

Memory is sampled at placement time. A job that has just started may not have
allocated yet, so a concurrent placement can still meet contention; the shared-GPU
OOM re-run covers that case. With `num_workers: 1` a VM never has a second job,
and scale-out never triggers.

The tier `rate` values order the menu and inform the model. After provisioning,
the rate the account actually reports is stored as `measured_rate` and shown
instead. The default GPU rates come from a third-party measurement
([mccormickml.com](http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/),
March 2026), not from Google pricing. Tier `use` text is guidance for the model,
not a measured capacity. Hardware differs between tiers, so compare run times
across nodes only when they used the same GPU.

### Cost controls

| Control | Behavior |
|---|---|
| Preflight | Launch fails before any model call if the CLI is missing, unauthenticated or below `min_balance`. |
| Lazy start | No VM exists until the first job for its tier. |
| Accelerator check | Tier names are validated before `colab new` (the CLI maps unknown names to A100). After allocation, `nvidia-smi` must match the tier. Otherwise the VM is stopped and that tier is skipped for 30 minutes. |
| Capacity refusals | A 5xx/"Service Unavailable" allocation is retried once after 20 s, then the tier is skipped for 3 minutes. Other refusals (entitlement, quota) skip it for 30 minutes. |
| Session cap | At `max_sessions` VMs, the priciest idle one is stopped to make room. If all are busy, a job for a new tier waits up to 30 minutes; a scale-out is skipped and the job queues. |
| Replica cap | At most `max_replicas` VMs per tier, and a scale-out needs 30 minutes of budget at that tier's rate. |
| Release after tree search | The launcher stops every VM before plotting, write-up and review, which need no GPU. |
| Keep-alive | GPU VMs whose kernel stayed inactive were reclaimed about 20-25 minutes after the last `colab exec` (three runs, CLI 0.7.4), even while the detached executor ran jobs; a CPU control was not. With the keep-alive, L4 VMs survived 40+ minutes. The watchdog runs a no-op `colab exec` on every live VM every 5 minutes. |
| Idle stop | The watchdog stops a VM with no running or queued jobs, or with an unreachable executor, after its tier's `idle_stop_minutes` (global default otherwise). The next job re-provisions it. |
| Budget | Spend is the larger of the balance drop and usage rate × time, polled every five minutes. At `max_compute_units` every VM stops and later jobs fail with `RemoteExecutorUnavailable`. The balance is account-wide, so other Colab use counts too. |
| Controller exit | The detached watchdog stops and releases every VM when the launcher process no longer exists, including after SIGKILL. |
| Unreachable tunnel | The bootstrap waits for cloudflared to register before reporting the URL. The controller waits 3 minutes, restarts the tunnel once, then stops the VM and reports the tunnel log. |

Recovery: a missing executor file (first job, after an idle stop) triggers
provisioning immediately. An executor unreachable for 60 seconds triggers a
reconnect of that job's tier. If its VM still exists, the running server and
jobs are kept and only the tunnel is replaced. Otherwise a new VM is created and
lost jobs are resubmitted as described under "Jobs and reconnection".
Provisioning is serialized per tier across worker processes with file locks.

Measured on this account with CLI 0.7.4 (without package installation): about
90–105 seconds from request to first result, for both a CPU runtime and a T4
(`Tesla T4, 15360 MiB`, measured 1.07 units/hour). T4 allocations were
sometimes refused with HTTP 503 while L4 was available. For tasks of a few
seconds, provisioning time dominates; a declared cheap tier is only worth it
when the job is long enough, or when its VM is already up.

State, executor credentials and watchdog logs are kept in `.state/colab/`
(gitignored). The run directory receives `colab_runtime.json` (per-tier session,
GPU, provisions, stop reasons, measured rates and the spend estimate) and
`compute.jsonl`. Neither contains a token. The CLI's own history
(`~/.config/colab-cli/history/`) records executed code and output. The token is
therefore uploaded as a file and never printed. Keep
`AI_SCIENTIST_REMOTE_URL`/`TOKEN` unset in this mode; the launcher refuses them.

Standalone use, for example with the manual workflow or `--check` (fixed mode,
one VM):

```bash
python -m autoresearch.colab_runtime up --gpu T4   # writes remote_executor.json
python -m ai_scientist.treesearch.remote_interpreter --check
python -m autoresearch.colab_runtime status        # balance, rate, sessions per tier
python -m autoresearch.colab_runtime down          # stop every managed pool
```

`up` starts an idle/budget watchdog unless `--no-watchdog` is given. `down`
without `--session` stops every managed pool that is still up, including one
that belongs to a running launch.

`colab exec` exits 0 even when remote code raises and when its timeout expires.
Bootstrap success is therefore decided by the printed `AISCI_URL` marker, not by
the exit code. Colab usage-policy caveats from the notebook still apply.
