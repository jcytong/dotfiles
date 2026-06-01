# n8n Cloud REST API Reference

Base URL: `$N8N_BASE_URL` (e.g. `https://myinstance.app.n8n.cloud/api/v1`)
Auth header: `X-N8N-API-KEY: $N8N_API_KEY`

The `n8n.py` script covers common operations. Use raw curl via the `exec` escape hatch for anything not exposed:

```bash
n8n-api exec bash -c 'curl -s "$N8N_BASE_URL/workflows" \
  -H "X-N8N-API-KEY: $N8N_API_KEY" \
  -H "Accept: application/json"'
```

## Pagination

All list endpoints return a `{ data: [...], nextCursor: "..." }` envelope. Pass `cursor=<nextCursor>` to fetch the next page. When `nextCursor` is absent (or null), you've reached the end.

## Workflows

| Method | Path | Description |
|--------|------|-------------|
| GET | `/workflows` | List workflows |
| GET | `/workflows/{id}` | Get workflow |
| POST | `/workflows` | Create workflow |
| PATCH | `/workflows/{id}` | Update workflow |
| DELETE | `/workflows/{id}` | Delete workflow |
| POST | `/workflows/{id}/activate` | Activate workflow |
| POST | `/workflows/{id}/deactivate` | Deactivate workflow |

**List query params:**
- `active`: `true` or `false`
- `tags`: comma-separated tag names
- `limit`: default 100, max 250
- `cursor`: pagination cursor

**Workflow object key fields:**
```json
{
  "id": "abc123",
  "name": "My Workflow",
  "active": true,
  "createdAt": "2024-01-01T00:00:00.000Z",
  "updatedAt": "2024-01-01T00:00:00.000Z",
  "tags": [{ "id": "1", "name": "production" }],
  "nodes": [...],
  "connections": {...},
  "settings": { "executionOrder": "v1" }
}
```

## Executions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/executions` | List executions |
| GET | `/executions/{id}` | Get execution |
| DELETE | `/executions/{id}` | Delete execution |

**List query params:**
- `workflowId`: filter by workflow
- `status`: `error`, `success`, `waiting`, `running`
- `includeData`: `true` to include full node data (large payload)
- `limit`: default 20, max 250
- `cursor`: pagination cursor

**Execution object key fields:**
```json
{
  "id": 42,
  "finished": true,
  "mode": "trigger",
  "retryOf": null,
  "retrySuccessId": null,
  "startedAt": "2024-01-01T12:00:00.000Z",
  "stoppedAt": "2024-01-01T12:00:05.123Z",
  "workflowId": "abc123",
  "waitTill": null,
  "status": "success",
  "data": { ... }
}
```

Note: `data` is only present when `includeData=true`. It can be very large for workflows that process big payloads.

## Credentials

| Method | Path | Description |
|--------|------|-------------|
| GET | `/credentials` | List credentials |
| POST | `/credentials` | Create credential |
| DELETE | `/credentials/{id}` | Delete credential |
| GET | `/credentials/schema/{type}` | Get schema for a credential type |

**Credential object (list view — secrets redacted):**
```json
{
  "id": "cred1",
  "name": "My Slack Credential",
  "type": "slackApi",
  "createdAt": "...",
  "updatedAt": "..."
}
```

## Tags

| Method | Path | Description |
|--------|------|-------------|
| GET | `/tags` | List tags |
| POST | `/tags` | Create tag |
| GET | `/tags/{id}` | Get tag |
| PATCH | `/tags/{id}` | Update tag |
| DELETE | `/tags/{id}` | Delete tag |

## Variables

| Method | Path | Description |
|--------|------|-------------|
| GET | `/variables` | List variables |
| POST | `/variables` | Create variable |
| DELETE | `/variables/{id}` | Delete variable |

Variable object: `{ "id": "var1", "key": "MY_VAR", "value": "abc", "type": "string" }`

## Users (admin only)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/users` | List users |
| GET | `/users/{id}` | Get user |
| DELETE | `/users/{id}` | Delete user |
| PATCH | `/users/{id}/role` | Change user role |

## Source Control

```bash
POST /source-control/pull
```

Pulls latest changes from the connected Git branch. Requires source control to be configured in n8n settings.

## Common error shapes

```json
{ "message": "Workflow not found", "code": 404 }
{ "message": "Unauthorized", "code": 401 }
```
