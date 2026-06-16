# ClickUp REST API v2 — reference

Read this when a request needs an endpoint the `clickup` script doesn't wrap (custom fields, time tracking, checklists, attachments, webhooks), or when debugging an unexpected response.

## Contents
- [Auth](#auth)
- [Hierarchy](#hierarchy)
- [Endpoint map](#endpoint-map)
- [Filtered team tasks — query params](#filtered-team-tasks--query-params)
- [Write payloads](#write-payloads)
- [Gotchas](#gotchas)
- [curl recipes](#curl-recipes)

## Auth

Base URL: `https://api.clickup.com/api/v2`

Header: `Authorization: <token>` — the **raw personal token** (`pk_…`). No `Bearer` prefix (that's only for OAuth access tokens). The wrapper sources `CLICKUP_TOKEN` from the env file; in a shell launched via the wrapper, `$CLICKUP_TOKEN` is set.

Rate limit: 100 requests/min per token on the Free Forever / per-token tier (higher on paid plans). A 429 means back off.

## Hierarchy

```
Workspace (called "team" in the API, id from /team)
└── Space            /team/{team_id}/space
    ├── Folder       /space/{space_id}/folder
    │   └── List     (folder.lists[] or /folder/{folder_id}/list)
    └── List         /space/{space_id}/list   (folderless)
        └── Task     /list/{list_id}/task  |  /team/{team_id}/task (filtered)
            └── Subtask, Comment, Custom field, Time entry, Checklist
```

"Workspace" in the UI == "team" in the API. Tasks always belong to exactly one list.

## Endpoint map

| Need | Method + path | Wrapped? |
|---|---|---|
| Authed user | `GET /user` | `whoami` |
| Workspaces | `GET /team` | `workspaces` |
| Spaces | `GET /team/{team_id}/space?archived=false` | `tree` |
| Folders | `GET /space/{space_id}/folder?archived=false` | `tree` |
| Folderless lists | `GET /space/{space_id}/list?archived=false` | `tree` |
| Lists in folder | `GET /folder/{folder_id}/list` | `tree` |
| Tasks in one list | `GET /list/{list_id}/task` | — (use `tasks --list`) |
| Filtered tasks (workspace-wide) | `GET /team/{team_id}/task` | `tasks` |
| One task | `GET /task/{task_id}` | `get` |
| Task comments | `GET /task/{task_id}/comment` | `comments` |
| Create task | `POST /list/{list_id}/task` | `create` |
| Update task | `PUT /task/{task_id}` | `update` |
| Post comment | `POST /task/{task_id}/comment` | `comment` |
| Delete task | `DELETE /task/{task_id}` | — (intentionally not wrapped) |
| Custom field value | `POST /task/{task_id}/field/{field_id}` | — raw |
| Time tracking | `GET/POST /team/{team_id}/time_entries` | — raw |
| Checklists | `POST /task/{task_id}/checklist` | — raw |
| Attachments | `POST /task/{task_id}/attachment` (multipart) | — raw |
| Webhooks | `GET/POST /team/{team_id}/webhook` | — raw |

## Filtered team tasks — query params

`GET /team/{team_id}/task` — the workhorse search. All array params repeat the key:
`?assignees[]=123&assignees[]=456`.

- **Pagination**: `page` (0-based; 100 tasks/page). No total count returned — a short page (<100) is the last one. The wrapper loops pages up to `--limit`.
- **Ordering**: `order_by` ∈ `id | created | updated | due_date`; `reverse=true`.
- **Containers**: `space_ids[]`, `project_ids[]` (folders), `list_ids[]`.
- **Properties**: `statuses[]` (spaces as `%20`, e.g. `in%20progress`), `assignees[]` (user ids), `tags[]`, `custom_items[]` (task types), `subtasks=true`, `include_closed=true`.
- **Dates** (epoch ms): `due_date_gt`/`due_date_lt`, `date_created_gt`/`_lt`, `date_updated_gt`/`_lt`, `date_done_gt`/`_lt`.
- **Other**: `parent=<task_id>` (subtasks of one parent), `custom_fields` (JSON), `include_markdown_description=true`.

No native full-text search here — the wrapper's `--name-contains` filters client-side after fetch.

## Write payloads

**Create** `POST /list/{list_id}/task`:
```json
{
  "name": "Title",                     // required
  "description": "plain text",         // or "markdown_content" for markdown
  "assignees": [123, 456],             // user ids
  "status": "in progress",             // must exist in the list
  "priority": 1,                       // 1 urgent, 2 high, 3 normal, 4 low, null clears
  "due_date": 1781481600000,           // epoch ms
  "due_date_time": true,               // false = date only (ignore time)
  "tags": ["urgent"],
  "parent": "subtask_parent_id"        // makes this a subtask
}
```

**Update** `PUT /task/{task_id}` — only include fields you're changing. Assignees use add/rem, not replace:
```json
{
  "status": "complete",
  "priority": 2,
  "due_date": 1781481600000,
  "assignees": { "add": [123], "rem": [456] }
}
```

**Comment** `POST /task/{task_id}/comment`:
```json
{ "comment_text": "…", "assignee": 123, "notify_all": false }
```

## Gotchas

- **Statuses are per-list.** `status` on create/update must match a status defined in that task's list (case-insensitive). There's no global status list — read one task in the target list (`get`) to see valid names.
- **Dates are epoch milliseconds**, not seconds. `1781481600000`, not `1781481600`.
- **`team_id` = workspace id.** Don't confuse with space/folder/list ids.
- **Custom task ids** (e.g. `GH-123`) require `custom_task_ids=true&team_id=<id>` on `GET /task/{id}` — the wrapper's `get --custom-task-id --team` handles this.
- **`priority`** is `1`–`4` on write but returns as an object (`{"priority":"urgent","id":"1"}`) on read.
- **Closed tasks** are excluded unless `include_closed=true`.

## curl recipes

```bash
# Workspaces
curl -s https://api.clickup.com/api/v2/team -H "Authorization: $CLICKUP_TOKEN"

# Tasks due before a date, assigned to user 123, in two statuses
curl -s "https://api.clickup.com/api/v2/team/$TEAM/task?assignees[]=123&statuses[]=open&statuses[]=in%20progress&due_date_lt=1781481600000" \
  -H "Authorization: $CLICKUP_TOKEN"

# Set a custom field value
curl -s -X POST "https://api.clickup.com/api/v2/task/$TASK/field/$FIELD" \
  -H "Authorization: $CLICKUP_TOKEN" -H "Content-Type: application/json" \
  -d '{"value": "anything"}'

# Start a running time entry
curl -s -X POST "https://api.clickup.com/api/v2/team/$TEAM/time_entries/start" \
  -H "Authorization: $CLICKUP_TOKEN" -H "Content-Type: application/json" \
  -d '{"tid": "'"$TASK"'"}'
```

Full reference: https://developer.clickup.com/reference
