# Fireflies GraphQL API reference

Endpoint: `https://api.fireflies.ai/graphql`
Auth: `Authorization: Bearer $FIREFLIES_API_KEY`

The `fireflies.py` script in this skill covers the common cases. Use this file when you need a custom query that the script doesn't expose.

## Calling the API raw

The `fireflies` wrapper loads `FIREFLIES_API_KEY` from `~/.config/fireflies/.env`. For raw shell access, use the `exec` escape hatch:

```bash
fireflies exec bash -c 'curl -s -X POST https://api.fireflies.ai/graphql \
  -H "Authorization: Bearer $FIREFLIES_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"query { ... }\"}"'
```

## Query: `transcripts` (list)

Arguments (all optional):
- `fromDate: String` — ISO 8601 (UTC), inclusive lower bound on meeting start
- `toDate: String` — ISO 8601 (UTC), inclusive upper bound on meeting start
- `title: String` — substring match on title (server-side, case-insensitive)
- `participant_email: String` — single email; meeting must have this participant
- `host_email: String` — single email; meeting host
- `organizer_email: String` — single email; calendar organizer
- `limit: Int` — max 50 per request; paginate with `skip: Int`
- `mine: Boolean` — only meetings owned by the API key's user

Caveat: `participant_email` accepts only one email. For OR across multiple emails, run multiple queries and dedupe by `id`.

## Query: `transcript` (single)

Argument: `id: String!`

Returns null if the id doesn't exist.

## Query: `active_meetings` (live / in-progress)

No required arguments. Returns meetings currently in progress. Backed by the `fireflies live` subcommand.

Arguments (all optional):
- `email: String` — filter to a specific user's active meetings (admin keys only; regular keys see only their own regardless)
- `state: MeetingState` — `active` or `paused`

`ActiveMeeting` fields (all the type exposes):

```graphql
active_meetings {
  id title organizer_email meeting_link
  start_time   # ISO UTC
  end_time     # ISO UTC — SCHEDULED end, not actual
  state        # MeetingState enum: active | paused
  privacy      # MeetingPrivacy enum, e.g. "link"
}
```

Notes:
- Metadata only. There is no transcript/summary/participant/sentence data while a meeting is live — those fields don't exist on `ActiveMeeting`. Use `transcript`/`transcripts` after the meeting ends.
- Permission scope: a regular API key returns only the key owner's active meetings; admin keys can pass `email` to see any team member's.
- The companion mutation "Add to Live Meeting" tells the Fireflies bot to join an active meeting (not implemented in this skill).

## Useful fields

```graphql
transcript {
  id title dateString date duration
  host_email organizer_email participants
  meeting_link meeting_attendees { displayName email phoneNumber name location }
  summary {
    overview short_summary bullet_gist
    action_items keywords topics_discussed shorthand_bullet
    outline meeting_type
  }
  sentences { index speaker_name speaker_id text start_time end_time }
  analytics { sentiments { negative_pct positive_pct neutral_pct } }
}
```

Notes:
- `date` is unix millis; `dateString` is ISO UTC.
- `duration` is in minutes (float).
- `sentences[].start_time` / `end_time` are seconds from start of recording.
- `summary.action_items` is a single string with markdown-style sections per speaker.
- Live/in-progress meetings do NOT appear in `transcripts` — only processed recordings. Use the `active_meetings` query (above) to list what's happening live.

## Pagination

```graphql
query { transcripts(limit: 50, skip: 50, fromDate: "...") { id } }
```

The API has no cursor — use `skip` offsets.
