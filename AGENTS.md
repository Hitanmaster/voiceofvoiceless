# Agent Instructions: Notion Chat Logging & Sync

## Purpose
This agent instruction guides logging conversation summaries, key decisions, debugging steps, and chat logs directly into your **AGY Brain** Notion Database using the `notion-mcp-server` MCP tools.

---

## Target Notion Database Details

- **Database Name**: `AGY Brain`
- **Database ID**: `3d50a18a-1962-808a-b9aa-e32846ba6c84`
- **Direct Link**: `https://app.notion.com/p/hiten-master/3d50a18a1962808ab9aae32846ba6c84`

---

## Notion MCP Server Configuration

When interacting with Notion, use `call_mcp_tool` with:
- **`ServerName`**: `"notion-mcp-server"`
- **Available Core Tools**:
  - `API-post-search`: Search for existing pages, databases, or workspaces.
  - `API-retrieve-a-database`: Retrieve schema and property definitions of a database.
  - `API-post-page`: Create a new entry/page inside the Notion database.
  - `API-patch-page`: Update properties or archive an existing page.
  - `API-patch-block-children`: Append markdown or block content to an existing page.

---

## Creating a Chat Log Entry

Use `API-post-page` to add a new record to **AGY Brain**:

```json
{
  "ServerName": "notion-mcp-server",
  "ToolName": "API-post-page",
  "Arguments": {
    "parent": {
      "database_id": "3d50a18a-1962-808a-b9aa-e32846ba6c84"
    },
    "properties": {
      "Name": {
        "title": [
          {
            "text": {
              "content": "[Category] Summary Title of the Conversation"
            }
          }
        ]
      }
    },
    "children": [
      {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
          "rich_text": [
            {
              "type": "text",
              "text": { "content": "Overview & Decisions" }
            }
          ]
        }
      },
      {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
          "rich_text": [
            {
              "type": "text",
              "text": {
                "content": "Detailed summary of the conversation turn, user requests, debugging steps taken, and code modifications."
              }
            }
          ]
        }
      }
    ]
  }
}
```

---

## Best Practices
1. **Automatic Sync**: After major tasks, feature milestones, or debugging sessions, write a concise entry detailing problem diagnosis and solution to `3d50a18a-1962-808a-b9aa-e32846ba6c84`.
2. **Format Blocks Cleanly**: Use Notion markdown blocks (`paragraph`, `code`, `bulleted_list_item`, `callout`) for structured readability.
