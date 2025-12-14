# .serena/memories Usage Guide

This guide provides best practices for reading from and writing to the .serena/memories system, which stores project context and historical decisions.

## What is .serena/memories?

.serena/memories is a persistent knowledge store that maintains project context across conversations. It enables:

1. **Context Continuity**: Preserving decisions and patterns across sessions
2. **Knowledge Sharing**: Communicating project information between different Claude instances
3. **Historical Record**: Tracking evolution of project architecture and decisions
4. **Efficiency**: Avoiding re-discovery of existing patterns and solutions

## .serena/memories Workflow

### Phase 1: Discovery (Always First)

**Before any planning, ALWAYS check .serena/memories:**

```markdown
Step 1: List Memories
Tool: mcp__serena-mcp__list_memories
Purpose: See what context is available
```

This returns memory filenames with descriptions, helping you identify relevant context.

**Example Output**:
```
Memories:

```

### Phase 2: Context Loading

**Read relevant memories based on the current task:**

```markdown
Step 2: Load Context
Tools:
- mcp__serena-mcp__read_memory
  memory_file_name: "architecture-decisions.md"
- mcp__serena-mcp__read_memory
  memory_file_name: "api-conventions.md"
Purpose: Understand existing patterns
```

**When to Read Each Memory**:

| Task Type | Memories to Read |
|-----------|------------------|
| API development | architecture-decisions, api-conventions, database-schema |
| Frontend work | architecture-decisions, component-patterns, styling-guide |
| Planning new feature | Related previous plans, architecture-decisions |
| Database changes | database-schema, api-conventions |
| Bug fixes | Known issues, debugging-notes |

### Phase 3: Reference CLAUDE.md

**After loading memories, check project-specific guidelines:**

```markdown
Step 3: Load Project Guidelines
Tool: mcp__serena-mcp__read_file
File: CLAUDE.md (if exists)
Purpose: Project-specific standards and workflows
```

CLAUDE.md typically contains:
- Repository structure overview
- Development workflows
- Tool preferences
- Testing requirements
- Documentation standards
- Git practices

### Phase 4: Documentation

**After creating the plan, save it to .serena/memories:**

```markdown
Step 8: Document Plan
Tool: mcp__serena-mcp__write_memory
Memory name: "plan-[feature-name]-[date].md"
Purpose: Preserve plan for future reference
```

## Memory Naming Conventions

### Structured Naming

Use consistent, descriptive names that clearly indicate content:

**Pattern**: `[category]-[specific-topic]-[date-if-temporal].md`

**Examples**:
- ✅ `plan-user-authentication-2025-10-31.md`
- ✅ `architecture-decisions.md`
- ✅ `api-conventions.md`
- ✅ `database-schema-v2.md`
- ❌ `notes.md` (too generic)
- ❌ `temp.md` (unclear purpose)
- ❌ `stuff.md` (non-descriptive)

### Categories

Common memory categories:

1. **plans-**: Execution plans for features
   - `plan-dark-mode-2025-10-31.md`
   - `plan-payment-integration-2025-11-01.md`

2. **architecture-**: Architectural decisions and patterns
   - `architecture-decisions.md`
   - `architecture-frontend-structure.md`
   - `architecture-api-design.md`

3. **conventions-**: Coding and design conventions
   - `conventions-api.md`
   - `conventions-naming.md`
   - `conventions-testing.md`

4. **schema-**: Database and data structures
   - `schema-database.md`
   - `schema-api-models.md`

5. **issues-**: Known problems and debugging notes
   - `issues-known-bugs.md`
   - `issues-performance.md`

6. **decisions-**: Important project decisions
   - `decisions-technology-stack.md`
   - `decisions-third-party-services.md`

## Memory Content Structure

### For Plans

```markdown
# Plan: [Feature Name] ([Date])

## Context
[Why this plan was created, what problem it solves]

## User Requirements
1. [Explicit requirement 1]
2. [Explicit requirement 2]
...

## Excluded Items
[Features/assumptions NOT requested by user]

## Key Decisions
- [Decision 1 and rationale]
- [Decision 2 and rationale]
...

## Execution Strategy

### Batch 1 (Parallel Execution)
[Tasks...]

### Batch 2 (After Batch 1)
[Tasks...]

## Subagent Assignments
- [Subagent 1]: [Tasks]
- [Subagent 2]: [Tasks]

## Tool Requirements
- [Critical tools mandated]
- [Rationale for tool choices]

## Implementation Notes
[Any important details for execution]

## Outcome
[If plan was executed: what happened, any deviations, lessons learned]
```

### For Architecture Decisions

```markdown
# Architecture: [Topic]

## Overview
[High-level description of architectural area]

## Key Decisions

### Decision 1: [Title]
**Context**: [Why this decision was needed]
**Decision**: [What was decided]
**Rationale**: [Why this choice was made]
**Alternatives Considered**: [Other options and why they were rejected]
**Implications**: [Impact on codebase]

### Decision 2: [Title]
...

## Patterns to Follow
[Standard patterns developers should use]

## Anti-Patterns to Avoid
[Common mistakes to prevent]

## Related Memories
[Links to other relevant memories]
```

### For Conventions

```markdown
# Conventions: [Topic]

## General Rules
[High-level guidelines]

## Specific Conventions

### [Category 1]
✅ DO:
- [Good practice 1]
- [Good practice 2]

❌ DON'T:
- [Anti-pattern 1]
- [Anti-pattern 2]

### [Category 2]
...

## Examples

### Good Example
```[language]
[code example following conventions]
```

### Bad Example
```[language]
[code example violating conventions]
```

## Enforcement
[How conventions are checked - linters, code review, etc.]
```

### For Schema

```markdown
# Schema: [Database/API]

## Last Updated
[Date]

## Overview
[High-level structure description]

## Tables / Models

### [Table/Model 1]
**Purpose**: [What this represents]

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY | Unique identifier |
| name | VARCHAR(255) | NOT NULL | User's name |
| ... | ... | ... | ... |

**Relationships**:
- [Related table/model and relationship type]

**Indexes**:
- [Index definitions]

### [Table/Model 2]
...

## Relationships Diagram
[Text-based diagram or description of relationships]

## Migration Notes
[Important notes about schema evolution]
```

## Best Practices

### 1. Always Read Before Writing

```markdown
✅ Good Practice:
1. mcp__serena-mcp__list_memories
2. mcp__serena-mcp__read_memory (relevant memories)
3. Create plan incorporating context
4. mcp__serena-mcp__write_memory (save plan)

❌ Bad Practice:
1. Create plan without checking memories
2. mcp__serena-mcp__write_memory
(Missing context, may conflict with existing patterns)
```

### 2. Keep Memories Focused

Each memory should have a single, clear purpose:

```markdown
✅ Good:
- api-conventions.md (only API standards)
- database-schema.md (only schema)
- plan-auth-2025-10-31.md (only auth plan)

❌ Bad:
- everything.md (mixed topics, hard to find info)
- notes.md (unclear scope)
```

### 3. Update, Don't Duplicate

When information changes, update existing memories:

```markdown
✅ Good:
- Read database-schema.md
- Update with new tables
- Write updated version

❌ Bad:
- Write database-schema-new.md
- Now have conflicting versions
```

### 4. Reference CLAUDE.md Standards

Always align memory format with CLAUDE.md guidelines:

```markdown
Step 1: mcp__serena-mcp__read_file
  file: "CLAUDE.md"
  Purpose: Understand project documentation standards

Step 2: mcp__serena-mcp__write_memory
  Formatted according to CLAUDE.md guidelines
```

### 5. Include Dates for Temporal Content

Plans and time-sensitive decisions should include dates:

```markdown
✅ Good:
- plan-user-profile-2025-10-31.md
- decisions-infrastructure-2025-10-15.md

❌ Bad:
- plan-user-profile.md (when was this?)
- decisions-infrastructure.md (how old is this?)
```

### 6. Link Related Memories

Create connections between related information:

```markdown
# Plan: Payment Integration (2025-10-31)

## Related Memories
- architecture-decisions.md (See "Payment Processing" section)
- api-conventions.md (Follow REST API patterns)
- database-schema.md (Users and Transactions tables)

This helps future readers understand relationships.
```

## Memory Reading Strategies

### Strategy 1: Targeted Reading

Read only memories relevant to current task:

```markdown
Task: Add new API endpoint

Relevant Memories:
✅ Read: api-conventions.md
✅ Read: architecture-decisions.md (API section)
❌ Skip: frontend-component-patterns.md (not relevant)
❌ Skip: styling-guide.md (not relevant)
```

### Strategy 2: Comprehensive Context Loading

For major features, load all related context:

```markdown
Task: Build entire user profile feature

Load Everything Related:
✅ architecture-decisions.md
✅ api-conventions.md
✅ database-schema.md
✅ component-patterns.md
✅ styling-guide.md
✅ testing-standards.md
```

### Strategy 3: Progressive Loading

Start with high-level, drill down as needed:

```markdown
1. Read: architecture-decisions.md (understand overall patterns)
2. Read: api-conventions.md (specific API details if needed)
3. Read: plan-similar-feature-*.md (if similar work was done before)
```

## Memory Writing Strategies

### Strategy 1: Plan Documentation

After creating any plan:

```markdown
Memory Name: plan-[feature-name]-[YYYY-MM-DD].md

Content:
- Full plan document
- Context and requirements
- Execution strategy
- Key decisions
- Tool requirements
- Expected outcomes
```

### Strategy 2: Architecture Updates

When making architectural decisions:

```markdown
Option A: Update existing memory
- If adding to existing architectural area
- Read current architecture-decisions.md
- Add new decision section
- Write updated version

Option B: Create new memory
- If entirely new architectural area
- Create architecture-[specific-area].md
- Reference from main architecture-decisions.md
```

### Strategy 3: Convention Documentation

When establishing new patterns:

```markdown
1. Check if convention category exists
2. Update existing conventions memory OR create new one
3. Include examples (good and bad)
4. Explain rationale
5. Reference in CLAUDE.md if appropriate
```

## Common Pitfalls

### Pitfall 1: Skipping Memory Check

```markdown
❌ Problem:
User: "Add logout button"
Claude: [Creates plan without checking memories]
Result: Plan conflicts with existing auth patterns

✅ Solution:
1. mcp__serena-mcp__list_memories
2. mcp__serena-mcp__read_memory ("architecture-decisions.md")
3. Create plan aligned with existing auth pattern
```

### Pitfall 2: Forgetting to Document

```markdown
❌ Problem:
[Creates excellent plan]
[Executes plan]
[Never writes to memory]
Result: Knowledge lost for future tasks

✅ Solution:
Always include memory write in planning workflow:
Step 8: mcp__serena-mcp__write_memory
```

### Pitfall 3: Generic Memory Names

```markdown
❌ Problem:
Memory: "notes.md"
Later: Which notes? About what?

✅ Solution:
Memory: "api-endpoint-conventions.md"
Clear, specific, discoverable
```

### Pitfall 4: Stale Information

```markdown
❌ Problem:
Database schema changes
Memory not updated
Future work uses outdated schema

✅ Solution:
After schema changes:
1. mcp__serena-mcp__read_memory ("database-schema.md")
2. Update with changes
3. mcp__serena-mcp__write_memory (updated version)
```

## Integration with Planning Process

### Planning Workflow with .serena/memories

```markdown
## Step 1: Gather Context from .serena/memories

1. mcp__serena-mcp__list_memories
   → See available memories

2. mcp__serena-mcp__read_memory [for each relevant memory]
   → Load project context

3. mcp__serena-mcp__read_file ("CLAUDE.md")
   → Load project guidelines

Purpose: Ensure plan aligns with existing patterns and decisions

## Step 2-7: [Planning steps]

## Step 8: Update .serena/memories

1. Reference CLAUDE.md for documentation format

2. mcp__serena-mcp__write_memory
   memory_name: "plan-[feature]-[date].md"
   content: [Complete plan + decisions + rationale]

Purpose: Document plan for future reference
```

## .serena/memories Checklist

Use this checklist for every planning task:

```markdown
Pre-Planning (Context Gathering):
- [ ] Called mcp__serena-mcp__list_memories
- [ ] Identified relevant memories
- [ ] Read all relevant memories
- [ ] Read CLAUDE.md if it exists
- [ ] Incorporated memory context into plan

Post-Planning (Documentation):
- [ ] Created comprehensive plan document
- [ ] Referenced CLAUDE.md for format standards
- [ ] Chose descriptive memory name
- [ ] Called mcp__serena-mcp__write_memory
- [ ] Included date in memory name (if temporal)
- [ ] Linked related memories
```

## Example .serena/memories Interaction

### Scenario: Planning User Profile Feature

```markdown
# Step 1: Discover Available Context

Tool: mcp__serena-mcp__list_memories

Output:
- architecture-decisions.md
- api-conventions.md
- database-schema.md
- component-patterns.md
- plan-user-authentication-2025-10-15.md

# Step 2: Load Relevant Context

Tool: mcp__serena-mcp__read_memory
Memories to read:
1. architecture-decisions.md
   → Learn: Project uses React + Express + PostgreSQL
   → Learn: File uploads go through S3
   → Learn: Auth uses JWT tokens

2. api-conventions.md
   → Learn: REST endpoints follow /api/v1/resource pattern
   → Learn: Use camelCase for JSON fields
   → Learn: PUT for updates, PATCH for partial updates

3. database-schema.md
   → Learn: Users table already exists
   → Learn: User ID is UUID type
   → Learn: Existing relationships and constraints

4. plan-user-authentication-2025-10-15.md
   → Learn: Auth system already implemented
   → Learn: Can leverage existing AuthContext
   → Learn: Login/logout patterns established

# Step 3: Reference Project Guidelines

Tool: mcp__serena-mcp__read_file
File: CLAUDE.md

Learn:
- Documentation format standards
- Testing requirements
- Git workflow preferences

# Step 4-7: Create Plan
[Incorporating all context from memories]

# Step 8: Document Plan

Tool: mcp__serena-mcp__write_memory

Memory name: plan-user-profile-2025-10-31.md

Content:
```markdown
# Plan: User Profile Feature (2025-10-31)

## Context
User requested: "Implement user profile page with avatar upload, bio editing, and settings"

## Related Memories
- architecture-decisions.md: Using React + Express + PostgreSQL, S3 for uploads
- api-conventions.md: Following REST patterns, camelCase JSON
- database-schema.md: User table exists with UUID IDs
- plan-user-authentication-2025-10-15.md: Auth system in place

## Key Decisions
1. Leverage existing auth system (don't rebuild)
2. Follow established API conventions
3. Use S3 for avatar uploads (consistent with architecture)
4. Extend existing User table (don't create new profile table)

## User Requirements
[...]

## Execution Strategy
[...]
```

---

By following these .serena/memories practices, plans will be context-aware, consistent with project patterns, and well-documented for future reference.
