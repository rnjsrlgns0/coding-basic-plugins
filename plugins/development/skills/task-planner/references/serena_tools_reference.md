# Serena MCP Tools Reference

This guide provides comprehensive information about Serena MCP tools and their usage patterns for task planning.

## Overview

Serena MCP is a Model Context Protocol server that provides semantic coding tools for intelligent codebase operations. These tools enable symbol-level understanding and manipulation of code, making them superior to basic file operations for most development tasks.

**Key Principle**: Always mandate serena tools for codebase operations in task plans. They provide semantic understanding that basic file tools cannot match.

## Core Tools Categories

### 1. Reading Tools
### 2. Writing/Editing Tools
### 3. Search and Discovery Tools
### 4. Memory Management Tools
### 5. Project Management Tools
### 6. Execution Tools

---

## 1. Reading Tools

### mcp__serena-mcp__read_file

**Purpose**: Read file contents or specific line ranges.

**When to Use**:
- Reading entire files
- Reading specific line ranges
- When symbolic tools are not appropriate (e.g., config files)

**Parameters**:
- `relative_path` (required): Path to file
- `start_line` (optional): First line (0-based)
- `end_line` (optional): Last line (0-based)
- `max_answer_chars` (optional): Character limit

**Best Practices**:
- Prefer symbolic tools (`find_symbol`) when working with code symbols
- Use for non-code files (JSON, YAML, text, etc.)
- Use line ranges to read only necessary portions

**Example Usage in Plans**:
```markdown
Task: Read configuration file
Tools Required:
- mcp__serena-mcp__read_file
  Path: config/app.json
  Purpose: Load application settings
```

---

### mcp__serena-mcp__get_symbols_overview

**Purpose**: Get high-level overview of symbols in a file (classes, functions, etc.).

**When to Use**:
- First step when understanding a new file
- Before deciding which symbols to read in detail
- To understand file structure without reading entire content

**Parameters**:
- `relative_path` (required): Path to file
- `max_answer_chars` (optional): Character limit

**Best Practices**:
- Always use this BEFORE reading full symbol bodies
- Reduces token usage by showing structure first
- Helps identify which symbols need detailed reading

**Example Usage in Plans**:
```markdown
Task: Understand UserService structure
Tools Required:
- mcp__serena-mcp__get_symbols_overview
  Path: src/services/UserService.ts
  Purpose: Identify available methods before reading implementation
```

---

### mcp__serena-mcp__find_symbol

**Purpose**: Find and retrieve symbol information (classes, methods, functions, etc.).

**When to Use**:
- Finding specific symbols by name
- Reading symbol bodies
- Understanding symbol structure and children
- Locating symbols across codebase

**Parameters**:
- `name_path` (required): Symbol name or path (e.g., "UserService/login")
- `relative_path` (optional): Restrict search to file/directory
- `depth` (optional): How many levels of children to retrieve
- `include_body` (optional): Include symbol source code
- `substring_matching` (optional): Allow partial name matches
- `include_kinds` (optional): Filter by LSP symbol kinds
- `exclude_kinds` (optional): Exclude specific symbol kinds

**Name Path Patterns**:
- `"method"`: Finds any symbol named "method"
- `"class/method"`: Finds "method" inside "class"
- `"/class/method"`: Finds only top-level "class" containing "method"

**Symbol Kinds** (for filtering):
- 5 = Class
- 6 = Method
- 12 = Function
- 13 = Variable
- 14 = Constant

**Best Practices**:
- Start with `include_body=False` to see structure
- Use `depth=1` to see immediate children (e.g., class methods)
- Add `relative_path` when you know the file location
- Use `substring_matching=True` for fuzzy searches

**Example Usage in Plans**:
```markdown
Task: Locate and read UserService login method
Tools Required:
- mcp__serena-mcp__find_symbol
  name_path: "UserService/login"
  relative_path: "src/services/"
  include_body: true
  Purpose: Read login method implementation
```

---

### mcp__serena-mcp__find_referencing_symbols

**Purpose**: Find all places where a symbol is referenced.

**When to Use**:
- Understanding symbol usage
- Impact analysis before changes
- Finding all call sites
- Refactoring preparation

**Parameters**:
- `name_path` (required): Symbol to find references for
- `relative_path` (required): File containing the symbol
- `include_kinds` (optional): Filter referencing symbols
- `exclude_kinds` (optional): Exclude symbol kinds

**Best Practices**:
- Always use before modifying widely-used symbols
- Check references before renaming or removing code
- Use for understanding data flow

**Example Usage in Plans**:
```markdown
Task: Find all usages of calculateTax function before refactoring
Tools Required:
- mcp__serena-mcp__find_referencing_symbols
  name_path: "calculateTax"
  relative_path: "src/utils/tax.ts"
  Purpose: Ensure all call sites are compatible with changes
```

---

## 2. Writing/Editing Tools

### mcp__serena-mcp__create_text_file

**Purpose**: Create new files.

**When to Use**:
- Creating new source files
- Adding new modules or components
- Writing new configuration files

**Parameters**:
- `relative_path` (required): Path for new file
- `content` (required): File content

**Best Practices**:
- Prefer editing existing files over creating new ones
- Consider if code belongs in existing file first
- Use proper project structure conventions

**Example Usage in Plans**:
```markdown
Task: Create new UserProfile component
Tools Required:
- mcp__serena-mcp__create_text_file
  relative_path: "src/components/UserProfile.tsx"
  Purpose: New React component file
```

---

### mcp__serena-mcp__replace_symbol_body

**Purpose**: Replace entire symbol body (function, method, class, etc.).

**When to Use**:
- Completely rewriting a function/method
- Significant logic changes
- When the entire symbol needs updating

**Parameters**:
- `name_path` (required): Symbol to replace
- `relative_path` (required): File containing symbol
- `body` (required): New symbol body

**Best Practices**:
- Use only when replacing entire symbol
- For small changes, use `replace_regex` instead
- Body includes signature but not docstrings/comments above it
- Verify symbol location with `find_symbol` first

**Example Usage in Plans**:
```markdown
Task: Completely rewrite login method logic
Tools Required:
- mcp__serena-mcp__find_symbol (locate method first)
- mcp__serena-mcp__replace_symbol_body
  name_path: "AuthService/login"
  relative_path: "src/services/AuthService.ts"
  Purpose: Replace login implementation
```

---

### mcp__serena-mcp__insert_after_symbol

**Purpose**: Insert content after a symbol definition.

**When to Use**:
- Adding new methods to a class
- Adding new functions to a module
- Inserting code after specific elements

**Parameters**:
- `name_path` (required): Symbol to insert after
- `relative_path` (required): File path
- `body` (required): Content to insert

**Best Practices**:
- Use for adding new class methods
- Use for adding functions at end of file
- Maintains proper indentation automatically

**Example Usage in Plans**:
```markdown
Task: Add new logout method to AuthService class
Tools Required:
- mcp__serena-mcp__insert_after_symbol
  name_path: "AuthService/login"
  relative_path: "src/services/AuthService.ts"
  body: "  async logout() { ... }"
  Purpose: Add logout method after login
```

---

### mcp__serena-mcp__insert_before_symbol

**Purpose**: Insert content before a symbol definition.

**When to Use**:
- Adding imports at file start
- Adding new class methods before others
- Inserting code before specific elements

**Parameters**:
- `name_path` (required): Symbol to insert before
- `relative_path` (required): File path
- `body` (required): Content to insert

**Best Practices**:
- Use for adding imports (insert before first symbol)
- Maintains proper indentation automatically
- Common for adding dependencies

**Example Usage in Plans**:
```markdown
Task: Add import statement at top of file
Tools Required:
- mcp__serena-mcp__insert_before_symbol
  name_path: "UserService"  (first symbol in file)
  relative_path: "src/services/UserService.ts"
  body: "import bcrypt from 'bcrypt';\n"
  Purpose: Add bcrypt import
```

---

### mcp__serena-mcp__replace_regex

**Purpose**: Replace content using regular expressions.

**When to Use**:
- Small changes within symbols
- Multi-line replacements
- Pattern-based replacements
- When symbol tools are too broad

**Parameters**:
- `relative_path` (required): File path
- `regex` (required): Python regex pattern
- `repl` (required): Replacement string
- `allow_multiple_occurrences` (optional): Replace all matches

**Best Practices**:
- Use for changes smaller than full symbols
- Use wildcards (.*?) for flexible matching
- Escape special regex characters
- Keep regex concise with wildcards

**Example Usage in Plans**:
```markdown
Task: Update API endpoint URL in fetch call
Tools Required:
- mcp__serena-mcp__replace_regex
  relative_path: "src/api/users.ts"
  regex: "fetch\\('/api/users'\\)"
  repl: "fetch('/api/v2/users')"
  Purpose: Update API version in URL
```

---

### mcp__serena-mcp__rename_symbol

**Purpose**: Rename symbol throughout entire codebase.

**When to Use**:
- Renaming variables, functions, classes
- Refactoring with name changes
- Ensuring all references are updated

**Parameters**:
- `name_path` (required): Symbol to rename
- `relative_path` (required): File containing symbol
- `new_name` (required): New symbol name

**Best Practices**:
- Use to ensure all references update
- Check references first with `find_referencing_symbols`
- Works across entire codebase

**Example Usage in Plans**:
```markdown
Task: Rename calculatePrice to computePrice throughout codebase
Tools Required:
- mcp__serena-mcp__find_referencing_symbols (check impact first)
- mcp__serena-mcp__rename_symbol
  name_path: "calculatePrice"
  relative_path: "src/utils/pricing.ts"
  new_name: "computePrice"
  Purpose: Rename function and all references
```

---

## 3. Search and Discovery Tools

### mcp__serena-mcp__search_for_pattern

**Purpose**: Search for regex patterns in codebase files.

**When to Use**:
- Finding code patterns
- Searching across multiple files
- Discovering usage patterns
- Text-based code search

**Parameters**:
- `substring_pattern` (required): Regex pattern
- `relative_path` (optional): Restrict to path
- `context_lines_before` (optional): Lines before match
- `context_lines_after` (optional): Lines after match
- `restrict_search_to_code_files` (optional): Only search code files
- `paths_include_glob` (optional): Include file patterns
- `paths_exclude_glob` (optional): Exclude file patterns

**Best Practices**:
- Use for discovering patterns across files
- Combine with glob patterns for targeted search
- Use context lines to understand usage
- Prefer `find_symbol` for known symbols

**Example Usage in Plans**:
```markdown
Task: Find all API endpoint definitions
Tools Required:
- mcp__serena-mcp__search_for_pattern
  substring_pattern: "app\\.(get|post|put|delete)\\("
  relative_path: "src/routes/"
  restrict_search_to_code_files: true
  Purpose: Locate all route definitions
```

---

### mcp__serena-mcp__list_dir

**Purpose**: List files and directories.

**When to Use**:
- Understanding directory structure
- Finding what files exist
- Exploring project layout

**Parameters**:
- `relative_path` (required): Directory to list
- `recursive` (required): Scan subdirectories
- `skip_ignored_files` (optional): Skip .gitignore files

**Best Practices**:
- Use before reading unknown directories
- Set recursive=true for deep scans
- Skip ignored files for cleaner output

**Example Usage in Plans**:
```markdown
Task: Understand src/components structure
Tools Required:
- mcp__serena-mcp__list_dir
  relative_path: "src/components"
  recursive: true
  Purpose: See all component files
```

---

### mcp__serena-mcp__find_file

**Purpose**: Find files by name or pattern.

**When to Use**:
- Searching for specific files
- Finding files by wildcard patterns
- Locating configuration files

**Parameters**:
- `file_mask` (required): Filename or pattern (* and ? wildcards)
- `relative_path` (required): Directory to search

**Best Practices**:
- Use wildcards for flexible matching
- Narrow search with specific relative_path
- Faster than list_dir for known filenames

**Example Usage in Plans**:
```markdown
Task: Find all TypeScript test files
Tools Required:
- mcp__serena-mcp__find_file
  file_mask: "*.test.ts"
  relative_path: "src"
  Purpose: Locate all test files
```

---

## 4. Memory Management Tools

### mcp__serena-mcp__list_memories

**Purpose**: List all available memory files.

**When to Use**:
- Start of planning process
- Checking what context exists
- Before reading memories

**Parameters**: None

**Best Practices**:
- Always call at start of planning
- Review memory names to identify relevant context
- Foundation for context-aware planning

**Example Usage in Plans**:
```markdown
Step 1: Gather Context
Tools Required:
- mcp__serena-mcp__list_memories
  Purpose: See available project memories
```

---

### mcp__serena-mcp__read_memory

**Purpose**: Read content of a memory file.

**When to Use**:
- Loading project context
- Understanding previous decisions
- Checking existing patterns
- Learning project conventions

**Parameters**:
- `memory_file_name` (required): Memory file to read
- `max_answer_chars` (optional): Character limit

**Best Practices**:
- Read relevant memories at plan start
- Use to understand project history
- Check for related previous plans

**Example Usage in Plans**:
```markdown
Step 1: Gather Context
Tools Required:
- mcp__serena-mcp__list_memories
- mcp__serena-mcp__read_memory
  memory_file_name: "architecture-decisions.md"
  Purpose: Understand architectural patterns
```

---

### mcp__serena-mcp__write_memory

**Purpose**: Write or update memory files.

**When to Use**:
- After creating plans
- Documenting decisions
- Recording implementation details
- Storing project knowledge

**Parameters**:
- `memory_name` (required): Memory file name
- `content` (required): Content to write

**Best Practices**:
- Always write plans to memory
- Use descriptive memory names
- Include date in memory name
- Reference CLAUDE.md for format

**Example Usage in Plans**:
```markdown
Step 8: Update Memory-Bank
Tools Required:
- mcp__serena-mcp__write_memory
  memory_name: "plan-user-profile-2025-10-31.md"
  content: [Full plan document]
  Purpose: Document plan for future reference
```

---

### mcp__serena-mcp__delete_memory

**Purpose**: Delete memory files.

**When to Use**:
- Removing outdated memories
- Cleaning up incorrect information
- Only when explicitly requested

**Parameters**:
- `memory_file_name` (required): Memory to delete

**Best Practices**:
- Rarely used - only on explicit user request
- Confirm before deletion
- Consider updating instead of deleting

---

## 5. Project Management Tools

### mcp__serena-mcp__activate_project

**Purpose**: Switch to a different project.

**When to Use**:
- Working with multiple projects
- Switching project contexts

**Parameters**:
- `project` (required): Project name or path

---

### mcp__serena-mcp__get_current_config

**Purpose**: Get current project configuration.

**When to Use**:
- Understanding current project setup
- Checking available tools
- Debugging configuration

**Parameters**: None

---

## 6. Execution Tools

### mcp__serena-mcp__execute_shell_command

**Purpose**: Execute shell commands.

**When to Use**:
- Running build commands
- Database migrations
- Running tests
- Starting services

**Parameters**:
- `command` (required): Shell command
- `cwd` (optional): Working directory
- `capture_stderr` (optional): Capture error output

**Best Practices**:
- Never run long-running processes
- Never run interactive commands
- Use for build, test, migrate commands
- Check memory for recommended commands first

**Example Usage in Plans**:
```markdown
Task: Run database migration
Tools Required:
- mcp__serena-mcp__execute_shell_command
  command: "npm run migrate"
  Purpose: Apply database schema changes
```

---

## Tool Selection Guidelines

### Decision Tree

```
What do you need to do?

├─ Read code by symbol name?
│  └─ → mcp__serena-mcp__find_symbol
│
├─ See file structure without details?
│  └─ → mcp__serena-mcp__get_symbols_overview
│
├─ Replace entire function/method?
│  └─ → mcp__serena-mcp__replace_symbol_body
│
├─ Add new method to class?
│  └─ → mcp__serena-mcp__insert_after_symbol
│
├─ Add import at file start?
│  └─ → mcp__serena-mcp__insert_before_symbol
│
├─ Small change within code?
│  └─ → mcp__serena-mcp__replace_regex
│
├─ Rename across codebase?
│  └─ → mcp__serena-mcp__rename_symbol
│
├─ Find symbol usage?
│  └─ → mcp__serena-mcp__find_referencing_symbols
│
├─ Search for patterns?
│  └─ → mcp__serena-mcp__search_for_pattern
│
├─ Load project context?
│  └─ → mcp__serena-mcp__read_memory
│
└─ Save plan/decisions?
   └─ → mcp__serena-mcp__write_memory
```

## Common Workflows

### Workflow 1: Understanding New Code

```markdown
1. mcp__serena-mcp__list_dir
   - See what files exist

2. mcp__serena-mcp__get_symbols_overview
   - Understand file structure

3. mcp__serena-mcp__find_symbol (with include_body=false, depth=1)
   - See symbol children (e.g., class methods)

4. mcp__serena-mcp__find_symbol (with include_body=true)
   - Read specific symbols needed
```

### Workflow 2: Adding New Feature

```markdown
1. mcp__serena-mcp__list_memories
   - Check for relevant context

2. mcp__serena-mcp__read_memory
   - Load related memories

3. mcp__serena-mcp__find_symbol
   - Locate where to add code

4. mcp__serena-mcp__insert_after_symbol
   - Add new code

5. mcp__serena-mcp__write_memory
   - Document the change
```

### Workflow 3: Refactoring

```markdown
1. mcp__serena-mcp__find_symbol
   - Locate symbol to refactor

2. mcp__serena-mcp__find_referencing_symbols
   - Check all usage sites

3. mcp__serena-mcp__rename_symbol OR mcp__serena-mcp__replace_symbol_body
   - Make the change

4. mcp__serena-mcp__write_memory
   - Document refactoring
```

### Workflow 4: Planning from Scratch

```markdown
1. mcp__serena-mcp__list_memories
   - See available context

2. mcp__serena-mcp__read_memory (for each relevant memory)
   - Load project context

3. mcp__serena-mcp__list_dir OR mcp__serena-mcp__search_for_pattern
   - Understand current codebase state

4. [Create plan]

5. mcp__serena-mcp__write_memory
   - Save plan to memory-bank
```

## Best Practices Summary

### 1. Always Use Serena for Code Operations
- ✅ Use serena tools for all code reading/writing
- ❌ Don't use bash cat, grep, or sed for code files
- ✅ Serena provides semantic understanding

### 2. Progressive Disclosure
- ✅ Start with `get_symbols_overview`
- ✅ Then `find_symbol` with `include_body=false`
- ✅ Finally `find_symbol` with `include_body=true`
- ❌ Don't read entire files unnecessarily

### 3. Memory-First Planning
- ✅ Always call `list_memories` first
- ✅ Read relevant memories before planning
- ✅ Write plans to memory after creation
- ❌ Don't skip memory context

### 4. Right Tool for the Job
- ✅ `replace_symbol_body` for entire symbols
- ✅ `replace_regex` for small changes
- ✅ `insert_after_symbol` for adding methods
- ❌ Don't use the wrong abstraction level

### 5. Mandate in Plans
- ✅ Explicitly list required tools in each task
- ✅ Specify exact tool names
- ✅ Explain why each tool is needed
- ❌ Don't assume tools will be chosen correctly
