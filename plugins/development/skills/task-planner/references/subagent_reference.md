# Subagent Reference Guide

This guide provides detailed information about all available subagents, their capabilities, and when to use them for task planning.

## Available Subagents

### 0. task-router (Primary Handler)

**Purpose**: Lightweight request analyzer and task delegator. This agent should be the FIRST choice for ambiguous or multi-faceted requests.

**Capabilities**:
- Request analysis and classification
- Task delegation to specialized agents
- Simple file reads (single file)
- Quick pattern searches
- Task planning with TodoWrite
- Clarification questions

**Tools (Minimal Set - 6 only)**:
- Task (for delegation)
- Read (simple file reads)
- Glob (file pattern matching)
- Grep (quick searches)
- TodoWrite (task planning)
- AskUserQuestion (clarification)

**When to Use**:
- As the DEFAULT first choice for general requests
- When task type is not immediately clear
- For multi-part requests requiring multiple agents
- When you need to analyze before acting

**When NOT to Use**:
- Request clearly matches a specific agent
- User explicitly requests a specialized agent
- Task is obviously single-domain (e.g., "fix this React bug")

**Routing Rules**:
| Request Pattern | Routes To |
|-----------------|-----------|
| API, endpoint, server, database | backend-api-developer |
| Component, UI, style, frontend | frontend-ui-developer |
| Design, wireframe, UX, layout | ui-ux-designer |
| Test, validation, QA, performance | web-app-tester |
| Bug, error, debug, fix | code-debugger |
| Documentation, README, guide | technical-documentation-writer |
| Structure, find, search (complex) | Explore |
| Plan, architecture, design | Plan |

**Example Tasks**:
- "Help me with the login feature" → Analyzes, then routes to appropriate agents
- "Read this config file" → Handles directly
- "Build a user management system" → Creates plan, delegates to multiple agents

---

### 1. backend-api-developer

**Purpose**: Server-side development, API creation, database operations, and backend logic.

**Capabilities**:
- Creating and modifying REST APIs
- Database schema design and migrations
- Authentication and authorization implementation
- Business logic implementation
- Database query optimization
- Server-side validation
- API documentation

**When to Use**:
- Creating new API endpoints
- Modifying existing backend routes
- Database schema changes
- Authentication/authorization features
- Server-side data processing
- Background job implementation
- API integration with third-party services

**Tools Available**:
- Full serena MCP toolset for codebase operations
- Bash for running servers and databases
- Read/Write/Edit for file operations
- Context7 for library documentation

**Example Tasks**:
- "Create a POST /api/users endpoint for user registration"
- "Add password hashing to the authentication service"
- "Design a database schema for an e-commerce platform"
- "Implement rate limiting middleware"
- "Optimize slow database queries"

---

### 2. frontend-ui-developer

**Purpose**: Frontend development, UI component creation, styling, and client-side logic.

**Capabilities**:
- React/Vue/Angular component implementation
- Responsive layout creation
- CSS/Tailwind/styled-components styling
- Client-side state management (Redux, Context API, etc.)
- Form handling and validation
- Frontend routing
- Integration with backend APIs
- Accessibility implementation

**When to Use**:
- Creating new UI components
- Implementing responsive designs
- Styling existing components
- Frontend state management
- Client-side form validation
- Translating wireframes to code
- Implementing frontend routing
- API integration on the client side

**Tools Available**:
- Full serena MCP toolset
- Bash for running dev servers and builds
- Read/Write/Edit for file operations
- Context7 for framework documentation

**Example Tasks**:
- "Create a responsive navigation bar component"
- "Implement a user profile page with React"
- "Add dark mode toggle with state persistence"
- "Style the checkout form using Tailwind CSS"
- "Integrate the user API with the frontend dashboard"

---

### 3. ui-ux-designer

**Purpose**: Interface design, user experience analysis, design system creation, and layout planning.

**Capabilities**:
- Interface analysis and evaluation
- User flow design
- Wireframe creation
- Design system recommendations
- Accessibility audits
- Layout restructuring
- Visual hierarchy improvements
- Responsive layout planning
- Screen structure documentation

**When to Use**:
- Before implementing new features (design phase)
- Evaluating existing UI/UX
- Creating wireframes for complex features
- Establishing design systems
- Accessibility reviews
- Layout optimization
- User flow analysis

**Tools Available**:
- Read/Write for documentation
- Web tools for research
- Standard design documentation tools

**Example Tasks**:
- "Design a user flow for the checkout process"
- "Create wireframes for the dashboard"
- "Review the current navigation for accessibility issues"
- "Recommend a design system for the application"
- "Evaluate the visual hierarchy of the homepage"

---

### 4. web-app-tester

**Purpose**: Testing web applications, performance monitoring, and quality assurance.

**Capabilities**:
- Automated test execution
- Performance monitoring and profiling
- UI/UX validation
- Regression testing
- Cross-browser testing
- Responsive design testing
- Accessibility testing
- Load testing

**When to Use**:
- After implementing new features
- Verifying bug fixes
- Performance optimization
- Regression testing before releases
- Validating responsive behavior
- Accessibility compliance checks

**Tools Available**:
- Playwright MCP for browser automation
- Bash for running test suites
- Read/Write for test documentation
- Performance profiling tools

**Example Tasks**:
- "Run tests to verify the new login feature works"
- "Check if the dashboard loads within 2 seconds"
- "Test the checkout flow on mobile devices"
- "Verify all forms are accessible via keyboard"
- "Run regression tests after the API changes"

---

### 5. code-debugger

**Purpose**: Identifying, analyzing, and resolving software bugs.

**Capabilities**:
- Error message and stack trace analysis
- Execution path tracing
- Variable state inspection
- Root cause investigation
- Environment factor consideration
- Code modification for bug fixes
- Prevention strategy recommendations

**When to Use**:
- When bugs or errors are reported
- Investigating failing tests
- Analyzing error logs
- Resolving runtime exceptions
- Debugging performance issues
- Fixing integration problems

**Tools Available**:
- Full serena MCP toolset
- Bash for running debug sessions
- Diagnostic tools
- Read/Write/Edit for code fixes
- Playwright for browser debugging

**Example Tasks**:
- "Fix the 500 error on the /api/users endpoint"
- "Debug why the login form is not submitting"
- "Investigate the memory leak in the data processing service"
- "Resolve the race condition in the state update"
- "Fix the failing unit tests in UserService"

---

### 6. technical-documentation-writer

**Purpose**: Creating and updating technical documentation, tracking progress, and maintaining project records.

**Capabilities**:
- Writing API documentation
- Creating user guides
- Documenting code changes
- Maintaining changelogs
- Progress tracking
- Memory-bank updates
- README creation and updates

**When to Use**:
- After completing features (documentation phase)
- During project planning (documenting decisions)
- When updating APIs or libraries
- For progress tracking
- When creating tutorials or guides

**Tools Available**:
- Serena MCP read tools
- Read/Write/Edit for documentation files
- Memory-bank tools (read_memory, write_memory)

**Example Tasks**:
- "Document the new authentication API endpoints"
- "Update the README with installation instructions"
- "Create a changelog for version 2.0"
- "Document the database schema changes in memory-bank"
- "Write a user guide for the admin dashboard"

---

### 7. Explore (Agent)

**Purpose**: Fast codebase exploration to understand structure, find files, and answer questions about the code.

**Capabilities**:
- Finding files by patterns
- Searching code for keywords
- Answering questions about codebase structure
- Quick pattern matching
- Symbol discovery

**Thoroughness Levels**:
- `quick`: Basic searches, first-match focus
- `medium`: Moderate exploration across common locations
- `very thorough`: Comprehensive analysis across multiple locations

**When to Use**:
- Understanding new codebases
- Finding files or patterns
- Answering "where is X?" questions
- Quick code searches
- Structure exploration

**Example Tasks**:
- "Find all React components in the src directory"
- "Where are API endpoints defined?"
- "What is the project structure?"
- "Find all files that import the UserService"

**Usage**:
```
Specify thoroughness level when calling:
- "quick" for basic searches
- "medium" for moderate exploration
- "very thorough" for comprehensive analysis
```

---

### 8. Plan (Agent)

**Purpose**: Fast codebase exploration for planning purposes (similar to Explore but optimized for planning context).

**Capabilities**:
- Same as Explore agent
- Optimized for gathering planning context

**When to Use**:
- During planning phase
- Before creating task breakdowns
- When you need to understand what exists before planning changes

**Example Tasks**:
- Same as Explore agent

---

### 9. general-purpose (Agent) - DEPRECATED

> **Note**: This agent is being replaced by `task-router`. Use `task-router` for general requests and let it delegate to specialized agents.

**Purpose**: Complex, multi-step tasks that don't fit specific agent categories.

**Capabilities**:
- Researching complex questions
- Multi-step code searching
- File pattern searches
- Complex investigations
- Code analysis across multiple files

**When to Use**:
- PREFER `task-router` instead
- Only use for truly complex research that requires ALL tools
- Deep codebase investigations spanning many files

**Example Tasks**:
- "Research how authentication is implemented across the app"
- "Analyze the data flow from frontend to database"
- "Find all places where user permissions are checked"
- "Investigate how error handling works throughout the codebase"

---

## Subagent Selection Decision Tree

```
Is the task about...

├─ Unclear or multi-faceted request?
│  └─ → task-router (PRIMARY - analyzes and delegates)
│
├─ Writing backend code?
│  └─ → backend-api-developer
│
├─ Writing frontend code?
│  └─ → frontend-ui-developer
│
├─ Designing UI/UX?
│  └─ → ui-ux-designer
│
├─ Testing?
│  └─ → web-app-tester
│
├─ Fixing bugs?
│  └─ → code-debugger
│
├─ Writing documentation?
│  └─ → technical-documentation-writer
│
├─ Understanding codebase structure?
│  └─ → Explore (with appropriate thoroughness)
│
└─ Deep research requiring ALL tools?
   └─ → general-purpose (rare cases only)
```

## Parallel Execution Guidelines

### Tasks That Can Run in Parallel

**Independent UI and Backend Work**:
```
✅ Parallel:
- frontend-ui-developer: Create React component
- backend-api-developer: Create API endpoint
```

**Multiple Backend Features**:
```
✅ Parallel:
- backend-api-developer: Create user registration endpoint
- backend-api-developer: Create password reset endpoint
```

**Design and Development**:
```
✅ Parallel:
- ui-ux-designer: Design checkout flow
- backend-api-developer: Implement payment processing
```

### Tasks That Must Run Sequentially

**Dependent Work**:
```
❌ Cannot Parallel:
1. ui-ux-designer: Create wireframe
2. frontend-ui-developer: Implement wireframe (depends on #1)
```

**Build on Previous Work**:
```
❌ Cannot Parallel:
1. backend-api-developer: Create database schema
2. backend-api-developer: Write migration scripts (depends on #1)
```

**Testing After Implementation**:
```
❌ Cannot Parallel:
1. frontend-ui-developer: Implement feature
2. web-app-tester: Test feature (depends on #1)
```

## Best Practices

### 1. Choose the Most Specific Agent
- Use `task-router` as the primary handler for unclear requests
- For clear single-domain tasks, use specialized agents directly
- Example: Use `backend-api-developer` instead of `task-router` for obvious API work

### 2. One Agent Per Task Type
- Don't mix agent responsibilities in a single subtask
- Example: Split "Design and implement user profile" into two tasks:
  - Task 1: ui-ux-designer for design
  - Task 2: frontend-ui-developer for implementation

### 3. Maximize Parallelism
- Always look for tasks that can run simultaneously
- Group independent tasks in the same batch
- Example: All Batch 1 tasks should have zero dependencies

### 4. Clear Task Descriptions
- Provide specific, actionable task descriptions
- Include enough context for the agent to work independently
- Example: "Create POST /api/users endpoint with validation" (good)
  vs. "Work on users" (bad)

### 5. Tool Mandates
- Always specify required tools for codebase operations
- Mandate serena tools for all code reading/writing
- Example: Explicitly list `mcp__serena-mcp__find_symbol` in task requirements

## Common Patterns

### Pattern 1: Design → Implement → Test

```markdown
Batch 1:
- ui-ux-designer: Create wireframe

Batch 2 (after Batch 1):
- frontend-ui-developer: Implement based on wireframe

Batch 3 (after Batch 2):
- web-app-tester: Test implementation
```

### Pattern 2: Parallel Frontend + Backend → Integration

```markdown
Batch 1 (parallel):
- frontend-ui-developer: Create UI component
- backend-api-developer: Create API endpoint

Batch 2 (after Batch 1):
- frontend-ui-developer: Connect UI to API
```

### Pattern 3: Multiple Independent Features → Documentation

```markdown
Batch 1 (parallel):
- backend-api-developer: Feature A
- backend-api-developer: Feature B
- frontend-ui-developer: Feature C

Batch 2 (after Batch 1):
- technical-documentation-writer: Document all features
```

### Pattern 4: Explore → Plan → Implement

```markdown
Batch 1:
- Explore: Understand codebase structure

Batch 2 (after Batch 1):
- Plan: Create detailed implementation plan

Batch 3 (after Batch 2):
- [appropriate-agent]: Implement based on plan
```
