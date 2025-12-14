---
name: branch-manager
description: This skill should be used when managing Git branches throughout the development workflow. Use when creating feature branches before coding work, merging branches after task completion, or cleaning up branches. Automatically activated after task-planner and memory-bank-updater skills, or when user requests branch management (e.g., "브랜치 관리해줘"). Essential for maintaining clean branch structure with separate leaf branches for AI work and core branches for human work.
---

# Branch Manager

## Overview

Manage Git branches systematically throughout the development lifecycle to improve development efficiency. This skill enforces a structured branching strategy with clear separation between AI-driven leaf components and human-managed core architecture, ensuring proper integration through develop branch before production deployment.

## When to Use This Skill

Use this skill in these scenarios:

1. **Before starting any coding work** - Create appropriate branch for the task
2. **After completing a task** - Merge successful work or delete failed branches
3. **When user explicitly requests** - "브랜치 관리해줘" or similar commands
4. **After task-planner skill** - Automatically create branches based on planned tasks
5. **After memory-bank-updater skill** - Clean up branches and update project state

## Branch Structure

The repository follows this hierarchical branch structure:

```
main (production)
└── develop (integration branch)
    ├── feat-<feature-name>/leaf-<component-description>  (AI work)
    ├── feat-<feature-name>/leaf-<another-component>      (AI work)
    └── feat-<feature-name>/core-architecture             (Human work)
```

**Branch Types:**

- **main**: Production-ready code, protected branch
- **develop**: Integration branch for testing before production
- **feat-<name>/leaf-***: AI-driven component work (UI, API, utilities)
- **feat-<name>/core-***: Human-managed architecture and critical systems

**Additional branch type prefixes** (when applicable):
- **bugfix-<name>/leaf-***: Bug fixes for specific components
- **hotfix-<name>/leaf-***: Critical production fixes
- **refactor-<name>/leaf-***: Code refactoring work

## Workflow

### 1. Before Starting Work (Branch Creation)

When beginning any task, create an appropriate branch following these steps:

**Step 1: Determine feature name**
- Extract or derive a concise feature name from the task description
- Use lowercase with hyphens (e.g., "button-feature", "user-authentication")
- Keep it 2-4 words maximum

**Step 2: Identify branch type**
- **AI work on components**: Use `leaf-` prefix (UI, API endpoints, utilities)
- **Human architecture work**: Use `core-` prefix (system design, critical infrastructure)
- **Default for AI**: Always use `leaf-` unless explicitly specified as core work

**Step 3: Create descriptive component name**
- Describe what component/aspect is being worked on
- Examples: `leaf-ui-button`, `leaf-api-integration`, `core-architecture`

**Step 4: Execute branch creation**

```bash
# Ensure on develop branch
git checkout develop
git pull origin develop

# Create and switch to new branch
git checkout -b feat-<feature-name>/leaf-<component-description>

# Example:
git checkout -b feat-button-feature/leaf-ui-component
```

**Step 5: Verify branch creation**
```bash
git branch --show-current
```

### 2. During Work

**Branch Awareness:**
- Always check current branch before making changes: `git branch --show-current`
- Ensure working on correct branch for the task type (leaf vs core)
- Keep commits focused on the branch's specific component

**Commit Practices:**
- Make regular commits with clear messages
- Follow conventional commit format: `feat-`, `fix:`, `refactor:`, etc.
- Keep commits atomic and related to the component

### 3. After Task Completion

Determine success or failure of the task, then take appropriate action:

#### Success Path: Merge to Develop

**Step 1: Prepare for merge**
```bash
# Ensure all changes are committed
git status

# Update develop branch
git checkout develop
git pull origin develop

# Return to feature branch
git checkout feat-<feature-name>/leaf-<component>
```

**Step 2: Merge into develop**
```bash
# Merge develop into feature branch (resolve conflicts if any)
git merge develop

# Switch to develop and merge feature
git checkout develop
git merge --no-ff feat-<feature-name>/leaf-<component>
```

**Step 3: Test in develop**
- Run tests if available
- Verify functionality
- Check for integration issues

**Step 4: Clean up branch (optional)**
```bash
# Delete local branch after successful merge
git branch -d feat-<feature-name>/leaf-<component>

# Delete remote branch if pushed
git push origin --delete feat-<feature-name>/leaf-<component>
```

#### Failure Path: Delete Branch

If the task failed or changes are not needed:

```bash
# Switch to develop branch
git checkout develop

# Delete local branch (use -D for force delete with uncommitted changes)
git branch -D feat-<feature-name>/leaf-<component>

# Delete remote branch if it was pushed
git push origin --delete feat-<feature-name>/leaf-<component>
```

### 4. Integration to Main (Production)

**Only after thorough testing in develop:**

```bash
# Ensure on develop and up to date
git checkout develop
git pull origin develop

# Merge to main
git checkout main
git pull origin main
git merge --no-ff develop

# Tag release if needed
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push to remote
git push origin main
git push origin --tags
```

## Branch Naming Conventions

Follow these conventions strictly for consistency:

### Format Pattern
```
<type>-<feature-name>/<work-type>-<component-description>
```

### Components

**Type prefix:**
- `feat-` - New features (most common)
- `bugfix:` - Bug fixes
- `hotfix:` - Critical production fixes
- `refactor:` - Code refactoring

**Feature name:**
- Lowercase with hyphens
- 2-4 words maximum
- Descriptive of the overall feature
- Examples: `user-auth`, `dashboard-ui`, `payment-integration`

**Work type:**
- `leaf-` - AI-driven component work
- `core-` - Human-managed architecture

**Component description:**
- Specific component being worked on
- Examples: `ui-button`, `api-endpoint`, `data-layer`, `architecture`

### Valid Examples

```
feat-user-auth/leaf-login-form
feat-user-auth/leaf-api-integration
feat-user-auth/core-architecture
bugfix:header/leaf-responsive-layout
hotfix:security/core-authentication
refactor:database/leaf-query-optimization
```

## Common Scenarios

### Scenario 1: Adding a Button to Page

User request: "작업중인 페이지에 버튼을 추가하고 싶어 계획을 수립해줘"

**Workflow:**
1. Task-planner skill creates plan
2. **Branch-manager**: Create branch → `feat-button-feature/leaf-ui-component`
3. Implement button functionality
4. Test and verify
5. **Branch-manager**: Merge to develop
6. Memory-bank-updater: Document changes

### Scenario 2: Multiple Component Feature

User request: "Build dashboard with charts and data fetching"

**Workflow:**
1. Task-planner creates plan with multiple components
2. **Branch-manager**: Create branches:
   - `feat-dashboard/leaf-chart-component`
   - `feat-dashboard/leaf-data-fetching`
   - `feat-dashboard/core-architecture` (if human handles this)
3. Work on each component in respective branch
4. **Branch-manager**: Merge each leaf branch to develop sequentially
5. Test integration in develop
6. Memory-bank-updater: Document completion

### Scenario 3: Failed Task Cleanup

User request: "The API integration didn't work, let's try a different approach"

**Workflow:**
1. Current branch: `feat-api-integration/leaf-endpoint`
2. **Branch-manager**: Delete failed branch
3. Task-planner: Create new plan
4. **Branch-manager**: Create new branch with revised approach
5. Continue work

## Git Command Reference

Quick reference for common branch operations (see `references/git-workflow.md` for detailed examples):

```bash
# Branch creation
git checkout -b <branch-name>

# Branch switching
git checkout <branch-name>

# View current branch
git branch --show-current

# List all branches
git branch -a

# Merge branch
git merge <branch-name>

# Delete local branch
git branch -d <branch-name>    # Safe delete
git branch -D <branch-name>    # Force delete

# Delete remote branch
git push origin --delete <branch-name>

# Update from remote
git pull origin <branch-name>

# Check status
git status
```

## Integration with Other Skills

### With task-planner
- After task-planner creates task breakdown, automatically create appropriate branches
- Map task components to leaf branches
- Create core branch if architecture work identified

### With memory-bank-updater
- After successful merge, trigger memory-bank-updater to document changes
- Include branch information in project history
- Update active context with completed features

## Important Notes

- **Always work on feature branches**, never commit directly to develop or main
- **Keep branch names concise** but descriptive enough to understand purpose
- **Delete branches after merging** to keep repository clean
- **Test in develop branch** before merging to main
- **Use leaf branches for AI work** to maintain clear separation
- **Protect main and develop branches** with repository settings if possible

## Resources

### references/
- `git-workflow.md`: Detailed Git workflow examples and conflict resolution
- `branch-strategy.md`: In-depth explanation of branching strategy and rationale
