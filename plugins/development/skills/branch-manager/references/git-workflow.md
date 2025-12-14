# Git Workflow - Detailed Examples

This reference provides comprehensive Git workflow examples for the branch-manager skill, including conflict resolution, advanced scenarios, and best practices.

## Table of Contents

1. [Basic Branch Operations](#basic-branch-operations)
2. [Conflict Resolution](#conflict-resolution)
3. [Advanced Scenarios](#advanced-scenarios)
4. [Recovery Operations](#recovery-operations)
5. [Best Practices](#best-practices)

## Basic Branch Operations

### Creating a New Feature Branch

```bash
# Start from develop branch
git checkout develop
git pull origin develop

# Create and switch to new branch
git checkout -b feat-user-auth/leaf-login-form

# Verify branch creation
git branch --show-current
# Output: feat-user-auth/leaf-login-form
```

### Making Changes and Committing

```bash
# Check current status
git status

# Stage specific files
git add src/components/LoginForm.tsx
git add src/styles/login.css

# Or stage all changes
git add .

# Commit with descriptive message
git commit -m "feat- implement login form with email validation"

# View commit history
git log --oneline -5
```

### Pushing to Remote

```bash
# First time pushing a new branch
git push -u origin feat-user-auth/leaf-login-form

# Subsequent pushes
git push
```

## Conflict Resolution

### Scenario 1: Merge Conflict When Updating from Develop

**Situation:** You're working on a feature branch and need to merge latest develop changes, but conflicts occur.

```bash
# Update your feature branch with latest develop
git checkout feat-user-auth/leaf-login-form
git fetch origin
git merge origin/develop

# If conflicts occur:
# CONFLICT (content): Merge conflict in src/components/LoginForm.tsx
```

**Resolution Steps:**

1. **Identify conflicted files:**
```bash
git status
# Shows files with conflicts marked as "both modified"
```

2. **Open conflicted file and look for conflict markers:**
```
<<<<<<< HEAD (your changes)
const handleLogin = async (email, password) => {
  return await loginUser(email, password);
};
=======
const handleLogin = (credentials) => {
  return loginUser(credentials);
};
>>>>>>> origin/develop (incoming changes)
```

3. **Resolve the conflict** by choosing one version or combining both:
```javascript
// Resolved version combining both approaches
const handleLogin = async (credentials) => {
  return await loginUser(credentials);
};
```

4. **Mark conflict as resolved and complete merge:**
```bash
# Stage resolved file
git add src/components/LoginForm.tsx

# Complete the merge
git commit -m "merge: resolve conflicts from develop branch"

# Verify all conflicts resolved
git status
```

### Scenario 2: Rebase Conflicts

**When to use rebase:** To maintain a linear history when updating from develop.

```bash
# Start rebase instead of merge
git checkout feat-user-auth/leaf-login-form
git rebase develop

# If conflicts occur, resolve them
# Edit the conflicted files
git add <resolved-files>
git rebase --continue

# If you want to abort the rebase
git rebase --abort
```

### Scenario 3: Merge Conflict When Merging to Develop

```bash
# Attempting to merge feature into develop
git checkout develop
git merge feat-user-auth/leaf-login-form

# If conflicts occur:
# 1. Resolve conflicts in files
# 2. Stage resolved files
git add .
# 3. Complete merge
git commit -m "merge: integrate login form feature"
```

## Advanced Scenarios

### Working with Multiple Leaf Branches

**Scenario:** Working on a dashboard feature with separate leaf branches for different components.

```bash
# Create first leaf branch for charts
git checkout develop
git checkout -b feat-dashboard/leaf-chart-component

# Work and commit
git add src/components/Chart.tsx
git commit -m "feat- implement chart component"

# Switch back to develop
git checkout develop

# Create second leaf branch for data fetching
git checkout -b feat-dashboard/leaf-data-fetching

# Work and commit
git add src/services/dataService.ts
git commit -m "feat- implement data fetching service"
```

### Sequential Merging of Leaf Branches

```bash
# Merge first component
git checkout develop
git merge --no-ff feat-dashboard/leaf-chart-component

# Test the integration
npm test

# If tests pass, merge second component
git merge --no-ff feat-dashboard/leaf-data-fetching

# Test combined functionality
npm test

# If all tests pass, clean up branches
git branch -d feat-dashboard/leaf-chart-component
git branch -d feat-dashboard/leaf-data-fetching
```

### Cherry-Picking Specific Commits

**Scenario:** You want to apply only specific commits from one branch to another.

```bash
# Identify commit hash you want to cherry-pick
git log feat-user-auth/leaf-login-form --oneline

# Switch to target branch
git checkout develop

# Cherry-pick specific commit
git cherry-pick abc1234

# If conflicts occur, resolve and continue
git add .
git cherry-pick --continue
```

### Stashing Work in Progress

**Scenario:** You need to switch branches but have uncommitted changes.

```bash
# Save current work
git stash push -m "WIP: login form validation"

# Switch branches and do other work
git checkout develop
# ... do work ...

# Return to original branch
git checkout feat-user-auth/leaf-login-form

# Restore stashed work
git stash pop

# List all stashes
git stash list

# Apply specific stash without removing it
git stash apply stash@{0}
```

## Recovery Operations

### Undo Last Commit (Keep Changes)

```bash
# Undo last commit but keep changes staged
git reset --soft HEAD~1

# Undo last commit and unstage changes
git reset HEAD~1

# Undo last commit and discard changes (DANGEROUS)
git reset --hard HEAD~1
```

### Recover Deleted Branch

```bash
# Find the commit hash of deleted branch
git reflog

# Create new branch from that commit
git checkout -b feat-user-auth/leaf-login-form abc1234
```

### Revert a Merged Feature

```bash
# Find merge commit hash
git log --oneline

# Revert the merge commit
git revert -m 1 abc1234

# Push the revert
git push origin develop
```

### Recover Lost Commits

```bash
# View all reference updates
git reflog

# Find the lost commit
# Checkout the commit
git checkout abc1234

# Create new branch from it
git checkout -b feat-recovered-feature
```

## Best Practices

### 1. Commit Message Format

Follow conventional commits format:

```bash
# Feature
git commit -m "feat- add user authentication form"

# Bug fix
git commit -m "fix: resolve login button alignment issue"

# Refactor
git commit -m "refactor: simplify authentication logic"

# Documentation
git commit -m "docs: update API documentation"

# Tests
git commit -m "test: add unit tests for login component"

# Style changes
git commit -m "style: format code with prettier"

# Performance
git commit -m "perf: optimize database queries"
```

### 2. Keep Commits Atomic

```bash
# Bad: Multiple unrelated changes
git add .
git commit -m "fix login and add dashboard and update styles"

# Good: Separate commits for each logical change
git add src/components/LoginForm.tsx
git commit -m "fix: resolve login validation issue"

git add src/pages/Dashboard.tsx
git commit -m "feat- add dashboard page"

git add src/styles/global.css
git commit -m "style: update button styles"
```

### 3. Regular Branch Updates

```bash
# Update your feature branch regularly with develop
git checkout feat-user-auth/leaf-login-form
git fetch origin
git merge origin/develop

# Or use rebase for cleaner history
git rebase origin/develop
```

### 4. Branch Cleanup

```bash
# Delete merged local branches
git branch -d feat-user-auth/leaf-login-form

# Delete unmerged branch (force)
git branch -D feat-failed-feature/leaf-component

# Delete remote branch
git push origin --delete feat-user-auth/leaf-login-form

# Prune deleted remote branches from local
git fetch --prune
```

### 5. Pre-Merge Checklist

Before merging any branch to develop:

```bash
# 1. Ensure all changes committed
git status

# 2. Update with latest develop
git merge origin/develop

# 3. Run tests
npm test

# 4. Run linter
npm run lint

# 5. Build project
npm run build

# 6. Only merge if all pass
git checkout develop
git merge --no-ff feat-user-auth/leaf-login-form
```

### 6. Using --no-ff Flag

Always use `--no-ff` when merging feature branches to maintain history:

```bash
# Bad: Fast-forward merge (loses branch history)
git merge feat-user-auth/leaf-login-form

# Good: Creates merge commit
git merge --no-ff feat-user-auth/leaf-login-form
```

This creates a clear visual history showing when features were merged.

### 7. Interactive Rebase for Cleaning History

Before merging, clean up commit history:

```bash
# Interactive rebase last 5 commits
git rebase -i HEAD~5

# Options in editor:
# pick   = use commit
# reword = use commit, but edit message
# squash = combine with previous commit
# drop   = remove commit

# Example: squashing "WIP" commits
pick abc1234 feat- add login form
squash def5678 WIP: fix typo
squash ghi9012 WIP: add validation
pick jkl3456 test: add login tests
```

## Quick Reference Commands

```bash
# Status and info
git status                          # Show working tree status
git log --oneline -10              # Show last 10 commits
git branch -a                       # List all branches
git branch --show-current          # Show current branch
git diff                           # Show unstaged changes
git diff --staged                  # Show staged changes

# Branch operations
git checkout -b <branch>           # Create and switch to branch
git checkout <branch>              # Switch to branch
git branch -d <branch>             # Delete merged branch
git branch -D <branch>             # Force delete branch

# Merging
git merge <branch>                 # Merge branch
git merge --no-ff <branch>         # Merge with merge commit
git merge --abort                  # Abort merge

# Updating
git fetch origin                   # Fetch from remote
git pull origin <branch>           # Pull from remote
git push origin <branch>           # Push to remote

# Conflict resolution
git status                         # See conflicted files
git add <file>                     # Mark as resolved
git merge --continue               # Continue after resolving

# Undoing
git reset HEAD~1                   # Undo last commit, keep changes
git reset --hard HEAD~1            # Undo last commit, discard changes
git revert <commit>                # Create new commit that undoes changes
git clean -fd                      # Remove untracked files and directories

# Stashing
git stash                          # Stash current changes
git stash pop                      # Apply and remove latest stash
git stash list                     # List all stashes
git stash drop                     # Remove latest stash

# History
git log --graph --oneline          # Visual branch history
git reflog                         # Show all reference updates
git blame <file>                   # Show who changed each line
```

## Common Error Messages and Solutions

### Error: "Your local changes would be overwritten by merge"

```bash
# Solution 1: Stash changes
git stash
git merge develop
git stash pop

# Solution 2: Commit changes first
git add .
git commit -m "WIP: save current work"
git merge develop
```

### Error: "fatal: refusing to merge unrelated histories"

```bash
# Allow merging unrelated histories (use carefully)
git merge --allow-unrelated-histories develop
```

### Error: "You have not concluded your merge"

```bash
# Complete the merge or abort it
git merge --continue  # After resolving conflicts
# OR
git merge --abort     # To cancel the merge
```

### Error: "Please commit your changes or stash them before you switch branches"

```bash
# Option 1: Commit
git add .
git commit -m "WIP: current progress"

# Option 2: Stash
git stash

# Option 3: Force switch (loses changes)
git checkout -f <branch>
```
