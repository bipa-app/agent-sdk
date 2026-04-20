---
name: pr
description: Create pull requests with comprehensive descriptions, test plans, and proper formatting. Analyzes all commits in the branch.
---

# Pull Request Skill

## Overview

Creates well-structured pull requests with detailed descriptions based on all commits in the current branch (not just the latest commit).

## Usage

```bash
# Create PR with auto-generated description
/pr

# Create draft PR
/pr --draft

# Create PR with custom title
/pr "Add business customer compliance evaluation"

# Create PR targeting specific base branch
/pr --base develop
```

## What This Does

1. **Analyzes branch history**:
   - Runs `git status` to check current state
   - Runs `git diff [base]...HEAD` to see ALL changes since branch diverged
   - Reviews ALL commits in branch (not just latest)
   - Checks if branch is up to date with remote

2. **Generates PR description**:
   - **Summary**: 1-3 bullet points covering all commits
   - **Test plan**: Checklist of testing steps
   - **Context**: Links to related issues/PRs
   - **Attribution**: Codex footer

3. **Creates PR**:
   - Pushes branch to remote if needed
   - Creates PR using `gh pr create --assignee @me`
   - Sets appropriate labels
   - Returns PR URL

## PR Description Format

```markdown
## Summary

- Implemented business customer compliance policy evaluation system
- Added gRPC endpoints for policy management and check run reviews
- Created database queries for policy run aggregation

## Test plan

- [ ] Run compliance policy evaluation with all checks approved
- [ ] Run compliance policy evaluation with one check rejected
- [ ] Verify check run review workflow (approve/reject)
- [ ] Test policy activation and archival
- [ ] Verify gRPC endpoints return correct data
- [ ] Run full test suite: `./scripts/test.sh`

## Related Issues

Closes #1234

🤖 Generated with Codex
```

## What NOT to Include

❌ Time estimates ("this will take 2-3 weeks")
❌ Implementation details already clear from code
❌ Generic descriptions ("updates to compliance system")
❌ Only the latest commit (must cover ALL commits)

## Validation Before PR Creation

This skill validates:

✅ All commits since branch diverged are analyzed
✅ Branch is pushed to remote (or pushes automatically)
✅ No merge conflicts with base branch
✅ All CI checks pass (if applicable)
✅ Commit messages follow conventions
✅ No uncommitted changes (warns if present)

## Example PR Descriptions

### Feature PR
```markdown
## Summary

- Add compliance policy evaluation system for business customers
- Implement check run aggregation and approval logic
- Create admin gRPC endpoints for policy management

## Technical Details

- New tables: `compliance_policy_runs`, `compliance_check_runs`
- State machine: Pending → Queued → Processing → Completed
- Evaluation logic handles partial check completion

## Test plan

- [ ] Create business customer policy with multiple checks
- [ ] Submit business customer for evaluation
- [ ] Verify check runs are created in Pending state
- [ ] Queue check runs and verify state transitions
- [ ] Complete check runs (mix of approved/rejected)
- [ ] Verify policy run evaluation completes correctly
- [ ] Test edge case: policy with no required checks
- [ ] Run test suite: `cargo test compliance`

🤖 Generated with Codex
```

### Bug Fix PR
```markdown
## Summary

- Fix authentication token refresh race condition
- Prevent session expiration during active usage

## Root Cause

Token refresh was triggered on expiration, but network latency
could cause refresh to complete after the token was already
rejected by the server.

## Solution

Refresh token 5 minutes before expiration to account for
network latency and processing time.

## Test plan

- [ ] Verify token refreshes 5 minutes before expiration
- [ ] Test long-running session (> 1 hour)
- [ ] Verify no authentication failures during active usage
- [ ] Test with slow network conditions
- [ ] Run auth test suite: `cargo test auth`

Closes #5678

🤖 Generated with Codex
```

### Refactor PR
```markdown
## Summary

- Extract common database query patterns to helper module
- Reduce code duplication across query files
- Improve consistency in pagination and filtering

## Changes

- New module: `db/src/query_helpers.rs`
- Refactored 15 query files to use helpers
- No behavior changes (refactor only)

## Test plan

- [ ] Run full test suite: `./scripts/test.sh`
- [ ] Verify all queries return same results as before
- [ ] Check query performance (no regressions)

🤖 Generated with Codex
```

## Branch Management

### Push Branch
If branch not pushed:
```bash
git push -u origin branch-name
```

### Base Branch Detection
Auto-detects base branch:
1. Check `git config` for default base
2. Use `master` or `main` if found
3. Allow override with `--base` flag

### Draft PRs
Create draft PR for work in progress:
```bash
gh pr create --draft --assignee @me --title "WIP: Feature name" --body "..."
```

## Labels and Metadata

Auto-applies labels based on changes:
- `feature`: New functionality
- `bug`: Bug fixes
- `refactor`: Code restructuring
- `database`: Schema changes
- `api`: API changes (gRPC/REST)
- `dependencies`: Dependency updates

## CI Integration

After PR creation:
1. Monitor CI checks with `gh pr checks`
2. Report status to user
3. Suggest fixes if checks fail

## Implementation Pattern

When implementing this skill:
1. Check git status and remote state
2. Get ALL commits since branch diverged from base
3. Analyze full diff (`git diff base...HEAD`)
4. Review each commit message
5. Generate comprehensive summary covering ALL changes
6. Create actionable test plan
7. Push branch if needed (with `-u` flag)
8. Create PR with proper formatting (always use `--assignee @me`)
9. Return PR URL prominently
10. Monitor initial CI status

## Error Handling

### No Remote Branch
If branch not pushed:
- Push automatically with `-u origin branch-name`
- Confirm with user first

### Merge Conflicts
If conflicts with base:
- Warn user
- Suggest `git pull --rebase base-branch`
- Don't create PR until resolved

### Empty PR
If no changes from base:
- Inform user
- Suggest checking branch

### CI Failures
If initial checks fail:
- Report failures clearly
- Link to check details
- Suggest fixes
