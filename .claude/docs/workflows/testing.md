# Testing Workflow

## Testing Philosophy

### Core Principles

**1. NEVER CHANGE TEST EXPECTATIONS TO MAKE TESTS PASS**

Tests validate expected behavior. When a test fails:
1. **Fix the code**, not the test
2. Only change expectations with explicit developer approval
3. Understand WHY the test failed before making changes

**2. TEST USER-FACING BEHAVIOR, NOT COVERAGE PERCENTAGE**

Write tests that:
- ✅ Cover behavior users depend on
- ✅ Validate features that, if broken, would frustrate users
- ✅ Test real workflows, not implementation details
- ✅ Catch regressions before users encounter them

Don't write tests that:
- ❌ Only exist to hit coverage targets
- ❌ Test what the compiler already guarantees
- ❌ Validate internal plumbing users never see
- ❌ Exercise trivial code with no business logic

**3. USE COVERAGE AS A DISCOVERY TOOL**

Coverage helps find **untested user-facing behavior**, not arbitrary percentage targets.

**Process**:
1. Run coverage: `cargo tarpaulin --out Html`
2. Review uncovered lines in the HTML report
3. For each uncovered line, ask: "Would breaking this affect users?"
4. If YES: Write a meaningful test validating the behavior
5. If NO: Mark with `#[cfg(not(tarpaulin_include))]` or skip
6. Commit: `test(module): describe user behavior being tested`

**Prioritize** testing:
- Error handling users will encounter
- State machine transitions
- Data validation and business logic
- API endpoints and responses
- Database operations with side effects
- External service integrations

**Deprioritize/Skip** testing:
- Boilerplate code (type conversions, simple delegators)
- Unreachable error branches
- Internal utilities not user-facing
- Code the compiler guarantees correct
- Trivial getters/setters with no logic

## Running Tests

### Using the Test Skill

```bash
# Run all tests
/test

# Run specific module
/test compliance

# Run specific test
/test compliance::test_policy_evaluation
```

### Using the Test Script

```bash
# Full test suite
./scripts/test.sh

# Specific module
cargo test compliance

# Specific test with output
cargo test test_name -- --nocapture

# With detailed logging
RUST_LOG=debug cargo test test_name -- --nocapture
```

## Test Structure

### Basic Test Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn feature_works() -> anyhow::Result<()> {
        // Setup: Get test database connection
        let conn = &mut crate::test::conn().await;

        // Use fixed time for deterministic tests
        let now = time::macros::datetime!(2024-01-01 00:00:00 UTC);

        // Create test data
        let user_id = create_test_user(conn, now).await?;

        // Execute function under test
        let result = my_function(conn, user_id, now).await?;

        // Assert expectations
        assert_eq!(result.status, ExpectedStatus);
        assert_eq!(result.value, expected_value);

        Ok(())
    }
}
```

### Time Handling

**Always use fixed times** for deterministic tests:

```rust
// ✅ GOOD: Fixed time
let now = time::macros::datetime!(2024-01-01 00:00:00 UTC);
let later = now + time::Duration::hours(1);

// ❌ BAD: Current time (non-deterministic)
let now = time::OffsetDateTime::now_utc();
```

## What Makes a Good Test

### Test Business Logic, Not the Type System

Rust's type system catches many bugs at compile time. **Don't test what the compiler guarantees**.

**❌ USELESS TESTS**:

```rust
// Testing that enum variants exist
#[tokio::test]
async fn status_is_active() {
    let status = Status::Active;
    assert!(matches!(status, Status::Active));
}

// Testing basic CRUD without business logic
#[tokio::test]
async fn create_returns_id() {
    let id = create(conn, params).await.unwrap();
    assert!(*id > 0);  // Diesel guarantees this
}

// Testing that Diesel works
#[tokio::test]
async fn can_find_created_record() {
    let id = create(conn, params).await.unwrap();
    let found = find_by_id(conn, id).await.unwrap();
    assert!(found.is_some());
}
```

**✅ USEFUL TESTS**:

```rust
// Tests actual business behavior
#[tokio::test]
async fn upsert_archives_old_and_creates_new() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;
    let now = datetime!(2024-01-01 00:00:00 UTC);

    // Create initial
    let first_id = create(conn, key, value1, now).await?;

    // Upsert should archive old + create new
    let later = now + Duration::hours(1);
    let second_id = upsert(conn, key, value2, later).await?;

    // Verify new is returned
    let current = find_current(conn, key, later).await?;
    assert_eq!(current.value, value2);

    // Verify old was archived
    let first = find_by_id(conn, first_id).await?.unwrap();
    assert!(first.archived_at.is_some());

    Ok(())
}

// Tests time-based logic
#[tokio::test]
async fn find_current_returns_effective_cost() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;
    let t1 = datetime!(2024-01-01 00:00:00 UTC);
    let t2 = t1 + Duration::days(1);

    // Create costs at different times
    create(conn, provider, check_type, 100, t1).await?;
    create(conn, provider, check_type, 200, t2).await?;

    // Query between costs
    let between = t1 + Duration::hours(12);
    let cost = find_current(conn, provider, check_type, between).await?;
    assert_eq!(cost.cost_cents, 100);

    // Query after second
    let after = t2 + Duration::hours(1);
    let cost = find_current(conn, provider, check_type, after).await?;
    assert_eq!(cost.cost_cents, 200);

    Ok(())
}
```

## State Machine Testing (CRITICAL)

**Must cover all state transitions** for tables with state/status columns.

### Single Table State Machine

```rust
#[tokio::test]
async fn check_run_state_transitions() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;
    let now = datetime!(2024-01-01 00:00:00 UTC);

    // Create in Pending state
    let id = create_check_run(conn, now).await?;

    // Test: Pending → Queued
    queue_check_run(conn, id, now).await?;
    assert_eq!(get_status(conn, id).await?, Status::Queued);

    // Test: Queued → Processing
    start_processing(conn, id, now).await?;
    assert_eq!(get_status(conn, id).await?, Status::Processing);

    // Test: Processing → Completed
    complete(conn, id, result, now).await?;
    assert_eq!(get_status(conn, id).await?, Status::Completed);

    // Test: Invalid transition rejected
    let result = start_processing(conn, id, now).await;
    assert!(result.is_err(), "Should reject invalid transition");

    Ok(())
}
```

### Multi-Table State Dependencies

```rust
#[tokio::test]
async fn policy_run_requires_all_checks_completed() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;
    let now = datetime!(2024-01-01 00:00:00 UTC);

    // Create policy run with 3 check runs
    let (policy_run_id, check_ids) = setup_with_checks(conn, 3, now).await?;

    // Complete only 2 of 3 checks
    for id in &check_ids[..2] {
        complete_check(conn, *id, approved, now).await?;
    }

    // Should still be pending
    let result = evaluate_policy(conn, policy_run_id).await?;
    assert!(matches!(result, EvaluationResult::StillPending));

    // Complete last check
    complete_check(conn, check_ids[2], approved, now).await?;

    // Now should complete
    let result = evaluate_policy(conn, policy_run_id).await?;
    assert!(matches!(result, EvaluationResult::Approved));

    Ok(())
}
```

## Edge Cases to Test

### Empty Inputs

```rust
#[tokio::test]
async fn process_empty_list_returns_empty() -> anyhow::Result<()> {
    let result = process_items(conn, vec![]).await?;
    assert_eq!(result.len(), 0);
    Ok(())
}
```

### Boundary Conditions

```rust
#[tokio::test]
async fn amount_within_limits() -> anyhow::Result<()> {
    // Test minimum
    let min_result = validate_amount(1).await?;
    assert!(min_result.is_valid);

    // Test maximum
    let max_result = validate_amount(1_000_000).await?;
    assert!(max_result.is_valid);

    // Test below minimum
    let below = validate_amount(0).await;
    assert!(below.is_err());

    // Test above maximum
    let above = validate_amount(1_000_001).await;
    assert!(above.is_err());

    Ok(())
}
```

### Concurrent Access / Idempotency

```rust
#[tokio::test]
async fn upsert_is_idempotent() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;
    let now = datetime!(2024-01-01 00:00:00 UTC);

    // First upsert
    let id1 = upsert(conn, key, value, now).await?;

    // Second upsert with same data
    let id2 = upsert(conn, key, value, now).await?;

    // Should create new record (archives old)
    assert_ne!(id1, id2);

    // Old should be archived
    let old = find_by_id(conn, id1).await?.unwrap();
    assert!(old.archived_at.is_some());

    Ok(())
}
```

### Error Conditions

```rust
#[tokio::test]
async fn validation_rejects_invalid_data() -> anyhow::Result<()> {
    let conn = &mut test::conn().await;

    // Empty email
    let result = create_user(conn, "", "Name").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("email"));

    // Invalid format
    let result = create_user(conn, "not-an-email", "Name").await;
    assert!(result.is_err());

    Ok(())
}
```

## Integration Tests

Test features end-to-end:

```rust
#[tokio::test]
async fn compliance_workflow_e2e() -> anyhow::Result<()> {
    let db = test::db().await;
    let msgs = test::msgs().await;
    let now = datetime!(2024-01-01 00:00:00 UTC);

    // 1. Create and activate policy
    let policy_id = db::write!(&db, async |conn| {
        let id = create_policy(conn, policy_params, now).await?;
        activate_policy(conn, id, now).await?;
        Ok(id)
    })?;

    // 2. Submit business customer
    let bc_id = db::write!(&db, async |conn| {
        create_business_customer(conn, bc_params, now).await
    })?;

    // 3. Trigger evaluation
    trigger_evaluation(&db, &msgs, bc_id, now).await?;

    // 4. Verify policy run created
    let policy_run = db::read!(&db, async |conn| {
        find_policy_run_for_bc(conn, bc_id).await
    })?;
    assert!(policy_run.is_some());

    // 5. Verify check runs created
    let check_runs = db::read!(&db, async |conn| {
        list_check_runs(conn, policy_run.unwrap().id).await
    })?;
    assert_eq!(check_runs.len(), 3);

    Ok(())
}
```

## Test Organization

Group related tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    mod create {
        use super::*;

        #[tokio::test]
        async fn succeeds_with_valid_data() { }

        #[tokio::test]
        async fn fails_with_invalid_data() { }
    }

    mod upsert {
        use super::*;

        #[tokio::test]
        async fn archives_old_creates_new() { }

        #[tokio::test]
        async fn creates_if_none_exists() { }
    }

    mod state_transitions {
        use super::*;

        #[tokio::test]
        async fn pending_to_queued() { }

        #[tokio::test]
        async fn invalid_transitions_rejected() { }
    }
}
```

## Test Expectations

### Developer Approval Required

When you're unsure about expected behavior:

1. **Ask the developer**:
   - "What should happen when X occurs?"
   - "Is this the expected output?"
   - "Should this edge case be an error or handled gracefully?"

2. **Write test with their expectations**

3. **If test fails**, fix the **code** not the **test**

4. **Only change expectations with explicit approval**

### Example Workflow

```rust
// Developer says: "Upsert should archive old and create new"

#[tokio::test]
async fn upsert_behavior() -> anyhow::Result<()> {
    let first_id = create(...).await?;
    let second_id = upsert(...).await?;

    // Test matches developer's expectations
    assert_ne!(first_id, second_id, "upsert should create new record");

    let first = find_by_id(first_id).await?.unwrap();
    assert!(first.archived_at.is_some(), "upsert should archive old");

    Ok(())
}

// If this test fails, fix the upsert() function, don't change the asserts!
```

## Related Documentation

- [Testing Patterns Agent](../agents/testing-patterns/agent.md) - Expert help with tests
- [Test Skill](../skills/test/skill.md) - Use `/test` command
- [Database Patterns](../patterns/database.md) - Testing database operations
