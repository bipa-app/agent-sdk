//! Dialect-agnostic assertions shared by the Postgres and `SQLite` migration
//! test modules.
//!
//! Both dialects ship the same migration as two files, and the structural
//! guarantees a reviewer cares about — how many columns it adds, that every
//! one is nullable, that it rewrites no existing row — are properties of the
//! SQL *text*, identical in either dialect. Asserting them from one place
//! keeps the two copies from drifting the first time only one file gains a
//! column.

/// Assert that `sql` adds exactly `expected_columns` nullable columns to
/// `table` and backfills nothing.
///
/// The nullability check is what makes the migration safe to apply to a live
/// table: a `NOT NULL` addition without a default fails outright on a
/// populated table, and one *with* a default silently invents evidence for
/// rows that never captured any.
pub fn assert_additive_nullable_migration(sql: &str, table: &str, expected_columns: usize) {
    assert_eq!(
        sql.matches("ADD COLUMN").count(),
        expected_columns,
        "migration must add exactly {expected_columns} columns",
    );
    for line in sql
        .lines()
        .map(|line| line.trim_start().to_ascii_uppercase())
        .filter(|line| line.starts_with("ADD COLUMN"))
    {
        // `NOT NULL` is checked FIRST and separately: it contains the literal
        // " NULL", so a lone `contains(" NULL")` accepts exactly the
        // declaration this assertion exists to reject.
        assert!(
            !line.contains("NOT NULL"),
            "added column must be nullable, got: {line}",
        );
        assert!(
            line.contains(" NULL"),
            "added column must declare NULL explicitly, got: {line}",
        );
    }
    let backfill = format!("UPDATE {}", table.to_ascii_uppercase());
    assert!(
        !sql.to_ascii_uppercase().contains(&backfill),
        "migration must not rewrite existing {table} rows",
    );
}
