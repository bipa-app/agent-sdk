//! Dialect-agnostic assertions shared by the Postgres and `SQLite` migration
//! test modules.
//!
//! Both dialects ship the same migration as two files, and the structural
//! guarantees a reviewer cares about — how many columns it adds, that every
//! one can apply to a populated table, that it rewrites no existing row — are
//! properties of the SQL *text*, identical in either dialect. Asserting them
//! from one place keeps the two copies from drifting the first time only one
//! file gains a column.

/// Assert that `sql` adds exactly `expected_columns` columns to `table`,
/// that every one can apply to a populated table, and that nothing is
/// backfilled.
///
/// Two column shapes are safe on a live table and everything else is
/// rejected:
///
/// - an explicit `NULL` column, whose legacy rows read as "not captured";
/// - a `NOT NULL` column **with** a `DEFAULT`, reserved for flags whose
///   default is the honest absent value (a bare `NOT NULL` addition fails
///   outright on a populated table, and evidence columns must stay nullable
///   so a default cannot invent evidence that was never captured).
pub fn assert_additive_migration(sql: &str, table: &str, expected_columns: usize) {
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
        // bare-`NOT NULL` declaration this assertion exists to reject.
        if line.contains("NOT NULL") {
            assert!(
                line.contains("DEFAULT"),
                "a NOT NULL addition must carry a DEFAULT to apply to a populated table, got: {line}",
            );
        } else {
            assert!(
                line.contains(" NULL"),
                "added column must declare NULL explicitly, got: {line}",
            );
        }
    }
    let backfill = format!("UPDATE {}", table.to_ascii_uppercase());
    assert!(
        !sql.to_ascii_uppercase().contains(&backfill),
        "migration must not rewrite existing {table} rows",
    );
}
