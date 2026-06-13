-- Deterministic seed for mini-sql-bench.
-- This file is the single source of truth for the task answer. If you change the
-- data here, recompute and update `task.expected` in config.yaml to match.
--
-- Reference query for the shipped question
-- ("total salary of all employees in the Engineering department"):
--
--   SELECT SUM(salary)
--   FROM employees e JOIN departments d ON e.dept_id = d.id
--   WHERE d.name = 'Engineering';
--
-- Engineering = 150000 + 120000 + 135000 = 405000

PRAGMA foreign_keys = ON;

CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE employees (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    dept_id INTEGER NOT NULL REFERENCES departments(id),
    salary  INTEGER NOT NULL
);

INSERT INTO departments (id, name) VALUES
    (1, 'Engineering'),
    (2, 'Sales'),
    (3, 'Marketing');

INSERT INTO employees (id, name, dept_id, salary) VALUES
    (1, 'Alice', 1, 150000),
    (2, 'Bob',   1, 120000),
    (3, 'Carol', 1, 135000),
    (4, 'Dave',  2,  90000),
    (5, 'Eve',   2, 110000),
    (6, 'Frank', 3,  95000);
