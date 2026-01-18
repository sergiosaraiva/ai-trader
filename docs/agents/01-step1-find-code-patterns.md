I want to discover development patterns in our codebase to create Claude Code Skills.

**Mode**: Create OR Update - If `.claude/discovery/codebase-patterns.md` already exists, read it first and update/enhance it. If it doesn't exist, create it from scratch.

**First, consolidate your knowledge from these best practices:**
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://github.com/anthropics/claude-cookbooks/tree/main/skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations

**When unclear about scope or priorities, ask me for clarification with specific options.**

**Then, analyze the codebase and identify:**

## 1. Technology Stack Patterns
- What frameworks and libraries are used? (search *.csproj, packages.config, package.json, pom.xml, requirements.txt)
- What versions? (extract exact version numbers)
- What architectural patterns? (MVC, microservices, monolith, layered, etc.)

## 2. Layer Architecture
- How is the code organized? (identify directory structure)
- What layers exist? (frontend, backend, database, infrastructure)
- How are files named and organized? (extract naming conventions)

## 3. Code Patterns by Layer

For each layer, find 5-10 representative files and extract:
- Common code structures (classes, functions, components)
- Naming conventions (PascalCase, camelCase, snake_case)
- Import/dependency patterns
- Error handling approaches
- Testing patterns
- Documentation styles

## 4. Development Workflows
- How are features typically implemented? (search recent commits: git log --since="3 months ago" --name-status)
- What files change together? (git log --follow --all --source --full-history)
- How are tests organized? (search *test*, *spec*, *Test*)
- How are builds configured? (find build scripts, CI/CD configs)

## 5. Domain-Specific Patterns
- Business logic patterns (search for common domain terms in code)
- Data models and their relationships
- API contracts and communication patterns
- Security and authorization patterns

## 6. Anti-Patterns & Deprecated Approaches (CRITICAL - prevents mistakes)

Search for patterns the team has learned to AVOID:
- Search git history for reverted commits: `git log --grep="revert" --oneline`
- Search code comments for warnings: `grep -r "TODO\|FIXME\|DEPRECATED\|DON'T\|NEVER\|HACK\|XXX" --include="*.cs" --include="*.ts" --include="*.js"`
- Identify patterns that were refactored away (files deleted or significantly changed in last 6 months)
- Look for code review comments indicating anti-patterns

**For each anti-pattern, document:**
- What NOT to do
- Why it's problematic
- What to do instead (with file path reference to correct approach)

## 7. Shared Utilities & Helpers

Identify commonly reused code:
- Utility classes imported by 5+ files
- Helper functions used across multiple modules
- Base classes and common interfaces
- Extension methods
- Shared constants and configuration

## 8. Error Handling Patterns

Document how errors are managed:
- Exception types used (custom vs standard)
- Try-catch patterns and where they're applied
- Logging conventions (what gets logged, format, levels)
- Validation approaches (input validation, business rule validation)
- Error response formats (API error responses, UI error messages)
- Error recovery strategies

---

## Output Requirements

**Create or Update**: `.claude/discovery/codebase-patterns.md`

- **If file exists**: Read existing content, preserve valid patterns, add new discoveries, update outdated entries, increment version
- **If file doesn't exist**: Create new file with all discovered patterns

**Output Format:**

```markdown
# Codebase Patterns Discovery Report

**Generated**: [Date]
**Codebase**: [Project Name]
**Analysis Period**: Last 3 months of commits

## Executive Summary
- Total patterns discovered: X
- High priority patterns: Y
- Anti-patterns documented: Z

---

## Patterns by Priority

### HIGH PRIORITY (used in 20+ files, critical for daily work)

#### Pattern: [Pattern Name]
- **Description**: [What this pattern does]
- **Priority**: HIGH
- **Frequency**: Used in X files, Y commits in last 3 months
- **Layer**: [Backend/Frontend/Database/etc.]

**Examples** (with file paths):
1. `path/to/file1.cs:45-67` - [Brief description]
2. `path/to/file2.cs:120-145` - [Brief description]
3. `path/to/file3.cs:30-55` - [Brief description]

**When to Use**:
- [Condition 1]
- [Condition 2]

**When NOT to Use**:
- [Anti-condition 1]
- [Anti-condition 2]

**Common Variations**:
- [Variation 1]: Used when [condition]
- [Variation 2]: Used when [condition]

**Quality Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]

**Related Patterns**: [Link to related patterns]

---

### MEDIUM PRIORITY (used in 10-20 files)

[Same structure as above]

---

### LOW PRIORITY (used in 5-10 files, specialized use cases)

[Same structure as above]

---

## Anti-Patterns (What NOT to Do)

### Anti-Pattern: [Name]
- **Why it's wrong**: [Explanation]
- **Evidence**: Found in [file:line] (marked for refactoring)
- **Correct approach**: See [Pattern Name] above, example in [file:line]

---

## Shared Utilities Index

| Utility | Location | Used By | Purpose |
|---------|----------|---------|---------|
| [Name] | [file:line] | X files | [Brief purpose] |

---

## Error Handling Summary

| Layer | Exception Type | Logging Pattern | Example |
|-------|---------------|-----------------|---------|
| Backend | [Type] | [Pattern] | [file:line] |
| Frontend | [Type] | [Pattern] | [file:line] |

---

## Technology Stack Summary

| Category | Technology | Version | Notes |
|----------|------------|---------|-------|
| Framework | [Name] | [Version] | [Notes] |
| Database | [Name] | [Version] | [Notes] |
```

---

## Parallel Execution Guidance

For efficiency, run independent operations in parallel:

**DO (parallel - faster):**
```
Read file1.cs || Read file2.cs || Read file3.cs
Grep "pattern1" path1/ || Grep "pattern2" path2/
```

**DON'T (sequential - slower):**
```
Read file1.cs → analyze → Read file2.cs → analyze
```

---

## Success Criteria

- [ ] At least 10-15 HIGH/MEDIUM priority patterns identified
- [ ] Each pattern has 3-5 concrete code examples with file paths
- [ ] Anti-patterns section documents at least 3-5 things to avoid
- [ ] Shared utilities are indexed
- [ ] Error handling patterns are documented
- [ ] All file path references verified to exist
- [ ] Patterns organized by priority (HIGH → MEDIUM → LOW)

Start with the highest-frequency patterns first (most commonly used in recent work).
