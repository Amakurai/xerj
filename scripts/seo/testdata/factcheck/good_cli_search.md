---
title: "Search your indexed code from the CLI"
evidence:
  - claim: "`xerj search \"<text>\"` retrieves code/passages from a running node"
    source: "engine/crates/xerj-server/src/main.rs:280"
  - claim: "`xerj def \"<symbol>\"` is the go-to-definition client"
    source: "engine/crates/xerj-server/src/main.rs:283"
---

Search the running node from the shell:

    xerj search "wal fsync"

and jump straight to a definition:

    xerj def "write_file_durable"

Both are real subcommands (`xerj search`, `xerj def`); the node itself is
everything else, over HTTP on :9200.
