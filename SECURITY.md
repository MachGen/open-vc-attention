# Security

Report a suspected vulnerability using this repository's GitHub private
vulnerability reporting facility if enabled. Do not post credentials or private
inputs in an issue. If private reporting is unavailable, open an issue asking
maintainers for a private channel without disclosing the vulnerability itself.

Do not load untrusted pickle-based model/capture files. The benchmark's optional
tensor capture loader uses PyTorch `weights_only=True`. GPU code executes with
the permissions of the hosting process; isolate untrusted workloads normally.
