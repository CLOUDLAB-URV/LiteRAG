# Security Policy

## Supported Versions

Security updates are currently provided for the latest state of the main branch.

## Reporting a Vulnerability

Please do not open a public issue for security vulnerabilities.

Use one of the following private channels:

1. GitHub Security Advisory (preferred), using the repository Security tab.
2. Direct private contact to maintainers when advisory reporting is unavailable.

Include:

- Clear description of the vulnerability.
- Reproduction steps or proof of concept.
- Impact assessment.
- Suggested mitigation (if available).

## Response Expectations

- Initial acknowledgment target: within 5 business days.
- Triage and severity classification after acknowledgment.
- Coordinated disclosure once a fix is available.

## Scope

This policy covers:

- LiteRAG source code.
- Dependency and supply chain risks.
- Configuration guidance that can lead to insecure deployments.

## Security Best Practices for Users

- Never commit API keys or credentials.
- Keep local secrets in environment variables or local-only config files.
- Rotate compromised credentials immediately.
- Review dependency updates regularly.
