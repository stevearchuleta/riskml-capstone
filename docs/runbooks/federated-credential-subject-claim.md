# Federated-Credential Subject Claim — Runbook

## Context

The riskml-capstone repository ships an automated Continuous
Deployment pipeline (see `.github/workflows/cd.yml`) that
authenticates to Azure via federated OpenID Connect, not via a
stored client secret. Every push to the `main` branch triggers a
workflow run, the workflow run mints a short-lived OIDC token
signed by GitHub Actions, and Azure Active Directory verifies the
token's `sub` (subject) claim against a single pre-registered
pattern on Service Principal `sp-riskml-capstone-cd` (appId
`55222d49-8447-4024-a8fd-e9477ded75df`).

That single pre-registered pattern is the entire trust boundary.
If the incoming token's subject claim matches the registered
pattern exactly, Azure issues an access token and the deploy
proceeds. If the subject claim does not match — even by one
character — every CLI call in `cd.yml` fails with `AADSTS70021`
or a similar authentication error, and no image build, no ACR
push, no `az containerapp update`, and no revision flip happens.

This runbook documents the exact subject-claim format, the
specific value registered for this study, the variations
available for different workflow trigger types, common ways the
string drifts, and the diagnostic procedure when authentication
fails.

## The subject-claim format

A GitHub Actions OIDC token's subject claim is a colon-delimited
string that GitHub Actions itself constructs at workflow-run time
based on what triggered the run. The general grammar is:
repo:<owner>/<repo>:<trigger-type>:<trigger-detail>
Three of those four segments are mandatory in this exact order.
The leading `repo:` is a literal prefix and never changes. The
`<owner>/<repo>` segment is the full GitHub repository path
(case-preserving). The `<trigger-type>` segment indicates what
kind of GitHub event produced the workflow run (branch push, tag
push, pull request, environment-scoped deploy). The
`<trigger-detail>` segment qualifies the trigger type with the
specific branch name, tag name, environment name, or special
token (`pull_request`).

## The specific value registered for this study

The federated credential on Service Principal
`sp-riskml-capstone-cd` was registered in Phase 4 Step 1 from a
one-time JSON file (`federated-credential.json`, deleted
immediately after the registration command succeeded) with the
following subject value:
repo:stevearchuleta/riskml-capstone:ref:refs/heads/main
Segment-by-segment anatomy:

| Segment        | Value                              | Meaning                                                              |
|----------------|------------------------------------|----------------------------------------------------------------------|
| Prefix         | `repo:`                            | Literal prefix, always present                                       |
| Repo path      | `stevearchuleta/riskml-capstone`   | GitHub repository path                                               |
| Trigger type   | `ref:`                             | This run was triggered by a Git ref push                             |
| Trigger detail | `refs/heads/main`                  | The specific ref pushed (full Git internal ref name for main branch) |

The `refs/heads/main` segment uses Git's full internal
ref-naming convention. Git stores every reference under a
hierarchical namespace: `refs/heads/<branch>` for branches,
`refs/tags/<tag>` for tags, `refs/remotes/<remote>/<branch>` for
remote-tracking branches. The federated-credential subject
expects the full path including the `refs/heads/` prefix;
abbreviating to `heads/main` or `main` will fail.

## Variations for different trigger types

The trigger-type and trigger-detail segments together support
several patterns. The currently-registered pattern protects this
study against unauthorized deploys by restricting authentication
to branch-push events on `main`. Other patterns serve other use
cases:

| Pattern                                                 | Use case                                                                        |
|---------------------------------------------------------|---------------------------------------------------------------------------------|
| `repo:<owner>/<repo>:ref:refs/heads/<branch>`           | Branch-push workflow (current usage)                                            |
| `repo:<owner>/<repo>:ref:refs/tags/<tag>`               | Tag-push workflow (release-triggered deploys)                                   |
| `repo:<owner>/<repo>:pull_request`                      | Pull-request workflow (PR-triggered staging deploys)                            |
| `repo:<owner>/<repo>:environment:<env-name>`            | GitHub-Environment-gated workflow (requires manual approval before token issue) |

A single Service Principal can hold multiple federated
credentials simultaneously, each with a different subject
pattern. This study registered exactly one credential
(`github-main-branch`) and explicitly skipped the Environment
pattern (a Phase 5 polish candidate). To add a second credential
— for example, to allow PR-triggered staging deploys — a second
`federated-credential.json` file with the PR-scoped subject
would be submitted via `az ad app federated-credential create`.

## Common drift patterns and how each one breaks

The string is rigidly structured and Azure AD does not validate
it at registration time — Azure accepts whatever string the
create command submits. Validation happens at runtime when Azure
compares an incoming token's claim against the registered
pattern. The surface failure mode (`AADSTS70021: No matching
federated identity record found for presented assertion`) is
identical for every drift case, which makes diagnosis difficult
without this runbook.

The drift patterns observed or anticipated for this project:

| Mistake                | Example                                                       | Effect                          |
|------------------------|---------------------------------------------------------------|---------------------------------|
| Wrong syntax for branch| `branch:main` in place of `ref:refs/heads/main`               | Subject claim never matches     |
| Abbreviated ref path   | `ref:heads/main` in place of `ref:refs/heads/main`            | Subject claim never matches     |
| Bare branch name       | `ref:main` in place of `ref:refs/heads/main`                  | Subject claim never matches     |
| Repo path case mismatch| `StevenArchuleta/riskml-capstone`                             | Subject claim never matches     |
| Wrong owner spelling   | `stevenarchuleta/riskml-capstone`                             | Subject claim never matches     |
| Trailing whitespace    | `repo:...refs/heads/main ` (note trailing space)              | Subject claim never matches     |
| Missing prefix         | `stevearchuleta/riskml-capstone:ref:refs/heads/main`          | Subject claim never matches     |

Each of those errors produces an identical `AADSTS70021` at
runtime with no diagnostic that indicates which segment is wrong.

## Diagnostic procedure for authentication failures

When a `cd.yml` run fails at the `azure/login@v2` step with
`AADSTS70021` or a similar authentication error, follow this
procedure in order:

1. **Read the failed run's log for the actual subject claim.**
   Some versions of `azure/login` surface the incoming token's
   subject claim in workflow logs. If the subject is not visible
   in the default log output, enable workflow debug logging by
   setting the `ACTIONS_STEP_DEBUG` repository secret to `true`,
   then re-run the failed workflow.

2. **List the federated credentials currently registered on the
   Service Principal:**
```powershell
   az ad app federated-credential list `
     --id 55222d49-8447-4024-a8fd-e9477ded75df `
     --output json
```
   Verify the `subject` field on the returned credential matches
   the expected pattern exactly.

3. **Verify the repository owner and name match between the
   federated credential and the actual workflow:**
```powershell
   git remote get-url origin
```
   Compare against the `<owner>/<repo>` segment of the federated
   credential's subject.

4. **Verify the workflow trigger is the registered branch:**
```powershell
   git branch --show-current
```
   Compare against the `<branch>` segment of the federated
   credential's subject.

5. **If the subject claim and the registered pattern do not
   match, fix one of two ways:**
   - Update the federated credential to match the actual
     workflow (when the workflow's behavior is correct): delete
     the old credential, then create a new one with the
     corrected subject.
   - Update the workflow to match the registered credential
     (when the credential's pattern is correct): adjust `cd.yml`
     triggers, or change which branch is being pushed.

## Cross-references

- Phase 4 Step 1 session log: `CD_Phase_Plan_NOTES.txt`
  (sections `P4.1-FEDERATED-CREDENTIAL-JSON` through
  `P4.1-FEDERATED-CREDENTIAL-VERIFY`)
- Workflow file using OIDC authentication:
  `.github/workflows/cd.yml`
- Service Principal appId:
  `55222d49-8447-4024-a8fd-e9477ded75df`
- Service Principal name: `sp-riskml-capstone-cd`
- Federated credential name (the only one currently registered):
  `github-main-branch`
- GitHub OIDC documentation:
  https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- Microsoft Entra federated identity credentials documentation:
  https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation
