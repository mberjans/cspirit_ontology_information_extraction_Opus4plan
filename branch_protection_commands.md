# Branch Protection Commands

## Quick Setup Commands

Execute these commands in your terminal to set up branch protection:

```bash
# Create the protection configuration and apply it
jq -n '{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismissal_restrictions": {},
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false,
    "bypass_pull_request_allowances": {}
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}' | gh api -X PUT repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master/protection --input -
```

## Verification Commands

```bash
# Check current protection status
gh api repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master/protection

# View branch information
gh api repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master

# List all protected branches
gh api repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches --jq '.[] | select(.protected == true) | .name'
```

## Future CI/CD Integration

When you set up GitHub Actions, add these status checks:

```bash
# Add status check requirements (run after setting up CI/CD)
gh api -X PATCH repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master/protection/required_status_checks \
  --field strict=true \
  --field contexts='["ci/python-tests", "ci/linting", "ci/security-scan"]'
```