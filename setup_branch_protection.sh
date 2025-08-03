#!/bin/bash

# Branch Protection Setup Script for AIM2 Project
# This script sets up recommended branch protection rules for the master branch

echo "Setting up branch protection rules for master branch..."

# Create the branch protection configuration
cat > /tmp/branch_protection_config.json << 'EOF'
{
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
}
EOF

# Apply the branch protection rules using GitHub CLI
echo "Applying branch protection rules..."
gh api --method PUT \
  repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master/protection \
  --input /tmp/branch_protection_config.json

if [ $? -eq 0 ]; then
    echo "✅ Branch protection rules successfully applied to master branch"
    echo ""
    echo "Protection rules enabled:"
    echo "  - Require pull request reviews (1 reviewer minimum)"
    echo "  - Dismiss stale reviews when new commits are pushed"
    echo "  - Require conversation resolution before merging"
    echo "  - Prevent force pushes"
    echo "  - Prevent branch deletion"
    echo "  - Allow fork syncing for collaboration"
    echo ""
    echo "To view current protection status:"
    echo "  gh api repos/mberjans/cspirit_ontology_information_extraction_Opus4plan/branches/master/protection"
else
    echo "❌ Failed to apply branch protection rules"
    echo "Please check your GitHub permissions and try again"
fi

# Clean up temporary file
rm -f /tmp/branch_protection_config.json

echo ""
echo "Next steps:"
echo "1. Set up CI/CD workflows for automated testing"
echo "2. Add status check requirements once CI is configured"
echo "3. Consider creating a CODEOWNERS file for automatic reviewer assignment"
echo "4. Review protection rules periodically as team grows"
