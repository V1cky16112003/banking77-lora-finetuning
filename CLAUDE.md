# gstack
- For web browsing, use the `/browse` skill from gstack. Never use `mcp__claude-in-chrome__*` tools.
- Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/scrape`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/document-generate`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`

## Skill routing
When a user's request matches an available skill, invoke it via the Skill tool. When in doubt:
- Ideas/brainstorming → `/office-hours`
- Strategy/scope → `/plan-ceo-review`, `/plan-eng-review`
- Design system/plan → `/design-consultation`, `/plan-design-review`, `/autoplan`
- Bugs/errors → `/investigate`
- QA/testing → `/qa`, `/qa-only`
- Review/diff → `/review`, `/design-review`
- Ship/deploy/PR → `/ship`, `/land-and-deploy`, `/context-save`, `/context-restore`
- Backlog-ready spec/issue → `/spec`
