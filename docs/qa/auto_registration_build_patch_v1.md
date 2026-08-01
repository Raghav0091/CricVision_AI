# Auto-registration build patch V1 QA

Date: 2026-08-01

## Scope

Frontend production-build diagnosis and regression only. No application source,
Next configuration, TypeScript configuration, dependency, lockfile, backend, or
UI behavior was changed.

## Environment

- Windows development workspace: `C:\CricVision_AI\apps\web`
- Node.js: `v24.18.0`
- npm: `11.16.0`
- Next.js: `14.2.35`
- System memory: about 16 GB total; 1.8-2.8 GB free during diagnosis
- Free space on drive C: about 531 GB
- Package state: project-local pnpm-layout `node_modules` and committed
  `pnpm-lock.yaml`; no repository-root `package.json`

## Reproduction and root cause

All CricVision Next workers were stopped before testing. The existing-workspace
build exceeded 600 seconds without emitting a Next build phase or creating a
new build trace. Directly cleaning the ignored `.next` directory then produced
hundreds of `Access denied` errors. A clean direct launch made the failure
explicit:

```text
EPERM: operation not permitted, mkdir 'C:\CricVision_AI\apps\web\.next'
```

The stale `.next` output was owned by the normal Windows user but was not
writable from the restricted command execution context. The apparent hang
occurred before Next compilation or static generation. Inspection found no
top-level API request, filesystem read, unresolved build promise, server-side
WebGL creation, or runtime backend dependency in the Virtual Pitch Lab route.

## Fix

The ignored `.next` directory was removed with normal user file access after
verifying its resolved path remained inside `apps/web`. The unchanged real
workspace build was then run with normal user file access. No repository code
or configuration workaround was justified.

## Results

- Production build: passed in `88.685 s` during specialist isolation and in
  `39.421 s` during the final lead-agent verification
- Compilation: passed
- Next lint and type validation: passed
- Page-data collection: passed
- Static generation: `9/9` pages passed
- Route optimization and build trace collection: passed
- Strict TypeScript (`tsc --noEmit --incremental false`): passed
- Five deterministic camera/bridge scripts: passed
- Desktop QA at `1440 x 900`: no horizontal overflow, one canvas, no console
  warnings or errors
- Mobile QA at `390 x 844`: no horizontal overflow, one canvas, no console
  warnings or errors

## Operational note

On Windows, stop CricVision Next workers before switching between development
and production builds. If `.next` reports `EPERM`, remove only that ignored
output directory from the same normal-user context that runs Node, then rerun
the build. Do not mask TypeScript or ESLint failures.
