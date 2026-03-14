# Multimodal Track README Enhancement Design

**Date:** 2026-03-14

**Goal:** Turn `tracks/multimodal/README.md` into a practical learning entrypoint for the new multimodal teaching track.

## Problem

The multimodal track now has three runnable lessons:

- CLIP-style retrieval
- BLIP-lite captioning plus ITM
- LLaVA-lite visual instruction answering

However, the current track README is only a short stub. It lists lesson names and one run command, but it does not explain:

- why the lessons are ordered this way
- what each lesson teaches
- how the three lessons differ architecturally
- how lessons relate to the `dlhub.multimodal.vlm` zoo
- how a learner should move through the track

This makes the new track harder to approach than it needs to be.

## Scope

This task only updates:

- `tracks/multimodal/README.md`

It may also add planning documents under `docs/plans/`.

This task will not:

- add a new lesson
- change `run_lesson.py`
- modify root README files already dirty in the worktree

## Approaches Considered

### Approach 1: Minimal README expansion

Add a few paragraphs and more commands.

Pros:

- quick
- low risk

Cons:

- still weak as a teaching entrypoint
- does not show progression clearly

### Approach 2: README as learning guide

Rewrite the README as a structured guide with:

- track goal
- prerequisites
- lesson progression
- lesson comparison matrix
- quick-start commands
- recommended study flow
- lesson vs zoo explanation

Pros:

- best teaching value
- turns the README into a usable index page
- no code risk

Cons:

- slightly longer document

### Approach 3: README plus new helper script

Add the guide and a track-level helper script.

Pros:

- more automation

Cons:

- unnecessary for current scale
- higher maintenance cost

## Chosen Direction

Use Approach 2.

The README should function as the track's front door. It should answer:

- what this track is for
- what each lesson covers
- why the lesson order matters
- how to run each lesson quickly
- how to connect lesson code with the zoo abstractions

## Proposed README Structure

Recommended sections:

1. Track goal
2. Why this track exists
3. Recommended progression
4. Lesson matrix
5. Quick start commands
6. How to study each lesson
7. Lessons vs zoo
8. Output conventions
9. Next steps

## Success Criteria

The README enhancement is complete when:

- it clearly explains the progression from CLIP to BLIP-lite to LLaVA-lite
- it contains runnable commands for lesson discovery and lesson smoke runs
- it helps users understand when to read lesson code versus zoo code
- the updated document remains concise and easy to scan
