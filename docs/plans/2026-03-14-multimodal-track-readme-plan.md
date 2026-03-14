# Multimodal Track README Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `tracks/multimodal/README.md` into a practical learning guide for the multimodal lesson track.

**Architecture:** This is a documentation-only enhancement. The README will become the single track-level entrypoint that explains lesson order, architectural progression, quick-start commands, and the relationship between the teaching lessons and the local VLM zoo.

**Tech Stack:** Markdown

---

### Task 1: Lock the README structure

**Files:**
- Modify: `tracks/multimodal/README.md`

**Step 1: Draft the target sections**

Write a structured outline containing:

- track goal
- progression summary
- lesson matrix
- quick-start commands
- lesson vs zoo explanation

**Step 2: Review against current track state**

Check the README against the existing three lessons so every lesson name and command is accurate.

**Step 3: Rewrite the document**

Replace the current stub with a teaching-oriented guide that remains concise.

**Step 4: Verify readability**

Re-open the README and confirm the sections scan cleanly and the commands are copy-pasteable.

### Task 2: Verify the updated entrypoint

**Files:**
- Modify: `tracks/multimodal/README.md`

**Step 1: Run discovery smoke**

Run:

`python scripts/run_lesson.py multimodal --list`

Expected: the three multimodal lessons appear exactly as documented.

**Step 2: Inspect the updated README**

Open:

`tracks/multimodal/README.md`

Expected: the file includes the new sections and matches the actual lesson set.
