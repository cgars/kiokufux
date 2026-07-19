from __future__ import annotations

import io
import json
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

from PIL import Image, ImageOps

from .faces import FaceStore, ReviewState

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="theme-color" content="#21342b">
<title>KiokuFux · People</title>
<style>
/* Faulmann gallery visual system, adapted for the local face-review workbench. */
:root {
  color-scheme: light;
  --pine-950: #17241f;
  --pine-900: #21342b;
  --pine-800: #2f493b;
  --pine-700: #45614d;
  --moss-500: #748166;
  --moss-300: #aeb89b;
  --paper: #f4f0e6;
  --paper-warm: #ebe4d5;
  --paper-light: #faf8f1;
  --ink: #26312b;
  --ink-soft: #687169;
  --rust: #a95735;
  --rust-dark: #824027;
  --danger: #9e4036;
  --line: rgba(38, 55, 47, 0.16);
  --text: var(--ink);
  --muted: var(--ink-soft);
  --blue: var(--pine-700);
  --blue-strong: var(--rust);
  --amber: #d69b45;
  --shadow: 0 18px 50px rgba(23, 36, 31, 0.12);
  --serif: Georgia, "Times New Roman", serif;
  --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  min-width: 280px;
  margin: 0;
  background:
    radial-gradient(circle at 10% 5%, rgba(174, 184, 155, 0.16), transparent 26rem),
    linear-gradient(90deg, rgba(47, 73, 59, 0.025) 1px, transparent 1px),
    var(--paper);
  background-size: auto, 34px 34px, auto;
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
  text-rendering: optimizeLegibility;
}

button, select, input { font: inherit; }

button:focus-visible,
select:focus-visible,
input:focus-visible,
summary:focus-visible,
.group-card:focus-visible,
.face:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--rust) 70%, white);
  outline-offset: 3px;
}

.app-shell {
  min-height: 100vh;
  display: block;
}

.topbar {
  position: relative;
  z-index: 1;
  height: auto;
  min-height: 270px;
  overflow: hidden;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 3rem max(1rem, calc((100vw - 1160px) / 2)) 5.6rem;
  border: 0;
  background:
    linear-gradient(115deg, rgba(23, 36, 31, 0.97), rgba(47, 73, 59, 0.91)),
    repeating-linear-gradient(8deg, transparent 0 11px, rgba(255, 255, 255, 0.04) 12px 13px);
  box-shadow: none;
  backdrop-filter: none;
}

.topbar::before,
.topbar::after {
  position: absolute;
  content: "";
  border: 1px solid rgba(244, 240, 230, 0.12);
  border-radius: 50%;
  pointer-events: none;
}

.topbar::before {
  width: 34rem;
  height: 34rem;
  right: -11rem;
  bottom: -26rem;
  box-shadow: 0 0 0 2rem rgba(244, 240, 230, 0.025), 0 0 0 5rem rgba(244, 240, 230, 0.02);
}

.topbar::after {
  width: 20rem;
  height: 20rem;
  left: -16rem;
  top: -9rem;
}

.brand {
  position: relative;
  display: grid;
  max-width: 13ch;
  gap: 0.65rem;
  color: var(--paper-light);
  font-family: var(--serif);
  font-size: clamp(2.5rem, 6vw, 4.8rem);
  font-weight: 400;
  letter-spacing: -0.045em;
  line-height: 0.98;
}

.brand::before {
  content: "KIOKUFUX · LOCAL REVIEW";
  color: var(--moss-300);
  font-family: var(--sans);
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.18em;
  line-height: 1;
}

.settings {
  position: relative;
  max-width: 24rem;
  margin-top: 4.4rem;
  color: rgba(244, 240, 230, 0.72);
  font-family: var(--serif);
  font-size: 1.08rem;
  line-height: 1.6;
  text-align: right;
}

.settings::after {
  display: block;
  margin-top: 0.65rem;
  color: var(--moss-300);
  content: "Private · local · yours";
  font-family: var(--sans);
  font-size: 0.7rem;
  font-weight: 750;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.tabs {
  position: relative;
  z-index: 5;
  width: min(100% - 2rem, 1160px);
  margin: -2.6rem auto 0;
  padding: 1rem;
  display: flex;
  gap: 0.35rem;
  overflow-x: auto;
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 1.15rem;
  background: rgba(250, 248, 241, 0.95);
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}

.tab {
  flex: 0 0 auto;
  padding: 0.72rem 1rem;
  border: 1px solid transparent;
  color: var(--ink-soft);
  border-radius: 999px;
  cursor: pointer;
  transition: 160ms ease;
}

.tab:hover {
  border-color: var(--line);
  background: var(--paper);
  color: var(--pine-900);
}

.tab.active {
  border-color: var(--pine-800);
  background: var(--pine-900);
  color: var(--paper-light);
  box-shadow: 0 6px 16px rgba(23, 36, 31, 0.16);
}

.workspace {
  display: grid;
  width: min(100% - 2rem, 1280px);
  grid-template-columns: minmax(0, 1fr) 285px;
  gap: 1.35rem;
  margin: 0 auto;
  padding: 2rem 0 4.5rem;
}

.content-card,
.actions-panel {
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 1.25rem;
  background: rgba(250, 248, 241, 0.94);
  box-shadow: var(--shadow);
}

.content-card {
  min-width: 0;
  padding: clamp(1.2rem, 3vw, 2rem);
}

.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 1.75rem;
  padding-bottom: 1.35rem;
  border-bottom: 1px solid var(--line);
}

.eyebrow,
.meta-label {
  font-weight: 750;
  color: var(--rust);
  font-size: 0.7rem;
  letter-spacing: 0.17em;
  text-transform: uppercase;
}

.title {
  max-width: 18ch;
  margin: 0.35rem 0 0.45rem;
  color: var(--pine-950);
  font-family: var(--serif);
  font-size: clamp(2rem, 4.5vw, 3.8rem);
  font-weight: 400;
  letter-spacing: -0.04em;
  line-height: 1;
  text-wrap: balance;
}

.subtitle {
  max-width: 42rem;
  color: var(--ink-soft);
  font-family: var(--serif);
  font-size: 1.03rem;
  line-height: 1.6;
  margin: 0;
}

.metadata { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.45rem; }

.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.65rem;
  border: 1px solid var(--line);
  background: var(--paper);
  color: var(--pine-700);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.pill.warning {
  border-color: rgba(169, 87, 53, 0.28);
  background: #f3dfc8;
  color: var(--rust-dark);
}

.group-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1.15rem;
}

.group-card {
  min-width: 0;
  gap: 0.9rem;
  padding: 0.7rem;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 1rem;
  background: var(--paper-light);
  cursor: pointer;
  display: grid;
  box-shadow: 0 8px 22px rgba(23, 36, 31, 0.06);
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.group-card:hover {
  transform: translateY(-4px);
  border-color: rgba(69, 97, 77, 0.42);
  box-shadow: 0 18px 35px rgba(23, 36, 31, 0.12);
}

.group-card img {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  border-radius: 0.72rem;
  filter: saturate(0.88) contrast(1.02);
}

.group-card b {
  display: block;
  padding: 0 0.25rem;
  color: var(--pine-950);
  font-family: var(--serif);
  font-size: 1.28rem;
  font-weight: 400;
  letter-spacing: -0.02em;
}

.group-card .hint { padding: 0 0.25rem; }
.group-card .metadata { justify-content: flex-start; padding: 0 0.2rem 0.2rem; }

.face-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
  gap: 1rem;
}

.face {
  position: relative;
  padding: 0.55rem;
  border: 1px solid var(--line);
  border-radius: 0.95rem;
  background: var(--paper-light);
  cursor: pointer;
  box-shadow: 0 7px 20px rgba(23, 36, 31, 0.06);
  transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}

.face:hover {
  transform: translateY(-2px);
  border-color: rgba(69, 97, 77, 0.45);
  box-shadow: 0 13px 26px rgba(23, 36, 31, 0.1);
}

.face.selected {
  border-color: var(--rust);
  background: #fff9f2;
  box-shadow: 0 0 0 3px rgba(169, 87, 53, 0.15), 0 14px 28px rgba(23, 36, 31, 0.11);
}

.face img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 0.68rem; }
.face input { position: absolute; opacity: 0; pointer-events: none; }

.checkmark {
  position: absolute;
  top: 0.95rem;
  right: 0.95rem;
  width: 2rem;
  height: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.78);
  background: rgba(23, 36, 31, 0.72);
  display: grid;
  place-items: center;
  color: transparent;
  box-shadow: 0 4px 12px rgba(23, 36, 31, 0.18);
}

.face.selected .checkmark {
  background: var(--rust);
  color: white;
}

.face-caption {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.45rem;
  flex-wrap: wrap;
  padding: 0.25rem 0.15rem 0.05rem;
  margin-top: 0.45rem;
  color: var(--ink-soft);
  font-size: 0.78rem;
}

.quality-badge {
  padding: 0.15rem 0.45rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  border-color: var(--line);
  background: var(--paper);
  color: var(--pine-700);
}

.quality-badge.low {
  border-color: rgba(169, 87, 53, 0.28);
  background: #f3dfc8;
  color: var(--rust-dark);
}

.actions-panel {
  position: sticky;
  top: 1.25rem;
  align-self: start;
  display: grid;
  gap: 0.85rem;
  padding: 1.15rem;
  border-color: rgba(23, 36, 31, 0.35);
  background:
    linear-gradient(155deg, rgba(23, 36, 31, 0.99), rgba(47, 73, 59, 0.97)),
    var(--pine-900);
  color: var(--paper-light);
}

.actions-panel h2 {
  margin: 0;
  color: var(--paper-light);
  font-family: var(--serif);
  font-size: 1.65rem;
  font-weight: 400;
}

.actions-panel .hint { color: rgba(244, 240, 230, 0.68); }
.action-stack { gap: 0.55rem; }
.action-stack { display: grid; }

.btn {
  min-height: 2.75rem;
  border: 1px solid var(--line);
  border-radius: 0.72rem;
  background: var(--paper-light);
  color: var(--pine-950);
  padding: 0.72rem 0.85rem;
  text-align: left;
  cursor: pointer;
  transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
}

.btn:hover {
  transform: translateY(-1px);
  border-color: var(--pine-700);
}

.btn.primary {
  border-color: var(--rust-dark);
  background: var(--rust);
  color: white;
  font-weight: 800;
  text-align: center;
  box-shadow: 0 8px 18px rgba(130, 64, 39, 0.2);
}

.btn.primary:hover { background: var(--rust-dark); }

.actions-panel .btn.secondary {
  border-color: rgba(244, 240, 230, 0.2);
  background: rgba(244, 240, 230, 0.08);
  color: var(--paper-light);
  text-align: center;
}

.actions-panel .btn.secondary:hover {
  border-color: rgba(244, 240, 230, 0.52);
  background: rgba(244, 240, 230, 0.13);
}

.shortcut {
  float: right;
  min-width: 1.5rem;
  padding: 0.05rem 0.35rem;
  border: 1px solid rgba(244, 240, 230, 0.28);
  border-radius: 0.3rem;
  color: var(--moss-300);
  font-size: 0.72rem;
}

.merge-select {
  width: 100%;
  min-height: 2.75rem;
  border-color: rgba(244, 240, 230, 0.2);
  background: rgba(23, 36, 31, 0.82);
  color: var(--paper-light);
  padding: 0.72rem;
  border-radius: 0.72rem;
}

.danger-menu {
  padding-top: 0.75rem;
  border-top: 1px solid rgba(244, 240, 230, 0.16);
  border-color: rgba(244, 240, 230, 0.16);
}

.danger-menu summary {
  padding: 0.35rem 0;
  color: var(--moss-300);
  cursor: pointer;
  font-weight: 700;
}

.danger-menu .btn { width: 100%; margin-top: 0.55rem; }
.btn.danger { border-color: rgba(213, 111, 99, 0.45); background: rgba(158, 64, 54, 0.18); color: #ffe6df; }

.hint { color: var(--ink-soft); font-size: 0.92rem; }
.hidden { display: none !important; }

.empty {
  grid-column: 1 / -1;
  padding: clamp(2rem, 6vw, 4rem);
  border-color: rgba(69, 97, 77, 0.24);
  background: rgba(235, 228, 213, 0.42);
  color: var(--ink-soft);
  text-align: center;
  border: 1px dashed rgba(69, 97, 77, 0.24);
  border-radius: 1rem;
}

.toast {
  position: fixed;
  right: 1.5rem;
  bottom: 1.5rem;
  z-index: 30;
  max-width: 360px;
  padding: 0.9rem 1rem;
  border: 1px solid rgba(244, 240, 230, 0.22);
  border-radius: 0.9rem;
  border-color: rgba(244, 240, 230, 0.22);
  background: rgba(23, 36, 31, 0.96);
  color: var(--paper-light);
  opacity: 0;
  transform: translateY(12px);
  pointer-events: none;
  transition: 180ms ease;
}

.toast.show { opacity: 1; transform: translateY(0); }
.toast.error { border-color: rgba(213, 111, 99, 0.65); color: #ffe6df; }

dialog {
  width: min(1480px, 96vw);
  max-height: 94vh;
  padding: 0;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.74);
  border-radius: 1.25rem;
  background: var(--paper-light);
  color: var(--ink);
  box-shadow: 0 28px 90px rgba(23, 36, 31, 0.34);
}

dialog::backdrop {
  background: rgba(23, 36, 31, 0.78);
  backdrop-filter: blur(5px);
}

.dialog-head {
  height: auto;
  min-height: 4.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1.15rem 0.85rem 1.5rem;
  border-color: var(--line);
  background: var(--paper-light);
}

.dialog-head h2 {
  margin: 0;
  color: var(--pine-950);
  font-family: var(--serif);
  font-size: clamp(1.5rem, 3vw, 2.35rem);
  font-weight: 400;
}

.dialog-head .btn { min-width: 5rem; text-align: center; }

.compare-items {
  display: grid;
  max-height: calc(94vh - 4.5rem);
  overflow: auto;
  gap: 1.25rem;
  padding: 1.25rem;
  background: var(--paper-warm);
}

.compare-item {
  display: grid;
  grid-template-columns: 190px minmax(0, 1fr);
  gap: 1.1rem;
  min-height: 60vh;
  padding: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 1rem;
  background: var(--paper-light);
  box-shadow: 0 10px 28px rgba(23, 36, 31, 0.08);
}

.reference-panel {
  padding: 0.9rem;
  align-self: start;
  border: 1px solid var(--line);
  border-radius: 1rem;
  border-color: var(--line);
  background: var(--paper);
}

.reference-panel h3 {
  margin: 0.3rem 0 0.8rem;
  color: var(--pine-950);
  font-family: var(--serif);
  font-size: 1.3rem;
  font-weight: 400;
}

.reference-panel img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 0.8rem; }
.photo-panel { min-width: 0; }

.canvas-toolbar { min-height: 2.8rem; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.65rem; }
.canvas-toolbar > b { color: var(--pine-950); font-family: var(--serif); font-size: 1.15rem; font-weight: 400; }

.zoom-controls button {
  padding: 0.45rem 0.65rem;
  border: 1px solid var(--line);
  border-radius: 0.65rem;
  border-color: var(--line);
  background: var(--paper);
  color: var(--pine-950);
  cursor: pointer;
}

.zoom-controls button:hover { border-color: var(--pine-700); background: white; }
.zoom-controls { display: flex; align-items: center; gap: 0.35rem; }
.zoom-controls output { min-width: 3.5rem; color: var(--ink-soft); text-align: center; }

.zoom-viewport {
  width: 100%;
  max-height: 70vh;
  overflow: auto;
  overscroll-behavior: contain;
  border: 1px solid rgba(23, 36, 31, 0.45);
  border-radius: 1rem;
  border-color: rgba(23, 36, 31, 0.45);
  background: var(--pine-950);
}

.photo-wrap { position: relative; width: 100%; margin: auto; line-height: 0; }
.photo-wrap > img { display: block; width: 100%; height: auto; border-radius: 1rem; }
.face-box { position: absolute; box-sizing: border-box; border: 3px solid #e5ad50; border-radius: 0.38rem; pointer-events: none; }
.face-box.active { border-color: #d9683d; box-shadow: 0 0 0 2px rgba(250, 248, 241, 0.9); }
.face-box span { position: absolute; top: -1.8rem; left: 0; padding: 0.15rem 0.4rem; border-radius: 999px; background: var(--pine-950); color: white; font-size: 0.72rem; line-height: normal; white-space: nowrap; }

.name-dialog { width: min(520px, calc(100vw - 2rem)); }
.name-form { display: grid; gap: 1rem; padding: clamp(1.25rem, 4vw, 2rem); }
.name-form .eyebrow { margin: 0; }
.name-form h2 { margin: 0; color: var(--pine-950); font-family: var(--serif); font-size: clamp(1.8rem, 5vw, 2.65rem); font-weight: 400; letter-spacing: -0.035em; }
.name-form p { margin: 0; color: var(--ink-soft); }
.name-form label { color: var(--pine-800); font-size: 0.78rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
.name-form input { width: 100%; min-height: 3rem; padding: 0.7rem 0.85rem; border: 1px solid var(--line); border-radius: 0.72rem; background: var(--paper); color: var(--ink); }
.name-form input:focus { border-color: var(--pine-700); background: white; box-shadow: 0 0 0 4px rgba(69, 97, 77, 0.1); outline: none; }
.form-actions { display: flex; justify-content: flex-end; gap: 0.65rem; padding-top: 0.35rem; }
.form-actions .btn { min-width: 7rem; text-align: center; }

@media (max-width: 900px) {
  .topbar { min-height: 250px; padding-bottom: 5.2rem; }
  .settings { max-width: 18rem; margin-top: 3.5rem; }
  .workspace { grid-template-columns: 1fr; }
  .actions-panel { position: static; order: -1; }
  .compare-item { grid-template-columns: 1fr; }
  .reference-panel { max-width: none; display: grid; grid-template-columns: 110px 1fr; gap: 0 1rem; }
  .reference-panel img { grid-row: 1 / 4; }
}

@media (max-width: 620px) {
  .topbar { min-height: 300px; flex-direction: column; gap: 1rem; padding-top: 2.2rem; }
  .brand { font-size: clamp(2.35rem, 13vw, 3.7rem); }
  .settings { max-width: 26rem; margin-top: 0; text-align: left; }
  .tabs { padding: 0.7rem; }
  .workspace { width: min(100% - 1rem, 1280px); padding-top: 1rem; }
  .content-card { padding: 1rem; }
  .section-head { display: grid; }
  .metadata { justify-content: flex-start; }
  .group-list { grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); }
  .face-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
  .compare-items { padding: 0.55rem; }
  .compare-item { min-height: auto; padding: 0.65rem; }
  .canvas-toolbar { align-items: flex-start; flex-direction: column; }
  .zoom-controls { width: 100%; overflow-x: auto; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
}
</style>
</head>
<body>
<div class="app-shell"><header class="topbar"><div class="brand">KiokuFux · People</div><div class="settings">Local archive workbench</div></header><nav class="tabs" aria-label="Face review sections"><button class="tab active" onclick="showGroups()">Groups</button><button class="tab" onclick="showUngrouped()">Ungrouped</button><button class="tab" onclick="showConfirmed()">Confirmed</button><button class="tab" onclick="showPlaceholder('Needs review')">Needs review</button></nav><main class="workspace"><section class="content-card"><div id="list"><div class="section-head"><div><div class="eyebrow">Anonymous discovery</div><h1 class="title">Possible recurring people</h1><p class="subtitle">Review machine-generated groups without turning them into identities.</p></div><div class="metadata"><span class="pill">Local only</span><span class="pill">No names proposed</span></div></div><div id="groups" class="group-list"></div></div><div id="detail" class="hidden"><div class="section-head"><div><button class="btn" onclick="showGroups()">← All groups</button><div class="eyebrow" style="margin-top:18px">Possible recurring person</div><h1 id="groupTitle" class="title"></h1><p id="groupMeta" class="subtitle"></p></div><div class="metadata" id="groupBadges"></div></div><p class="hint">Click a face to see it in the source photograph. Select one or more faces for comparison and corrections.</p><div id="faces" class="face-grid"></div></div></section><aside class="actions-panel" id="actions"><h2>Actions</h2><p class="hint" id="selectionHint">Open a group to review its face occurrences.</p><div class="action-stack"><button class="btn primary" onclick="confirmPerson()">Confirm person</button><button class="btn secondary" onclick="compareSelected()">Compare selected <span class="shortcut">C</span></button><button class="btn secondary" onclick="act('split')">Split selected <span class="shortcut">S</span></button><button class="btn secondary" onclick="createGroupFromSelected()">Create group from selected</button><select class="merge-select" id="mergeTarget" aria-label="Merge with another group"></select><button class="btn secondary" onclick="mergeGroup()">Merge into selected group</button><button class="btn secondary" onclick="reviewGroup()">Mark group reviewed <span class="shortcut">R</span></button><select class="merge-select" id="personMergeTarget" aria-label="Merge with another confirmed person"></select><button class="btn secondary" onclick="mergeConfirmedPerson()">Merge confirmed people</button></div><details class="danger-menu"><summary>More actions</summary><button class="btn danger" onclick="act('reject-face')">Reject detection</button><button class="btn danger" onclick="act('exclude-from-clustering')">Exclude poor crop</button></details></aside></main></div><div id="toast" class="toast" role="status" aria-live="polite"></div>
<dialog id="context"><div class="dialog-head"><h2 id="compareTitle">Face comparison</h2><button class="btn" onclick="context.close()">Close</button></div><div id="compareItems" class="compare-items"></div></dialog>
<dialog id="confirmDialog" class="name-dialog" aria-labelledby="confirmTitle"><form class="name-form" onsubmit="submitPerson(event)"><div class="eyebrow">Human confirmation</div><h2 id="confirmTitle">Confirm this person</h2><p>A name is optional. Leaving it blank keeps the permanent anonymous friendly name.</p><label for="personName">Display name</label><input id="personName" type="text" autocomplete="off" placeholder="Optional name"><div class="form-actions"><button class="btn" type="button" onclick="confirmDialog.close()">Cancel</button><button class="btn primary" type="submit">Confirm person</button></div></form></dialog>
<script>
let collectionId, currentGroup, currentPerson, allGroups=[], allPeople=[];
const api=async(path,options={})=>{let r=await fetch(path,options);let data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data};
const mutate=(path,body={})=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({collection_id:collectionId,...body})});
function notify(message,type='ok'){toast.textContent=message;toast.className='toast show'+(type==='error'?' error':'');clearTimeout(notify.timer);notify.timer=setTimeout(()=>toast.className='toast',3200)}
async function withFeedback(label,fn){try{let result=await fn();notify(label+' saved.');return result}catch(e){notify(e.message,'error');throw e}}
function setActiveTab(label){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.textContent.trim()===label))}
function setListHeader(eyebrow,title,subtitle,badges){let head=list.querySelector('.section-head');head.querySelector('.eyebrow').textContent=eyebrow;head.querySelector('.title').textContent=title;head.querySelector('.subtitle').textContent=subtitle;head.querySelector('.metadata').innerHTML=badges.map(value=>`<span class="pill">${value}</span>`).join('')}
function setActionMode(mode){actions.dataset.mode=mode;let show=(selector,visible)=>{let element=actions.querySelector(selector);if(element)element.classList.toggle('hidden',!visible)};show('[onclick="confirmPerson()"]',mode==='group');show('[onclick="compareSelected()"]',mode==='group');show('[onclick="act(\'split\')"]',mode==='group');show('[onclick="createGroupFromSelected()"]',mode==='ungrouped');show('#mergeTarget',mode==='group');show('[onclick="mergeGroup()"]',mode==='group');show('[onclick="reviewGroup()"]',mode==='group');show('#personMergeTarget',mode==='person');show('[onclick="mergeConfirmedPerson()"]',mode==='person');show('.danger-menu',mode==='group')}
function setDetailKind(kind,backLabel,backAction){let back=detail.querySelector('.section-head .btn');detail.querySelector('.eyebrow').textContent=kind;back.textContent='← '+backLabel;back.onclick=backAction}
async function start(){collectionId=(await api('/api/status')).collection_id;await showGroups()}
async function showGroups(){currentPerson=null;setActiveTab('Groups');setActionMode('browse');setListHeader('Anonymous discovery','Possible recurring people','Review machine-generated groups without turning them into identities.',['Local only','No names proposed']);detail.classList.add('hidden');list.classList.remove('hidden');allGroups=await api('/api/groups');groups.className='group-list';groups.innerHTML=allGroups.length?allGroups.map(groupCard).join(''):'<p class="empty">No recurring groups yet.</p>';selectionHint.textContent='Open a group to review its face occurrences.';mergeTarget.innerHTML='<option value="">Merge with…</option>'}
function groupCard(g){let state=g.conflict?'Conflict':g.review_state.replace('_',' ');return `<article class="group-card" role="button" tabindex="0" onclick="openGroup('${g.group_id}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openGroup('${g.group_id}')}"><img src="/api/faces/${g.representative_face_id}/thumbnail" alt="Representative face for ${g.friendly_id}"><div><b>${g.friendly_id}</b><div class="hint">Possible recurring person</div></div><div class="metadata"><span class="pill">${g.photo_count} photos</span><span class="pill">${g.face_count} occurrences</span><span class="pill ${g.conflict?'warning':''}">${state}</span></div></article>`}
async function showConfirmed(){currentGroup=null;currentPerson=null;setActiveTab('Confirmed');setActionMode('browse');setListHeader('Known with care','Confirmed people','Only people you explicitly confirm receive a durable identity.',['User confirmed','Local only']);detail.classList.add('hidden');list.classList.remove('hidden');groups.className='group-list';allPeople=await api('/api/people');groups.innerHTML=allPeople.length?allPeople.map(personCard).join(''):'<p class="empty">No confirmed people yet. Review a coherent group, then choose Confirm person.</p>';selectionHint.textContent='Open a confirmed person to inspect their photographs or merge duplicates.';mergeTarget.innerHTML='<option value="">Merge with…</option>';personMergeTarget.innerHTML='<option value="">Merge confirmed person with…</option>'}
function personCard(p){let title=p.display_name||p.friendly_name;let secondary=p.display_name?`<div class="hint">${p.friendly_name} · permanent friendly name</div>`:'<div class="hint">Unnamed confirmed person</div>';let image=p.representative_face_id?`<img src="/api/faces/${p.representative_face_id}/thumbnail" alt="Representative face for ${title}">`:'';return `<article class="group-card" role="button" tabindex="0" onclick="openPerson('${p.person_id}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openPerson('${p.person_id}')}">${image}<div><b>${title}</b>${secondary}</div><div class="metadata"><span class="pill">confirmed</span><span class="pill">${p.face_count} faces</span><span class="pill">${p.photo_count} photos</span></div></article>`}
async function openPerson(id){currentGroup=null;currentPerson=await api('/api/people/'+id);setActionMode('person');setDetailKind('Confirmed person','All confirmed people',showConfirmed);list.classList.add('hidden');detail.classList.remove('hidden');let title=currentPerson.display_name||currentPerson.friendly_name;groupTitle.textContent=title;groupMeta.textContent=`${currentPerson.face_count} confirmed faces · ${currentPerson.photo_count} photographs`;groupBadges.innerHTML=`<span class="pill">confirmed person</span><span class="pill">${currentPerson.friendly_name}</span>`;faces.innerHTML=currentPerson.faces.map(faceCard).join('');personMergeTarget.innerHTML='<option value="">Merge confirmed person with…</option>'+allPeople.filter(p=>p.person_id!==id).map(p=>`<option value="${p.person_id}">${p.display_name||p.friendly_name} · ${p.face_count} faces</option>`).join('');selectionHint.textContent='Click a confirmed face to open its source photograph.'}
async function showUngrouped(){currentGroup=null;currentPerson=null;setActiveTab('Ungrouped');setActionMode('ungrouped');setListHeader('Loose photographs','Ungrouped detections','Select related face crops to begin a new anonymous recurring-person group.',['Not identified','Local only']);detail.classList.add('hidden');list.classList.remove('hidden');groups.className='face-grid';let fs=await api('/api/ungrouped');groups.innerHTML=fs.length?fs.map(faceCard).join(''):'<p class="empty">No ungrouped faces.</p>';selectionHint.textContent='Select two or more ungrouped faces with the checkmark, then create a group.';mergeTarget.innerHTML='<option value="">Merge with…</option>'}
function showPlaceholder(label){currentGroup=null;currentPerson=null;setActiveTab(label);setActionMode('browse');setListHeader('Review queue',label,'Decisions that need another look will collect here.',['Local only']);detail.classList.add('hidden');list.classList.remove('hidden');groups.className='group-list';groups.innerHTML=`<p class="empty">${label} will appear here as review decisions are available.</p>`;selectionHint.textContent='Choose Groups or Ungrouped to continue reviewing.'}
function confidenceBadge(f){let value=Math.round((f.confidence||0)*100);let low=value<90;return `<span class="quality-badge ${low?'low':''}">${low?'Low confidence '+value+'%':'Clear crop'}</span>`}
function faceCard(f){return `<article class="face" role="button" tabindex="0" aria-pressed="false" data-id="${f.face_id}" onclick="faceClick(event,this,'${f.image_id}','${f.face_id}')" onkeydown="if(event.key==='Enter'){viewContext(event,'${f.image_id}','${f.face_id}')}else if(event.key===' '){event.preventDefault();toggleFace(this)}"><input type="checkbox" aria-label="Select face"><span class="checkmark">✓</span><img src="/api/faces/${f.face_id}/thumbnail" alt="Detected face"><div class="face-caption"><span>Open photograph</span>${confidenceBadge(f)}</div></article>`}
async function openGroup(id){currentPerson=null;currentGroup=await api('/api/groups/'+id);setActionMode('group');setDetailKind('Possible recurring person','All groups',showGroups);list.classList.add('hidden');detail.classList.remove('hidden');groupTitle.textContent=currentGroup.friendly_id;groupMeta.textContent=`${currentGroup.faces.length} occurrences · possible recurring person`;groupBadges.innerHTML=`<span class="pill">${currentGroup.review_state.replace('_',' ')}</span>${currentGroup.conflict?'<span class="pill warning">same-photo conflict</span>':''}`;faces.innerHTML=currentGroup.faces.map(faceCard).join('');mergeTarget.innerHTML='<option value="">Merge with…</option>'+allGroups.filter(g=>g.group_id!==id).map(g=>`<option value="${g.group_id}">${g.friendly_id} · ${g.face_count} faces</option>`).join('');selectionHint.textContent='No faces selected.'}
function faceClick(e,el,imageId,faceId){if(e.target.closest('.checkmark')||e.shiftKey||e.metaKey||e.ctrlKey){toggleFace(el);return}viewContext(e,imageId,faceId)}
function toggleFace(el){el.classList.toggle('selected');let isSelected=el.classList.contains('selected');el.querySelector('input').checked=isSelected;el.setAttribute('aria-pressed',String(isSelected));let count=selected().length;selectionHint.textContent=count?`${count} selected. Press C to compare, S to split, R to mark reviewed.`:'No faces selected.'}
const selected=()=>[...document.querySelectorAll('.face.selected')].map(x=>x.dataset.id);
async function act(name){let ids=selected();if(!ids.length)return notify('Select at least one face.','error');if(!currentGroup)return notify('Open a group before using this action, or create a group from ungrouped selections.','error');try{await withFeedback(name.replaceAll('-',' '),()=>mutate('/api/review/'+name,{group_id:currentGroup.group_id,face_ids:ids}));await openGroup(currentGroup.group_id)}catch(e){}}
async function mergeGroup(){if(!currentGroup)return notify('Open a group before merging.','error');if(!mergeTarget.value)return notify('Choose another group.','error');try{await withFeedback('Merge',()=>mutate('/api/review/merge',{source_group_id:currentGroup.group_id,target_group_id:mergeTarget.value}));await showGroups()}catch(e){}}
async function reviewGroup(){if(!currentGroup)return notify('Open a group before marking it reviewed.','error');try{await withFeedback('Review state',()=>mutate('/api/review/mark-group-reviewed',{group_id:currentGroup.group_id}));await openGroup(currentGroup.group_id)}catch(e){}}
function confirmPerson(){if(!currentGroup)return notify('Open a reviewed group first.','error');personName.value='';confirmDialog.showModal();setTimeout(()=>personName.focus(),0)}
async function submitPerson(event){event.preventDefault();let name=personName.value.trim();try{await withFeedback('Person confirmation',()=>mutate('/api/people',{group_id:currentGroup.group_id,display_name:name||null}));confirmDialog.close();await showGroups()}catch(e){}}
async function createGroupFromSelected(){let ids=selected();if(ids.length<2)return notify('Select at least two ungrouped faces to create a group.','error');try{let result=await withFeedback('New group',()=>mutate('/api/review/create-group',{face_ids:ids}));await showGroups();if(result.group_id)await openGroup(result.group_id)}catch(e){}}
async function mergeConfirmedPerson(){if(!currentPerson)return notify('Open a confirmed person before merging.','error');if(!personMergeTarget.value)return notify('Choose another confirmed person.','error');try{let result=await withFeedback('Confirmed people merge',()=>mutate('/api/people/merge',{source_person_id:currentPerson.person_id,target_person_id:personMergeTarget.value}));await showConfirmed();await openPerson(result.person_id)}catch(e){}}
function applyZoom(item,scale){scale=Math.max(.5,Math.min(5,scale));let wrap=item.querySelector('.photo-wrap');wrap.dataset.scale=scale;wrap.style.width=(scale*100)+'%';item.querySelector('.zoom-value').value=Math.round(scale*100)+'%'}
function zoomBy(button,factor){let item=button.closest('.compare-item'),wrap=item.querySelector('.photo-wrap');applyZoom(item,(Number(wrap.dataset.scale)||1)*factor)}
function resetZoom(button){let item=button.closest('.compare-item');applyZoom(item,1);item.querySelector('.zoom-viewport').scrollTo(0,0)}
function wheelZoom(event){if(!event.ctrlKey&&!event.metaKey)return;event.preventDefault();let item=event.currentTarget.closest('.compare-item'),wrap=item.querySelector('.photo-wrap');applyZoom(item,(Number(wrap.dataset.scale)||1)*(event.deltaY<0?1.2:1/1.2))}
async function comparisonItem(face,index){let item=document.createElement('article');item.className='compare-item';item.innerHTML=`<aside class="reference-panel"><div class="eyebrow">Selected face</div><h3>Occurrence ${index+1}</h3><img src="/api/faces/${face.face_id}/thumbnail"><p class="hint">Rust outline marks this face. Amber outlines mark other detections.</p></aside><section class="photo-panel"><div class="canvas-toolbar"><b>Source photograph</b><div class="zoom-controls"><button onclick="resetZoom(this)">Fit</button><button onclick="resetZoom(this)">100%</button><button onclick="zoomBy(this,1/1.25)" aria-label="Zoom out">−</button><output class="zoom-value">100%</output><button onclick="zoomBy(this,1.25)" aria-label="Zoom in">+</button></div></div><div class="zoom-viewport" onwheel="wheelZoom(event)"><div class="photo-wrap" data-scale="1"><img src="/api/images/${face.image_id}/thumbnail"></div></div></section>`;compareItems.appendChild(item);let wrap=item.querySelector('.photo-wrap');let detections=await api('/api/images/'+face.image_id+'/faces');detections.forEach((f,i)=>{let box=document.createElement('div');box.className='face-box'+(f.face_id===face.face_id?' active':'');box.style.left=(f.x1*100)+'%';box.style.top=(f.y1*100)+'%';box.style.width=((f.x2-f.x1)*100)+'%';box.style.height=((f.y2-f.y1)*100)+'%';box.innerHTML='<span>'+(f.face_id===face.face_id?'selected face':'face '+(i+1))+'</span>';wrap.appendChild(box)})}
async function showComparison(items){compareItems.innerHTML='';compareTitle.textContent=items.length===1?'Photograph context':`${items.length} selected occurrences side by side`;try{await Promise.all(items.map(comparisonItem));context.showModal()}catch(err){notify(err.message,'error')}}
function compareSelected(){let ids=selected();if(ids.length<2)return notify('Select at least two faces to compare.','error');showComparison(currentGroup.faces.filter(f=>ids.includes(f.face_id)))}
function viewContext(e,imageId,faceId){e.stopPropagation();let face=(currentGroup?.faces||[]).find(f=>f.face_id===faceId)||{image_id:imageId,face_id:faceId};showComparison([face])}
document.addEventListener('keydown',event=>{if(event.target.matches('input,select,textarea'))return;if(event.key.toLowerCase()==='c')compareSelected();if(event.key.toLowerCase()==='s')act('split');if(event.key.toLowerCase()==='r')reviewGroup();if(event.key==='Escape'&&context.open)context.close()});
start().catch(e=>notify(e.message,'error'));
</script>
</body>
</html>"""


def safe_collection_path(root: Path, candidate: str) -> Path:
    windows_candidate = PureWindowsPath(candidate)
    native_candidate = Path(candidate)
    if windows_candidate.anchor:
        raise ValueError("path leaves collection")
    relative_candidate = Path(*windows_candidate.parts) if "\\" in candidate else native_candidate
    resolved = (root / relative_candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path leaves collection")
    return resolved


def _refresh_group(store: FaceStore, group_id: str) -> None:
    members = store.db.execute("""SELECT f.face_id,f.image_id FROM face_group_members m
      JOIN face_occurrences f USING(face_id) WHERE m.group_id=? ORDER BY f.face_id""", (group_id,)).fetchall()
    if not members:
        store.db.execute("DELETE FROM face_groups WHERE group_id=?", (group_id,))
        return
    conflict = len(members) != len({member["image_id"] for member in members})
    store.db.execute("UPDATE face_groups SET representative_face_id=?,conflict=? WHERE group_id=?",
                     (members[0]["face_id"], int(conflict), group_id))


def _confirmed_person(store: FaceStore, state: ReviewState, person_id: str, *, include_faces: bool = False) -> dict[str, object] | None:
    person = next((dict(person) for person in state.people.get("people", []) if person.get("person_id") == person_id), None)
    face_ids = list(dict.fromkeys((state.review.get("person_faces") or {}).get(person_id, [])))
    if person is None and not face_ids:
        return None
    person = person or {"person_id": person_id, "friendly_name": person_id, "display_name": None}
    if face_ids:
        placeholders = ",".join("?" for _ in face_ids)
        face_rows = store.db.execute(f"""SELECT face_id,image_id,confidence,quality,x1,y1,x2,y2 FROM face_occurrences
          WHERE face_id IN ({placeholders}) ORDER BY face_id""", face_ids).fetchall()
    else:
        face_rows = []
    result: dict[str, object] = {
        "person_id": person_id,
        "friendly_name": person.get("friendly_name"),
        "display_name": person.get("display_name"),
        "face_count": len(face_rows),
        "photo_count": len({row["image_id"] for row in face_rows}),
        "representative_face_id": face_rows[0]["face_id"] if face_rows else None,
    }
    if include_faces:
        result["faces"] = [dict(row) for row in face_rows]
    return result


def _confirmed_people(store: FaceStore, state: ReviewState) -> list[dict[str, object]]:
    person_ids = sorted({*(person.get("person_id") for person in state.people.get("people", []) if person.get("person_id")), *(state.review.get("person_faces") or {}).keys()})
    return [person for person_id in person_ids if (person := _confirmed_person(store, state, person_id)) is not None]


def _merge_people(state: ReviewState, source_person_id: str, target_person_id: str) -> dict[str, object]:
    if not source_person_id or not target_person_id or source_person_id == target_person_id:
        raise ValueError("choose two distinct confirmed people")
    people = state.people.get("people", [])
    if not any(person.get("person_id") == source_person_id for person in people):
        raise KeyError(source_person_id)
    target = next((person for person in people if person.get("person_id") == target_person_id), None)
    if target is None:
        raise KeyError(target_person_id)
    source_faces = (state.review.get("person_faces") or {}).pop(source_person_id, [])
    target_faces = (state.review.get("person_faces") or {}).get(target_person_id, [])
    state.review["person_faces"][target_person_id] = list(dict.fromkeys([*target_faces, *source_faces]))
    state.people["people"] = [person for person in people if person.get("person_id") != source_person_id]
    state.review.setdefault("actions", []).append({"action": "merge-people", "face_ids": list(dict.fromkeys(source_faces)), "details": {"source_person_id": source_person_id, "target_person_id": target_person_id}, "created_at": datetime.now(timezone.utc).isoformat()})
    state.save()
    return dict(target)


_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


def _send_response_body(handler: BaseHTTPRequestHandler, status: int, content_type: str, data: bytes) -> None:
    """Send an HTTP response body, ignoring clients that disconnect mid-write."""
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except _CLIENT_DISCONNECT_ERRORS:
        return


def make_server(root: Path, workspace: Path, host: str = "127.0.0.1", port: int = 0):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("face review may only bind to loopback")
    with FaceStore(workspace):
        pass
    state = ReviewState(workspace)
    state_lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            with state_lock:
                try:
                    super().handle()
                except _CLIENT_DISCONNECT_ERRORS:
                    return

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            _send_response_body(self, status, "application/json", data)

        def send_jpeg(self, data: bytes):
            _send_response_body(self, 200, "image/jpeg", data)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                data = HTML.encode()
                _send_response_body(self, 200, "text/html; charset=utf-8", data)
                return
            if route == "/api/status":
                return self.send_json({"collection_id": state.review["collection_id"], "local_only": True})
            with FaceStore(workspace) as store:
                if route == "/api/people":
                    return self.send_json(_confirmed_people(store, state))
                parts = route.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["api", "people"]:
                    person = _confirmed_person(store, state, parts[2], include_faces=True)
                    return self.send_json(person) if person else self.send_json({"error": "person not found"}, 404)
                if route == "/api/groups":
                    return self.send_json(store.groups())
                if route == "/api/ungrouped":
                    return self.send_json(store.ungrouped())
                if len(parts) == 3 and parts[:2] == ["api", "groups"]:
                    group = store.group(parts[2])
                    return self.send_json(group) if group else self.send_json({"error": "group not found"}, 404)
                if len(parts) == 4 and parts[:2] == ["api", "faces"] and parts[3] == "thumbnail":
                    row = store.db.execute("SELECT face_id FROM face_occurrences WHERE face_id=?", (parts[2],)).fetchone()
                    path = workspace / "cache" / "face-thumbnails" / f"{parts[2]}.jpg"
                    return self.send_jpeg(path.read_bytes()) if row and path.exists() else self.send_json({"error": "face not found"}, 404)
                if len(parts) == 4 and parts[:2] == ["api", "images"] and parts[3] == "thumbnail":
                    row = store.db.execute("SELECT image_path FROM face_occurrences WHERE image_id=? LIMIT 1", (parts[2],)).fetchone()
                    if not row:
                        return self.send_json({"error": "image not found"}, 404)
                    try:
                        path = safe_collection_path(root, str(row["image_path"]))
                    except ValueError:
                        return self.send_json({"error": "image outside collection"}, 403)
                    try:
                        with Image.open(path) as image:
                            rendered = ImageOps.exif_transpose(image).convert("RGB")
                            rendered.thumbnail((1400, 1400))
                            output = io.BytesIO()
                            rendered.save(output, "JPEG", quality=88)
                        return self.send_jpeg(output.getvalue())
                    except (OSError, ValueError):
                        return self.send_json({"error": "image unavailable"}, 404)
                if len(parts) == 4 and parts[:2] == ["api", "images"] and parts[3] == "faces":
                    rows = store.db.execute("""SELECT face_id,x1,y1,x2,y2,confidence
                      FROM face_occurrences WHERE image_id=? ORDER BY face_id""", (parts[2],)).fetchall()
                    if not rows:
                        return self.send_json({"error": "image not found"}, 404)
                    return self.send_json([dict(row) for row in rows])
            return self.send_json({"error": "not found"}, 404)

        def _body(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            except (ValueError, TypeError):
                self.send_json({"error": "invalid JSON"}, 400)
                return None
            if body.get("collection_id") != state.review["collection_id"]:
                self.send_json({"error": "collection identity mismatch"}, 409)
                return None
            return body

        def do_POST(self):
            body = self._body()
            if body is None:
                return
            route = urlparse(self.path).path
            with FaceStore(workspace) as store:
                if route == "/api/people/merge":
                    try:
                        merged = _merge_people(state, str(body.get("source_person_id", "")), str(body.get("target_person_id", "")))
                    except KeyError:
                        return self.send_json({"error": "person not found"}, 404)
                    except ValueError as exc:
                        return self.send_json({"error": str(exc)}, 400)
                    person = _confirmed_person(store, state, merged["person_id"], include_faces=True) or merged
                    return self.send_json(person)
                if route == "/api/people":
                    group = store.group(str(body.get("group_id", "")))
                    if not group:
                        return self.send_json({"error": "group not found"}, 404)
                    if group["review_state"] != "reviewed" or group["conflict"]:
                        return self.send_json({"error": "group must be reviewed and conflict-free"}, 409)
                    try:
                        person = state.create_person([f["face_id"] for f in group["faces"]], body.get("display_name"), group.get("friendly_name") or group.get("friendly_id"))
                    except ValueError as exc:
                        return self.send_json({"error": str(exc)}, 409)
                    return self.send_json(person, 201)
                if route == "/api/review/create-group":
                    face_ids = body.get("face_ids", [])
                    if not isinstance(face_ids, list) or len(set(face_ids)) < 2:
                        return self.send_json({"error": "select at least two faces"}, 400)
                    unique_face_ids = list(dict.fromkeys(str(face_id) for face_id in face_ids))
                    placeholders = ",".join("?" for _ in unique_face_ids)
                    rows = store.db.execute(f"""SELECT f.face_id,f.image_id,f.backend_id,f.model_id,f.model_version,
                      f.preprocessing_version,f.embedding_dimensions,m.group_id
                      FROM face_occurrences f LEFT JOIN face_group_members m USING(face_id)
                      WHERE f.face_id IN ({placeholders}) AND f.excluded=0""", unique_face_ids).fetchall()
                    if len(rows) != len(unique_face_ids):
                        return self.send_json({"error": "all selected faces must exist and be usable"}, 400)
                    if any(row["group_id"] for row in rows):
                        return self.send_json({"error": "selected faces must be ungrouped"}, 409)
                    model_keys = {":".join(str(row[key]) for key in ("backend_id", "model_id", "model_version", "preprocessing_version", "embedding_dimensions")) for row in rows}
                    if len(model_keys) != 1:
                        return self.send_json({"error": "selected faces use incompatible face models"}, 409)
                    run_id = str(uuid.uuid4())
                    group_id = str(uuid.uuid4())
                    params = {"source": "manual-review", "action": "create-group"}
                    store.db.execute("INSERT INTO cluster_runs VALUES(?,?,?,?)", (run_id, sorted(model_keys)[0], json.dumps(params, sort_keys=True), datetime.now(timezone.utc).isoformat()))
                    store.db.execute("INSERT INTO face_groups VALUES(?,?,?,?,0)", (group_id, run_id, unique_face_ids[0], "needs_review"))
                    store.db.executemany("INSERT INTO face_group_members VALUES(?,?,?)", [(group_id, face_id, 1.0) for face_id in unique_face_ids])
                    _refresh_group(store, group_id)
                    store.db.commit()
                    state.record_action("must-link", unique_face_ids, group_id=group_id, source="create-group")
                    group = store.group(group_id) or {"group_id": group_id}
                    return self.send_json(group, 201)
                if route == "/api/review/merge":
                    source, target = body.get("source_group_id"), body.get("target_group_id")
                    if not source or not target or source == target:
                        return self.send_json({"error": "two distinct groups are required"}, 400)
                    if not store.group(source) or not store.group(target):
                        return self.send_json({"error": "group not found"}, 404)
                    face_ids = [r[0] for r in store.db.execute("SELECT face_id FROM face_group_members WHERE group_id IN (?,?)", (source, target))]
                    store.db.execute("UPDATE OR IGNORE face_group_members SET group_id=? WHERE group_id=?", (target, source))
                    store.db.execute("DELETE FROM face_group_members WHERE group_id=?", (source,))
                    store.db.execute("DELETE FROM face_groups WHERE group_id=?", (source,))
                    _refresh_group(store, target)
                    store.db.commit()
                    return self.send_json(state.record_action("must-link", face_ids, source_group_id=source, target_group_id=target))
                group_id = str(body.get("group_id", ""))
                group = store.group(group_id)
                if not group:
                    return self.send_json({"error": "group not found"}, 404)
                known = {face["face_id"] for face in group["faces"]}
                face_ids = body.get("face_ids", [])
                if not isinstance(face_ids, list) or not set(face_ids) <= known:
                    return self.send_json({"error": "face_ids must belong to the group"}, 400)
                if route == "/api/review/split":
                    if not face_ids or len(face_ids) == len(known):
                        return self.send_json({"error": "select some, but not all, group faces"}, 400)
                    new_group = str(uuid.uuid4())
                    store.db.execute("INSERT INTO face_groups VALUES(?,?,?,?,0)", (new_group, group["cluster_run_id"], face_ids[0], "unreviewed"))
                    store.db.executemany("UPDATE face_group_members SET group_id=? WHERE group_id=? AND face_id=?", [(new_group, group_id, face_id) for face_id in face_ids])
                    _refresh_group(store, group_id)
                    _refresh_group(store, new_group)
                    store.db.commit()
                    return self.send_json(state.record_action("cannot-link", face_ids, group_id=group_id, new_group_id=new_group))
                if route in {"/api/review/reject-face", "/api/review/exclude-from-clustering"}:
                    if not face_ids:
                        return self.send_json({"error": "select at least one face"}, 400)
                    store.db.executemany("DELETE FROM face_group_members WHERE group_id=? AND face_id=?", [(group_id, face_id) for face_id in face_ids])
                    if route.endswith("exclude-from-clustering"):
                        store.db.executemany("UPDATE face_occurrences SET excluded=1 WHERE face_id=?", [(face_id,) for face_id in face_ids])
                    _refresh_group(store, group_id)
                    store.db.commit()
                    action = "reject-face" if route.endswith("reject-face") else "exclude-from-clustering"
                    return self.send_json(state.record_action(action, face_ids, group_id=group_id))
                if route == "/api/review/mark-group-reviewed":
                    if group["conflict"]:
                        return self.send_json({"error": "resolve the same-photograph conflict first"}, 409)
                    store.db.execute("UPDATE face_groups SET review_state='reviewed' WHERE group_id=?", (group_id,))
                    store.db.commit()
                    return self.send_json(state.record_action("mark-group-reviewed", list(known), group_id=group_id))
            return self.send_json({"error": "not found"}, 404)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve_review(root: Path, workspace: Path, host="127.0.0.1", port=0, open_browser=True):
    server = make_server(root, workspace, host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Face review: {url}\nPress Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
