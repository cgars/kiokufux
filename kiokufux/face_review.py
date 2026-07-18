from __future__ import annotations

import io
import json
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps

from .faces import FaceStore, ReviewState

HTML = r"""<!doctype html><meta charset="utf-8"><title>KiokuFux · People</title>
<style>
:root{color-scheme:dark;--bg:#171512;--panel:#211f1b;--panel-2:#2a2722;--line:#3a352d;--text:#f3eee4;--muted:#b8ad9e;--blue:#74a7c6;--blue-strong:#8cc9ee;--amber:#d6a34c;--danger:#d56f63;--shadow:0 18px 50px #0008}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#24211c,var(--bg) 42rem);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}.app-shell{min-height:100vh;display:grid;grid-template-rows:auto 1fr}.topbar{height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--line);background:#191713e6;backdrop-filter:blur(18px);position:sticky;top:0;z-index:10}.brand{font-weight:720;letter-spacing:.02em}.settings{color:var(--muted)}.tabs{display:flex;gap:6px;padding:14px 28px 0}.tab{border:0;background:transparent;color:var(--muted);border-radius:999px;padding:.65rem .95rem;font-weight:650}.tab.active,.tab:hover{background:#2b332f;color:var(--text)}button,select,input{font:inherit}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:24px;padding:22px 28px 36px;max-width:1500px;width:100%;margin:0 auto}.content-card,.actions-panel{background:linear-gradient(180deg,#25221e,#1f1d19);border:1px solid var(--line);border-radius:22px;box-shadow:var(--shadow)}.content-card{padding:24px}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:start;margin-bottom:20px}.eyebrow,.meta-label{text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:var(--muted);font-weight:750}.title{font-size:clamp(1.8rem,3vw,3.2rem);line-height:1.05;margin:.25rem 0}.subtitle{color:var(--muted);margin:0}.metadata{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}.pill{display:inline-flex;align-items:center;gap:.35rem;padding:.35rem .65rem;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:#171512}.pill.warning{color:#ffd589;border-color:#7b5b24;background:#2a2114}.group-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}.group-card{border:1px solid var(--line);border-radius:18px;background:#28241f;padding:14px;cursor:pointer;transition:.16s ease;display:grid;gap:12px}.group-card:hover{transform:translateY(-2px);border-color:#566b69}.group-card img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:14px}.group-card b{font-size:1.05rem}.face-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px}.face{position:relative;border:1px solid var(--line);border-radius:18px;background:#28241f;padding:10px;cursor:pointer;transition:.16s ease}.face:hover{border-color:#566b69}.face.selected{border-color:var(--blue-strong);box-shadow:0 0 0 3px #74a7c633;background:#243039}.face img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:14px;display:block}.face input{position:absolute;opacity:0;pointer-events:none}.checkmark{position:absolute;top:18px;right:18px;width:30px;height:30px;border-radius:999px;background:#111b;border:1px solid #fff4;display:grid;place-items:center;color:transparent}.face.selected .checkmark{background:var(--blue-strong);color:#10212c}.face-caption{display:flex;align-items:center;justify-content:space-between;margin-top:10px;color:var(--muted);font-size:.86rem}.quality-badge{border:1px solid var(--line);border-radius:999px;padding:.15rem .45rem}.quality-badge.low{color:#ffd589;border-color:#805f2d}.actions-panel{position:sticky;top:92px;align-self:start;padding:18px;display:grid;gap:12px}.actions-panel h2{font-size:1rem;margin:0 0 .25rem}.action-stack{display:grid;gap:8px}.btn{border:1px solid var(--line);background:#302c26;color:var(--text);border-radius:12px;padding:.72rem .85rem;text-align:left;cursor:pointer}.btn:hover{border-color:#5b6e6e}.btn.primary{background:linear-gradient(180deg,#86bad7,#669dbd);border-color:#9bcde8;color:#10212c;font-weight:800;text-align:center}.btn.secondary{text-align:center}.btn.danger{color:#ffd9d5;border-color:#6f3731;background:#33201e}.merge-select{width:100%;border:1px solid var(--line);background:#191713;color:var(--text);border-radius:12px;padding:.72rem}.danger-menu{border-top:1px solid var(--line);padding-top:12px}.danger-menu summary{cursor:pointer;color:var(--muted);font-weight:700}.empty{color:var(--muted);padding:24px;border:1px dashed var(--line);border-radius:16px}.hint{color:var(--muted);font-size:.92rem}.hidden{display:none!important}dialog{width:min(1500px,96vw);max-height:92vh;padding:0;border:1px solid var(--line);border-radius:24px;background:#201d19;color:var(--text);box-shadow:var(--shadow)}dialog::backdrop{background:#050505aa}.dialog-head{height:60px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid var(--line)}.compare-items{display:grid;gap:18px;padding:18px}.compare-item{display:grid;grid-template-columns:180px minmax(0,1fr);gap:18px;min-height:65vh}.reference-panel{background:#171512;border:1px solid var(--line);border-radius:18px;padding:14px;align-self:start}.reference-panel img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:14px}.photo-panel{min-width:0}.canvas-toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}.zoom-controls{display:flex;align-items:center;gap:6px}.zoom-controls button{border:1px solid var(--line);background:#2a2722;color:var(--text);border-radius:10px;padding:.45rem .65rem}.zoom-controls output{min-width:3.5rem;text-align:center;color:var(--muted)}.zoom-viewport{width:100%;max-height:72vh;overflow:auto;background:#0f0e0c;border:1px solid #111;border-radius:18px;overscroll-behavior:contain}.photo-wrap{position:relative;width:100%;line-height:0;margin:auto}.photo-wrap>img{display:block;width:100%;height:auto;border-radius:18px}.face-box{position:absolute;border:3px solid var(--amber);box-sizing:border-box;pointer-events:none;border-radius:6px}.face-box.active{border-color:var(--blue-strong);box-shadow:0 0 0 2px #111}.face-box span{position:absolute;top:-1.8rem;left:0;background:#15130f;color:#fff;padding:.15rem .4rem;border-radius:999px;font-size:.72rem;line-height:normal;white-space:nowrap}@media(max-width:900px){.workspace{grid-template-columns:1fr}.actions-panel{position:static;order:-1}.compare-item{grid-template-columns:1fr}.reference-panel{max-width:220px}}
</style>
<div class="app-shell"><header class="topbar"><div class="brand">KiokuFux · People</div><div class="settings">Local archive workbench</div></header><nav class="tabs" aria-label="Face review sections"><button class="tab active" onclick="showGroups()">Groups</button><button class="tab" onclick="showUngrouped()">Ungrouped</button><button class="tab" onclick="showPlaceholder('Confirmed')">Confirmed</button><button class="tab" onclick="showPlaceholder('Needs review')">Needs review</button></nav><main class="workspace"><section class="content-card"><div id="list"><div class="section-head"><div><div class="eyebrow">Anonymous discovery</div><h1 class="title">Possible recurring people</h1><p class="subtitle">Review machine-generated groups without turning them into identities.</p></div><div class="metadata"><span class="pill">Local only</span><span class="pill">No names proposed</span></div></div><div id="groups" class="group-list"></div></div><div id="detail" class="hidden"><div class="section-head"><div><button class="btn" onclick="showGroups()">← All groups</button><div class="eyebrow" style="margin-top:18px">Possible recurring person</div><h1 id="groupTitle" class="title"></h1><p id="groupMeta" class="subtitle"></p></div><div class="metadata" id="groupBadges"></div></div><p class="hint">Click a face to see it in the source photograph. Select one or more faces for comparison and corrections.</p><div id="faces" class="face-grid"></div></div></section><aside class="actions-panel" id="actions"><h2>Actions</h2><p class="hint" id="selectionHint">Open a group to review its face occurrences.</p><div class="action-stack"><button class="btn primary" onclick="confirmPerson()">Confirm person</button><button class="btn secondary" onclick="compareSelected()">Compare selected <span class="shortcut">C</span></button><button class="btn secondary" onclick="act('split')">Split selected <span class="shortcut">S</span></button><select class="merge-select" id="mergeTarget" aria-label="Merge with another group"></select><button class="btn secondary" onclick="mergeGroup()">Merge into selected group</button><button class="btn secondary" onclick="reviewGroup()">Mark group reviewed <span class="shortcut">R</span></button></div><details class="danger-menu"><summary>More actions</summary><button class="btn danger" onclick="act('reject-face')">Reject detection</button><button class="btn danger" onclick="act('exclude-from-clustering')">Exclude poor crop</button></details></aside></main></div>
<dialog id="context"><div class="dialog-head"><h2 id="compareTitle">Face comparison</h2><button class="btn" onclick="context.close()">Close</button></div><div id="compareItems" class="compare-items"></div></dialog>
<script>
let collectionId, currentGroup, allGroups=[];
const api=async(path,options={})=>{let r=await fetch(path,options);let data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data};
const mutate=(path,body={})=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({collection_id:collectionId,...body})});
function setActiveTab(label){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.textContent.trim()===label))}
async function start(){collectionId=(await api('/api/status')).collection_id;await showGroups()}
async function showGroups(){setActiveTab('Groups');detail.classList.add('hidden');list.classList.remove('hidden');allGroups=await api('/api/groups');groups.className='group-list';groups.innerHTML=allGroups.length?allGroups.map(groupCard).join(''):'<p class="empty">No recurring groups yet.</p>';selectionHint.textContent='Open a group to review its face occurrences.';mergeTarget.innerHTML='<option value="">Merge with…</option>'}
function groupCard(g){let state=g.conflict?'Conflict':g.review_state.replace('_',' ');return `<article class="group-card" onclick="openGroup('${g.group_id}')"><img src="/api/faces/${g.representative_face_id}/thumbnail" alt="Representative face for ${g.friendly_id}"><div><b>${g.friendly_id}</b><div class="hint">Possible recurring person</div></div><div class="metadata"><span class="pill">${g.photo_count} photos</span><span class="pill">${g.face_count} occurrences</span><span class="pill ${g.conflict?'warning':''}">${state}</span></div></article>`}
async function showUngrouped(){setActiveTab('Ungrouped');detail.classList.add('hidden');list.classList.remove('hidden');groups.className='face-grid';let fs=await api('/api/ungrouped');groups.innerHTML=fs.length?fs.map(faceCard).join(''):'<p class="empty">No ungrouped faces.</p>';selectionHint.textContent='Ungrouped faces can be inspected in context.';mergeTarget.innerHTML='<option value="">Merge with…</option>'}
function showPlaceholder(label){setActiveTab(label);detail.classList.add('hidden');list.classList.remove('hidden');groups.className='group-list';groups.innerHTML=`<p class="empty">${label} will appear here as review decisions are available.</p>`;selectionHint.textContent='Choose Groups or Ungrouped to continue reviewing.'}
function confidenceBadge(f){let value=Math.round((f.confidence||0)*100);let low=value<90;return `<span class="quality-badge ${low?'low':''}">${low?'Low confidence '+value+'%':'Clear crop'}</span>`}
function faceCard(f){return `<article class="face" data-id="${f.face_id}" onclick="faceClick(event,this,'${f.image_id}','${f.face_id}')"><input type="checkbox" aria-label="Select face"><span class="checkmark">✓</span><img src="/api/faces/${f.face_id}/thumbnail" alt="Detected face"><div class="face-caption"><span>Open photograph</span>${confidenceBadge(f)}</div></article>`}
async function openGroup(id){currentGroup=await api('/api/groups/'+id);list.classList.add('hidden');detail.classList.remove('hidden');groupTitle.textContent=currentGroup.friendly_id;groupMeta.textContent=`${currentGroup.faces.length} occurrences · possible recurring person`;groupBadges.innerHTML=`<span class="pill">${currentGroup.review_state.replace('_',' ')}</span>${currentGroup.conflict?'<span class="pill warning">same-photo conflict</span>':''}`;faces.innerHTML=currentGroup.faces.map(faceCard).join('');mergeTarget.innerHTML='<option value="">Merge with…</option>'+allGroups.filter(g=>g.group_id!==id).map(g=>`<option value="${g.group_id}">${g.friendly_id} · ${g.face_count} faces</option>`).join('');selectionHint.textContent='No faces selected.'}
function faceClick(e,el,imageId,faceId){if(e.target.closest('.checkmark')||e.shiftKey||e.metaKey||e.ctrlKey){toggleFace(el);return}viewContext(e,imageId,faceId)}
function toggleFace(el){el.classList.toggle('selected');el.querySelector('input').checked=el.classList.contains('selected');let count=selected().length;selectionHint.textContent=count?`${count} selected. Press C to compare, S to split, R to mark reviewed.`:'No faces selected.'}
const selected=()=>[...document.querySelectorAll('.face.selected')].map(x=>x.dataset.id);
async function act(name){let ids=selected();if(!ids.length)return alert('Select at least one face.');try{await mutate('/api/review/'+name,{group_id:currentGroup.group_id,face_ids:ids});await openGroup(currentGroup.group_id)}catch(e){alert(e.message)}}
async function mergeGroup(){if(!mergeTarget.value)return alert('Choose another group.');try{await mutate('/api/review/merge',{source_group_id:currentGroup.group_id,target_group_id:mergeTarget.value});await showGroups()}catch(e){alert(e.message)}}
async function reviewGroup(){try{await mutate('/api/review/mark-group-reviewed',{group_id:currentGroup.group_id});await openGroup(currentGroup.group_id)}catch(e){alert(e.message)}}
async function confirmPerson(){if(!currentGroup)return alert('Open a reviewed group first.');let name=prompt('Optional display name (leave blank to remain unnamed):','');if(name===null)return;try{await mutate('/api/people',{group_id:currentGroup.group_id,display_name:name||null});alert('Confirmed as a person.');await showGroups()}catch(e){alert(e.message)}}
function applyZoom(item,scale){scale=Math.max(.5,Math.min(5,scale));let wrap=item.querySelector('.photo-wrap');wrap.dataset.scale=scale;wrap.style.width=(scale*100)+'%';item.querySelector('.zoom-value').value=Math.round(scale*100)+'%'}
function zoomBy(button,factor){let item=button.closest('.compare-item'),wrap=item.querySelector('.photo-wrap');applyZoom(item,(Number(wrap.dataset.scale)||1)*factor)}
function resetZoom(button){let item=button.closest('.compare-item');applyZoom(item,1);item.querySelector('.zoom-viewport').scrollTo(0,0)}
function wheelZoom(event){if(!event.ctrlKey&&!event.metaKey)return;event.preventDefault();let item=event.currentTarget.closest('.compare-item'),wrap=item.querySelector('.photo-wrap');applyZoom(item,(Number(wrap.dataset.scale)||1)*(event.deltaY<0?1.2:1/1.2))}
async function comparisonItem(face,index){let item=document.createElement('article');item.className='compare-item';item.innerHTML=`<aside class="reference-panel"><div class="eyebrow">Selected face</div><h3>Occurrence ${index+1}</h3><img src="/api/faces/${face.face_id}/thumbnail"><p class="hint">Blue outline marks this face. Amber outlines mark other detections.</p></aside><section class="photo-panel"><div class="canvas-toolbar"><b>Source photograph</b><div class="zoom-controls"><button onclick="resetZoom(this)">Fit</button><button onclick="resetZoom(this)">100%</button><button onclick="zoomBy(this,1/1.25)" aria-label="Zoom out">−</button><output class="zoom-value">100%</output><button onclick="zoomBy(this,1.25)" aria-label="Zoom in">+</button></div></div><div class="zoom-viewport" onwheel="wheelZoom(event)"><div class="photo-wrap" data-scale="1"><img src="/api/images/${face.image_id}/thumbnail"></div></div></section>`;compareItems.appendChild(item);let wrap=item.querySelector('.photo-wrap');let detections=await api('/api/images/'+face.image_id+'/faces');detections.forEach((f,i)=>{let box=document.createElement('div');box.className='face-box'+(f.face_id===face.face_id?' active':'');box.style.left=(f.x1*100)+'%';box.style.top=(f.y1*100)+'%';box.style.width=((f.x2-f.x1)*100)+'%';box.style.height=((f.y2-f.y1)*100)+'%';box.innerHTML='<span>'+(f.face_id===face.face_id?'selected face':'face '+(i+1))+'</span>';wrap.appendChild(box)})}
async function showComparison(items){compareItems.innerHTML='';compareTitle.textContent=items.length===1?'Photograph context':`${items.length} selected occurrences side by side`;try{await Promise.all(items.map(comparisonItem));context.showModal()}catch(err){alert(err.message)}}
function compareSelected(){let ids=selected();if(ids.length<2)return alert('Select at least two faces to compare.');showComparison(currentGroup.faces.filter(f=>ids.includes(f.face_id)))}
function viewContext(e,imageId,faceId){e.stopPropagation();let face=(currentGroup?.faces||[]).find(f=>f.face_id===faceId)||{image_id:imageId,face_id:faceId};showComparison([face])}
document.addEventListener('keydown',event=>{if(event.target.matches('input,select,textarea'))return;if(event.key.toLowerCase()==='c')compareSelected();if(event.key.toLowerCase()==='s')act('split');if(event.key.toLowerCase()==='r')reviewGroup();if(event.key==='Escape'&&context.open)context.close()});
start().catch(e=>alert(e.message));
</script>"""


def safe_collection_path(root: Path, candidate: str) -> Path:
    resolved = (root / candidate).resolve()
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
                super().handle()

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_jpeg(self, data: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                data = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route == "/api/status":
                return self.send_json({"collection_id": state.review["collection_id"], "local_only": True})
            with FaceStore(workspace) as store:
                if route == "/api/groups":
                    return self.send_json(store.groups())
                if route == "/api/ungrouped":
                    return self.send_json(store.ungrouped())
                parts = route.strip("/").split("/")
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
                    stored_path = Path(row["image_path"])
                    path = (root / stored_path).resolve() if not stored_path.is_absolute() else stored_path.resolve()
                    if not path.is_relative_to(root.resolve()):
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
