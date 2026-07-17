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

HTML = r"""<!doctype html><meta charset="utf-8"><title>KiokuFux face review</title>
<style>
body{font:16px system-ui;max-width:1200px;margin:auto;padding:2rem;background:#181818;color:#eee}button,select,input{padding:.55rem;margin:.2rem}.toolbar{position:sticky;top:0;background:#181818;padding:.5rem 0;z-index:2}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:1rem}.card{background:#292929;padding:1rem;border-radius:8px;cursor:pointer}.card img,.face img{width:100%;aspect-ratio:1;object-fit:cover}.face{background:#292929;padding:.6rem;border:2px solid transparent}.face.selected{border-color:#6cf}.face input{position:absolute}.muted{color:#aaa}.conflict{color:#ff9b7a}dialog{width:min(1150px,94vw);background:#222;color:#eee;border:1px solid #555}.comparison{display:grid;grid-template-columns:minmax(180px,1fr) minmax(0,3fr);gap:1rem;align-items:start}.crop-pane img{width:100%;aspect-ratio:1;object-fit:cover}.photo-wrap{position:relative;display:inline-block;max-width:100%}.photo-wrap>img{display:block;max-width:100%;max-height:78vh}.face-box{position:absolute;border:3px solid #ffd54f;box-sizing:border-box;pointer-events:none}.face-box.active{border-color:#55e7ff;box-shadow:0 0 0 2px #111}.face-box span{position:absolute;top:-1.5rem;left:-3px;background:#111;color:#fff;padding:.1rem .3rem;font-size:.75rem;white-space:nowrap}@media(max-width:700px){.comparison{grid-template-columns:1fr}.crop-pane{max-width:220px}}.hidden{display:none}
</style>
<header><h1>Face review</h1><p>These are possible recurring people, not identities. Select occurrences to correct a group, then explicitly review and confirm it.</p></header>
<section id="list"><div class="toolbar"><button onclick="showGroups()">Groups</button><button onclick="showUngrouped()">Ungrouped</button></div><div id="groups" class="grid"></div></section>
<section id="detail" class="hidden"><div class="toolbar"><button onclick="showGroups()">← All groups</button><button onclick="act('split')">Split selected</button><button onclick="act('reject-face')">Reject detection</button><button onclick="act('exclude-from-clustering')">Exclude poor crop</button><select id="mergeTarget"></select><button onclick="mergeGroup()">Merge group</button><button onclick="reviewGroup()">Mark reviewed</button><button onclick="confirmPerson()">Confirm as person</button></div><h2 id="groupTitle"></h2><p class="muted">Click a face to select it. Click “View photograph” to inspect the complete source image.</p><div id="faces" class="grid"></div></section>
<dialog id="context"><button onclick="context.close()">Close</button><h2>Face in photograph</h2><div class="comparison"><section class="crop-pane"><h3>Selected face</h3><img id="contextCrop"><p class="muted">The blue box marks this occurrence.</p></section><section><h3>Original photograph</h3><div class="photo-wrap" id="photoWrap"><img id="contextImage"></div><p class="muted">All detected faces are marked. Yellow boxes are other occurrences in this photograph.</p></section></div></dialog>
<script>
let collectionId, currentGroup, allGroups=[];
const api=async(path,options={})=>{let r=await fetch(path,options);let data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data};
const mutate=(path,body={})=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({collection_id:collectionId,...body})});
async function start(){collectionId=(await api('/api/status')).collection_id;await showGroups()}
async function showGroups(){detail.classList.add('hidden');list.classList.remove('hidden');allGroups=await api('/api/groups');groups.innerHTML=allGroups.length?allGroups.map(g=>`<article class="card" onclick="openGroup('${g.group_id}')"><img src="/api/faces/${g.representative_face_id}/thumbnail"><b>${g.face_count} face occurrences</b><p>${g.photo_count} photographs · <span class="${g.conflict?'conflict':''}">${g.conflict?'conflict':g.review_state}</span></p><small>${g.group_id.slice(0,8)}</small></article>`).join(''):'<p>No recurring groups yet.</p>'}
async function showUngrouped(){let fs=await api('/api/ungrouped');groups.innerHTML=fs.length?fs.map(faceCard).join(''):'<p>No ungrouped faces.</p>'}
function faceCard(f){return `<article class="face" data-id="${f.face_id}" onclick="toggleFace(event,this)"><input type="checkbox"><img src="/api/faces/${f.face_id}/thumbnail"><button onclick="viewContext(event,'${f.image_id}','${f.face_id}')">View in photograph</button><small>confidence ${(f.confidence*100).toFixed(1)}%</small></article>`}
async function openGroup(id){currentGroup=await api('/api/groups/'+id);list.classList.add('hidden');detail.classList.remove('hidden');groupTitle.textContent=`Possible recurring person · ${currentGroup.faces.length} occurrences${currentGroup.conflict?' · conflict':''}`;faces.innerHTML=currentGroup.faces.map(faceCard).join('');mergeTarget.innerHTML='<option value="">Merge with…</option>'+allGroups.filter(g=>g.group_id!==id).map(g=>`<option value="${g.group_id}">${g.group_id.slice(0,8)} · ${g.face_count} faces</option>`).join('')}
function toggleFace(e,el){if(e.target.tagName==='BUTTON')return;el.classList.toggle('selected');el.querySelector('input').checked=el.classList.contains('selected')}
const selected=()=>[...document.querySelectorAll('.face.selected')].map(x=>x.dataset.id);
async function act(name){let ids=selected();if(!ids.length)return alert('Select at least one face.');try{await mutate('/api/review/'+name,{group_id:currentGroup.group_id,face_ids:ids});await openGroup(currentGroup.group_id)}catch(e){alert(e.message)}}
async function mergeGroup(){if(!mergeTarget.value)return alert('Choose another group.');try{await mutate('/api/review/merge',{source_group_id:currentGroup.group_id,target_group_id:mergeTarget.value});await showGroups()}catch(e){alert(e.message)}}
async function reviewGroup(){try{await mutate('/api/review/mark-group-reviewed',{group_id:currentGroup.group_id});await openGroup(currentGroup.group_id)}catch(e){alert(e.message)}}
async function confirmPerson(){let name=prompt('Optional display name (leave blank to remain unnamed):','');if(name===null)return;try{await mutate('/api/people',{group_id:currentGroup.group_id,display_name:name||null});alert('Confirmed as a person.');await showGroups()}catch(e){alert(e.message)}}
async function viewContext(e,imageId,faceId){e.stopPropagation();contextCrop.src='/api/faces/'+faceId+'/thumbnail';contextImage.src='/api/images/'+imageId+'/thumbnail';photoWrap.querySelectorAll('.face-box').forEach(x=>x.remove());try{let detections=await api('/api/images/'+imageId+'/faces');detections.forEach((f,i)=>{let box=document.createElement('div');box.className='face-box'+(f.face_id===faceId?' active':'');box.style.left=(f.x1*100)+'%';box.style.top=(f.y1*100)+'%';box.style.width=((f.x2-f.x1)*100)+'%';box.style.height=((f.y2-f.y1)*100)+'%';box.innerHTML='<span>'+(f.face_id===faceId?'selected face':'face '+(i+1))+'</span>';photoWrap.appendChild(box)})}catch(err){alert(err.message)}context.showModal()}
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
                    path = Path(row["image_path"]).resolve()
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
                    return self.send_json(state.create_person([f["face_id"] for f in group["faces"]], body.get("display_name")), 201)
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
