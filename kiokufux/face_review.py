from __future__ import annotations

import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps

from .faces import FaceStore, ReviewState

HTML = """<!doctype html><meta charset=utf-8><title>KiokuFux face review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem;background:#181818;color:#eee}button{padding:.5rem}#groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}.card{background:#292929;padding:1rem;border-radius:8px}.card img{width:100%;aspect-ratio:1;object-fit:cover}</style>
<h1>Possible recurring people</h1><p>Groups are anonymous machine suggestions until you explicitly review and confirm one.</p><div id=groups></div>
<script>fetch('/api/groups').then(r=>r.json()).then(gs=>groups.innerHTML=gs.map(g=>`<article class=card><img src="/api/faces/${g.representative_face_id}/thumbnail"><b>${g.face_count} face occurrences</b><p>${g.photo_count} photographs · ${g.conflict?'conflict':g.review_state}</p></article>`).join(''))</script>"""

def safe_collection_path(root: Path, candidate: str) -> Path:
    resolved=(root/candidate).resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("path leaves collection")
    return resolved

def make_server(root:Path, workspace:Path, host:str="127.0.0.1", port:int=0):
    if host not in {"127.0.0.1","::1","localhost"}: raise ValueError("face review may only bind to loopback")
    store=FaceStore(workspace); state=ReviewState(workspace)
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,value,status=200):
            data=json.dumps(value).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            route=urlparse(self.path).path
            if route=="/":
                data=HTML.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data); return
            if route=="/api/status": return self.send_json({"collection_id":state.review["collection_id"],"local_only":True})
            if route=="/api/groups": return self.send_json(store.groups())
            parts=route.strip('/').split('/')
            if len(parts)==4 and parts[:2]==["api","faces"] and parts[3]=="thumbnail":
                face_id=parts[2]; row=store.db.execute("SELECT face_id FROM face_occurrences WHERE face_id=?",(face_id,)).fetchone()
                if not row:return self.send_json({"error":"not found"},404)
                path=workspace/"cache"/"face-thumbnails"/f"{face_id}.jpg"
                if not path.exists():return self.send_json({"error":"not found"},404)
                data=path.read_bytes(); self.send_response(200); self.send_header("Content-Type","image/jpeg"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
            return self.send_json({"error":"not found"},404)
        def do_POST(self):
            length=int(self.headers.get("Content-Length","0"));
            try: body=json.loads(self.rfile.read(length) or b"{}")
            except ValueError:return self.send_json({"error":"invalid JSON"},400)
            if body.get("collection_id")!=state.review["collection_id"]:return self.send_json({"error":"collection identity mismatch"},409)
            if self.path=="/api/people":
                try: return self.send_json(state.create_person(body.get("face_ids",[]),body.get("display_name")),201)
                except (TypeError,ValueError) as exc:return self.send_json({"error":str(exc)},400)
            actions={"/api/review/reject-face":"reject-face","/api/review/exclude-from-clustering":"exclude-from-clustering",
                     "/api/review/merge":"must-link","/api/review/split":"cannot-link","/api/review/mark-group-reviewed":"mark-group-reviewed"}
            if self.path in actions:
                face_ids=body.get("face_ids",[])
                if not isinstance(face_ids,list) or not all(isinstance(x,str) for x in face_ids):return self.send_json({"error":"face_ids must be a list of strings"},400)
                known={r[0] for r in store.db.execute("SELECT face_id FROM face_occurrences")}
                if not set(face_ids)<=known:return self.send_json({"error":"unknown face_id"},404)
                if actions[self.path]=="exclude-from-clustering" and face_ids:
                    store.db.executemany("UPDATE face_occurrences SET excluded=1 WHERE face_id=?",[(x,) for x in face_ids]);store.db.commit()
                return self.send_json(state.record_action(actions[self.path],face_ids,group_id=body.get("group_id")))
            if self.path=="/api/review/undo": return self.send_json(state.undo())
            return self.send_json({"error":"not found"},404)
        def do_PATCH(self):
            length=int(self.headers.get("Content-Length","0"))
            try: body=json.loads(self.rfile.read(length) or b"{}")
            except ValueError:return self.send_json({"error":"invalid JSON"},400)
            if body.get("collection_id")!=state.review["collection_id"]:return self.send_json({"error":"collection identity mismatch"},409)
            parts=urlparse(self.path).path.strip('/').split('/')
            if len(parts)==3 and parts[:2]==["api","people"]:
                try:return self.send_json(state.rename_person(parts[2],body.get("display_name")))
                except KeyError:return self.send_json({"error":"person not found"},404)
            return self.send_json({"error":"not found"},404)
        def log_message(self,*_): pass
    server=ThreadingHTTPServer((host,port),Handler); server.face_store=store
    return server

def serve_review(root:Path, workspace:Path, host="127.0.0.1",port=0,open_browser=True):
    server=make_server(root,workspace,host,port); actual=server.server_address[1]; url=f"http://{host}:{actual}/"
    print(f"Face review: {url}\nPress Ctrl+C to stop.")
    if open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.shutdown(); server.server_close(); server.face_store.close()
