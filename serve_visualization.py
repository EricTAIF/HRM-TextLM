import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    ThreadingHTTPServer = HTTPServer  # fallback
from threading import Thread

import torch
from transformers import AutoTokenizer

from utils.functions import load_model_class
import yaml
import json as pyjson


_MODEL_CACHE = {}


def build_model(checkpoint: str, tokenizer_name: str, overrides=None, tag: str = "gen"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_key = (tokenizer_name,)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ckpt_path = os.path.abspath(checkpoint)
    ckpt_dir = os.path.dirname(ckpt_path)

    cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config next to checkpoint: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    meta = {}
    try:
        with open(os.path.join(cfg["data_path"], "meta.json"), "r") as mf:
            meta = pyjson.load(mf)
    except Exception:
        pass

    arch = cfg["arch"]
    model_cls = load_model_class(arch["name"])  # models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1
    loss_head_cls = load_model_class(arch["loss"]["name"])  # losses@ACTLossHead

    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg.update({
        "batch_size": 1,
        "seq_len": meta.get("block_size", 1024),
        "vocab_size": meta.get("vocab_size", tok.vocab_size),
        "num_puzzle_identifiers": 1,
        "task": "text_lm",
        "pad_token_id": meta.get("pad_token_id", tok.pad_token_id),
        "forward_dtype": "bfloat16" if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else "float16",
    })

    # Apply optional runtime overrides (e.g., force more H/L cycles for visualization)
    overrides = overrides or {}
    for k in ("H_cycles", "L_cycles", "halt_max_steps", "halt_exploration_prob"):
        if k in overrides and overrides[k] is not None:
            model_cfg[k] = overrides[k]

    cache_key = (tag, ckpt_path, tokenizer_name, model_cfg.get("H_cycles"), model_cfg.get("L_cycles"), model_cfg.get("halt_max_steps"))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with torch.device(device):
        base = model_cls(model_cfg)
        model = loss_head_cls(base, **{k: v for k, v in arch["loss"].items() if k != "name"})
        model = model.to(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and not any(isinstance(v, torch.nn.Module) for v in state.values()):
            model.load_state_dict(state, strict=False)
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], torch.nn.Module):
            model.load_state_dict(state["model"].state_dict(), strict=False)
        else:
            raise ValueError("Unsupported checkpoint format")
        model.eval()
    _MODEL_CACHE[cache_key] = (model, tok)
    return _MODEL_CACHE[cache_key]


def sse_headers(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()


def stream_generation(handler, params):
    checkpoint = params.get("checkpoint", [""])[0]
    prompt = params.get("prompt", [""])[0]
    tokenizer_name = params.get("tokenizer", ["mistralai/Mistral-7B-v0.1"])[0]
    max_new_tokens = int(params.get("max_new_tokens", ["64"])[0])
    temperature = float(params.get("temperature", ["1.0"])[0])
    top_p = float(params.get("top_p", ["0.9"])[0])
    greedy = params.get("greedy", ["false"])[0].lower() == "true"
    delay_ms = float(params.get("delay_ms", ["0"])[0])

    h_cycles = int(params.get("h_cycles", ["0"])[0]) if params.get("h_cycles") else None
    l_cycles = int(params.get("l_cycles", ["0"])[0]) if params.get("l_cycles") else None
    viz_override = params.get("viz_override", ["false"])[0].lower() == "true"

    try:
        # Generation model: always original config (no overrides) for best quality
        gen_model, tok = build_model(checkpoint, tokenizer_name, overrides=None, tag="gen")
        # Visualization model: optionally apply H/L overrides for richer trace
        if viz_override:
            viz_model, _ = build_model(checkpoint, tokenizer_name, overrides={"H_cycles": h_cycles, "L_cycles": l_cycles}, tag="viz")
        else:
            viz_model = gen_model
    except Exception as e:
        payload = {"type": "error", "message": str(e)}
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        return

    device = next(gen_model.parameters()).device
    x = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    generated = x.clone()

    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                # 1) Get logits from generation model (no overrides)
                gen_batch = {"input_ids": generated, "attention_mask": torch.ones_like(generated)}
                gen_carry = gen_model.initial_carry(gen_batch)
                gen_carry, _, _, gen_outs, _ = gen_model(carry=gen_carry, batch=gen_batch, return_keys=["logits"])  # type: ignore
                logits = gen_outs["logits"]  # [1, T, V]

                cur_len = generated.size(1)
                try:
                    max_ctx = getattr(getattr(gen_model, "model", gen_model), "inner").config.seq_len  # type: ignore[attr-defined]
                except Exception:
                    max_ctx = cur_len
                last_idx = min(cur_len, max_ctx) - 1

                step_logits = logits[:, last_idx, :] / max(1e-6, temperature)
                probs = torch.softmax(step_logits, dim=-1)
                topk = 10
                top_probs, top_idx = torch.topk(probs, k=topk, dim=-1)
                top_probs = top_probs[0].tolist()
                top_idx = top_idx[0].tolist()
                top_tokens = [tok.decode([i]) for i in top_idx]

                if greedy or temperature <= 0 or top_p <= 0:
                    next_id = torch.argmax(step_logits, dim=-1)
                else:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cumsum > top_p).float().argmax(dim=-1).item()
                    keep_idx = sorted_idx[:, :cutoff + 1]
                    keep_probs = torch.gather(probs, 1, keep_idx)
                    keep_probs = keep_probs / keep_probs.sum(dim=-1, keepdim=True)
                    next_id = keep_idx[0, torch.multinomial(keep_probs[0], 1)]

                generated = torch.cat([generated, next_id.view(1, 1)], dim=1)

                entropy = (-probs * (probs.clamp_min(1e-9)).log()).sum().item()
                # 2) Stream trace frames from viz model (can be overridden for richer pondering)
                viz_batch = {"input_ids": generated, "attention_mask": torch.ones_like(generated), "debug_trace": True}
                viz_carry = viz_model.initial_carry(viz_batch)
                _, _, _, viz_outs, _ = viz_model(carry=viz_carry, batch=viz_batch, return_keys=["trace"])  # type: ignore
                trace = viz_outs.get("trace") or []
                for rec in trace:
                    # values per token for current step
                    vals = rec["z_norm"][0].tolist()  # [T]
                    payload_trace = {
                        "type": "trace",
                        "level": rec["level"],
                        "h": rec["h"],
                        "l": rec.get("l"),
                        "values": vals,
                    }
                    handler.wfile.write(f"data: {json.dumps(payload_trace)}\n\n".encode())
                    handler.wfile.flush()
                    if delay_ms > 0:
                        time.sleep(delay_ms / 1000.0)
                payload = {
                    "type": "token",
                    "step": step,
                    "token_id": int(next_id.item()),
                    "token": tok.decode([int(next_id.item())]),
                    "top_tokens": top_tokens,
                    "top_probs": top_probs,
                    "entropy": entropy,
                    "text": tok.decode(generated[0].tolist()),
                }
                handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                handler.wfile.flush()
    except Exception as e:
        handler.wfile.write(f"data: {json.dumps({'type':'error','message':str(e)})}\n\n".encode())
    finally:
        try:
            handler.wfile.write(b"data: {\"type\": \"done\"}\n\n")
            handler.wfile.flush()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HRM Text Generation Visualizer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; }
    .row { display: flex; gap: 24px; }
    .col { flex: 1; }
    label { display:block; margin-top:8px; font-size: 12px; color:#444 }
    input, button { width:100%; padding:8px; margin-top:4px; }
    #output { white-space: pre-wrap; border:1px solid #ddd; padding:12px; min-height:120px }
    .bar { display:flex; align-items:center; gap:8px; margin:2px 0; }
    .bar .label { width: 80px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar .track { background:#eee; height:10px; flex:1; }
    .bar .fill { background:#4a90e2; height:10px; }
  </style>
  <script>
    function startStream(){
      const checkpoint = document.getElementById('checkpoint').value;
      const prompt = document.getElementById('prompt').value;
      const tokenizer = document.getElementById('tokenizer').value;
      const max_new_tokens = document.getElementById('max_new_tokens').value;
      const temperature = document.getElementById('temperature').value;
      const top_p = document.getElementById('top_p').value;
      const greedy = document.getElementById('greedy').checked;
      const delay_ms = document.getElementById('delay_ms').value;
      const h_cycles = document.getElementById('h_cycles').value;
      const l_cycles = document.getElementById('l_cycles').value;
      const viz_override = document.getElementById('viz_override').checked;

      document.getElementById('output').textContent = '';
      document.getElementById('topk').innerHTML = '';
      document.getElementById('trace').innerHTML = '';

      const url = `/stream?checkpoint=${encodeURIComponent(checkpoint)}&prompt=${encodeURIComponent(prompt)}&tokenizer=${encodeURIComponent(tokenizer)}&max_new_tokens=${max_new_tokens}&temperature=${temperature}&top_p=${top_p}&greedy=${greedy}&delay_ms=${delay_ms}&h_cycles=${h_cycles}&l_cycles=${l_cycles}&viz_override=${viz_override}`;
      const es = new EventSource(url);
      es.onmessage = (e)=>{
        const data = JSON.parse(e.data);
        if(data.type==='error'){
          alert(data.message);
          es.close();
          return;
        }
        if(data.type==='trace'){
          const vals = data.values;
          const w = 480, h = 70;
          const mx = Math.max(...vals, 1e-6);
          const pts = vals.map((v,i)=>{
            const x = i/(vals.length-1||1)*w;
            const y = h - (v/mx)*h;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          }).join(' ');
          const svg = `<div>Level: ${data.level} h=${data.h}${data.l!=null?(' l='+data.l):''}</div><svg width="${w}" height="${h}"><polyline fill="none" stroke="#e67e22" stroke-width="2" points="${pts}" /></svg>`;
          document.getElementById('trace').innerHTML = svg;
          return;
        }
        if(data.type==='token'){
          document.getElementById('output').textContent = data.text;
          const list = document.getElementById('topk');
          list.innerHTML = '';
          const probs = data.top_probs;
          const toks = data.top_tokens;
          const maxp = Math.max(...probs, 1e-6);
          for(let i=0;i<probs.length;i++){
            const row = document.createElement('div'); row.className='bar';
            const lab = document.createElement('div'); lab.className='label'; lab.textContent = toks[i];
            const track = document.createElement('div'); track.className='track';
            const fill = document.createElement('div'); fill.className='fill'; fill.style.width = `${(probs[i]/maxp*100).toFixed(1)}%`;
            track.appendChild(fill);
            const pr = document.createElement('div'); pr.style.width='48px'; pr.textContent = probs[i].toFixed(3);
            row.appendChild(lab); row.appendChild(track); row.appendChild(pr);
            list.appendChild(row);
          }
          document.getElementById('entropy').textContent = data.entropy.toFixed(4);
        }
        if(data.type==='done') es.close();
      };
    }
  </script>
  </head>
  <body>
    <h2>HRM Text Generation Visualizer</h2>
    <div class="row">
      <div class="col">
        <label>Checkpoint path</label>
        <input id="checkpoint" placeholder="/abs/path/to/checkpoints/.../step_xxx" />
        <label>Tokenizer</label>
        <input id="tokenizer" value="mistralai/Mistral-7B-v0.1" />
        <label>Prompt</label>
        <input id="prompt" value="Hello" />
        <div class="row">
          <div class="col">
            <label>Max new tokens</label>
            <input id="max_new_tokens" value="50" />
          </div>
          <div class="col">
            <label>Temperature</label>
            <input id="temperature" value="1.0" />
          </div>
          <div class="col">
            <label>Top-p</label>
            <input id="top_p" value="0.9" />
          </div>
          <div class="col">
            <label>Trace delay (ms)</label>
            <input id="delay_ms" value="150" />
          </div>
          <div class="col">
            <label>H cycles (viz)</label>
            <input id="h_cycles" value="2" />
          </div>
          <div class="col">
            <label>L cycles (viz)</label>
            <input id="l_cycles" value="2" />
          </div>
        </div>
        <label><input type="checkbox" id="greedy" /> Greedy</label>
        <label><input type="checkbox" id="viz_override" checked /> Use H/L override for visualization only</label>
        <button onclick="startStream()">Start</button>
      </div>
      <div class="col">
        <h4>Generated</h4>
        <div id="output"></div>
        <h4>Entropy</h4>
        <div id="entropy">0.0</div>
        <h4>Top-K</h4>
        <div id="topk"></div>
        <h4>H/L trace</h4>
        <div id="trace"></div>
      </div>
    </div>
  </body>
  </html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return
        if parsed.path == "/stream":
            params = urllib.parse.parse_qs(parsed.query)
            sse_headers(self)
            stream_generation(self, params)
            return
        self.send_response(404)
        self.end_headers()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving visualizer at http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
