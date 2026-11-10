import json
import os
import secrets
import socket
import struct
import sys
import threading
from contextlib import contextmanager
from itertools import islice
from paths import ROOT, EMPTY, check

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT / "pkg"))
import numpy as np
from finshybrid.de101 import DE101Model
from finshybrid.multistep_bias import MultiStepBiasUpdater, PERIOD_NS
from finshybrid.scales import PC, PS, VS, US

FRAME = struct.Struct("<32d")
REPLY = struct.Struct("<?8d")

def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def physical(values):
    return np.r_[values[:6] * PS + PC, values[6:] * VS]

class Adapter:
    def __init__(self, model_path, backend="native"):
        self.model = DE101Model(model_path)
        check(self.model.arm=="H_bias","wrong model")
        self.update = MultiStepBiasUpdater(self.model,backend=backend,cache_dir=ROOT)
        self.count = 0

    def step(self, packet):
        previous_t,t = packet[:2]
        prev, u, obs = np.split(np.asarray(packet[2:]), [12,18])
        start = round(previous_t*10)
        check(start==self.count and abs(previous_t-.1*start)<.003 and abs(t-.1*(start+1))<.003,"interval alignment")
        self.count += 1
        end = self.count*PERIOD_NS
        info = self.update.update(start*PERIOD_NS,end,physical(prev),u*US,physical(obs),available_ns=end)
        return info, self.update.b.copy()

def packet(row):
    return (row["previousTime"],row["t"],*row["previous"][1:13],*row["input"],*row["observed"][1:13])

def read_exact(stream,n):
    data = stream.read(n)
    if not data:
        return None
    check(len(data)==n,"truncated wire frame")
    return data

@contextmanager
def serve(model_path, log_path):
    token = secrets.token_hex(16)
    errors = []
    stop = threading.Event()
    adapter = Adapter(model_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1",0))
    listener.listen(1)
    listener.settimeout(.5)
    port = listener.getsockname()[1]
    def serve_one():
        try:
            while not stop.is_set():
                try:
                    conn,_ = listener.accept()
                    break
                except socket.timeout:
                    continue
            else:
                return
            with conn, conn.makefile("rb") as stream, log_path.open("x",encoding="utf-8") as log:
                conn.settimeout(60)
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                check(read_exact(stream,32)==token.encode("ascii"),"unexpected client")
                while (data:=read_exact(stream,FRAME.size)) is not None:
                    info, bias = adapter.step(FRAME.unpack(data))
                    conn.sendall(REPLY.pack(info["accepted"],info["before"] or 0.,info["after"] or 0.,*bias))
                    log.write(json.dumps(dict(info,bias=bias.tolist()),allow_nan=False)+"\n")
        except Exception as exc:
            errors.append(exc)
    thread=threading.Thread(target=serve_one,daemon=True)
    thread.start()
    try:
        yield {"FINSSIM_MULTISTEP_TOKEN":token,"FINSSIM_MULTISTEP_PORT":str(port)}
    finally:
        stop.set()
        thread.join(65)
        listener.close()
        check(not thread.is_alive(),"update worker failed to exit")
        if errors:
            raise RuntimeError("online adapter failed") from errors[0]

def reference_check(model_path, trace_path, n=24):
    a=Adapter(model_path)
    b=Adapter(model_path,"numpy")
    peak=0.
    for row in islice(rows(trace_path),n):
        x=packet(row)
        info,u=a.step(x)
        original,v=b.step(x)
        check(info["reason"]==original["reason"],(a.count,info,original))
        peak=max(peak,float(np.max(abs(u-v))))
    check(a.count==n and peak<1e-8,"multi-step update parity")
    return dict(ticks=n,max_bias_error=peak,accepted=a.update.accepted,rejected=a.update.rejected)
