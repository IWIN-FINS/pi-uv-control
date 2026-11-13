"""Bounded MPC side channels. Only the controller process owns the actuator socket.

Display rendering and disk/terminal I/O run in separate spawned processes. Queue
backpressure fails closed; neither worker can replay or enable motor commands.
"""
from __future__ import annotations

import json
import math
import gc
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty,Full
import select
import socket
import time
import traceback

from control_log_storage import prepare_local_logs


def initialize_controller_input(pygame):
    """SDL's hidden controller window has no focus; input must be allowed in background."""
    os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    pygame.joystick.init()


class RealtimeGarbageCollector:
    """Defer cyclic GC to confirmed disabled output; refcount freeing remains.

    A full generation-2 collection must never run inside an enabled control
    cycle. The controller's histories/queues remain bounded independently.
    """
    def __init__(self,interval_sec=5.):
        if not math.isfinite(interval_sec) or interval_sec<=0:
            raise ValueError('Expected positive idle GC interval')
        self.was_enabled=gc.isenabled()
        gc.disable()
        self.interval_sec=interval_sec;self.next_collection=0.
        self.collections=0;self.last_pause_ms=0.;self.closed=False

    def collect_if_idle(self,now,*,armed,outputs_disabled):
        if self.closed:raise RuntimeError('GC scope already closed')
        if armed or not outputs_disabled or now<self.next_collection:return False
        started=time.monotonic()
        gc.collect()
        ended=time.monotonic()
        self.collections+=1;self.last_pause_ms=(ended-started)*1000
        self.next_collection=ended+self.interval_sec
        return True

    def close(self):
        if not self.closed:
            self.closed=True
            if self.was_enabled:gc.enable()

    def __enter__(self):return self

    def __exit__(self,*_):self.close()


def control_time_limits(period_sec):
    """Allow bounded host scheduling jitter; never change the command rate.

    At 20 Hz permit 75 ms processing and 120 ms dispatch gaps. At 10 Hz
    retain 85/160 ms. Both remain below the MCU's 250 ms command timeout.
    """
    if not math.isfinite(period_sec) or not 0<period_sec<=.1:
        raise ValueError('Unsupported control period')
    return max(.85*period_sec,.075),max(1.6*period_sec,.120)


class CycleWatchdog:
    period_sec = .1
    maximum_gap_sec = .16
    maximum_compute_sec = .085

    def __init__(self,period_sec=.1):
        self.period_sec=period_sec
        self.maximum_compute_sec,self.maximum_gap_sec=control_time_limits(period_sec)
        self.reset()

    def reset(self):
        self.last_start = self.last_send = self.started = None
        self.start_gap_sec = self.send_gap_sec = None
        self.compute_sec = None

    def begin(self, now):
        self.start_gap_sec = None if self.last_start is None else now-self.last_start
        if self.start_gap_sec is not None and self.start_gap_sec > self.maximum_gap_sec:
            raise TimeoutError(f'Control cycle timed out ({self.start_gap_sec*1000:.1f} ms > {self.maximum_gap_sec*1000:.1f} ms)')
        self.started = self.last_start = now

    def before_send(self, now):
        if self.started is None:
            raise RuntimeError('Control cycle has not started')
        self.compute_sec=now-self.started
        self.send_gap_sec = None if self.last_send is None else now-self.last_send
        if self.compute_sec > self.maximum_compute_sec:
            raise TimeoutError(f'Computation timed out; discard output ({self.compute_sec*1000:.1f} ms > {self.maximum_compute_sec*1000:.1f} ms)')
        if self.send_gap_sec is not None and self.send_gap_sec > self.maximum_gap_sec:
            raise TimeoutError(f'Command interval timed out ({self.send_gap_sec*1000:.1f} ms > {self.maximum_gap_sec*1000:.1f} ms)')

    def sent(self, now):
        self.last_send = now


def _journal_worker(queue, status, directory):
    files = {}
    try:
        root = Path(directory)
        for name in ('control.jsonl', 'actuator_snapshots.jsonl', 'sequence_events.jsonl'):
            files[name] = (root/name).open('w', buffering=1024*1024)
        next_flush=time.monotonic()+.2
        status.value = 1
        while True:
            try:item = queue.get(timeout=max(.001,next_flush-time.monotonic()))
            except Empty:item = ()
            if item is None:
                break
            if item:
                name, payload = item
                if name == 'console':
                    print(payload, flush=True)
                elif name in files:
                    files[name].write(json.dumps(payload, allow_nan=False)+'\n')
                    if name=='sequence_events.jsonl':files[name].flush()
                elif name == 'initial_heading.json' or (name.startswith('trajectory_plans/') and name.endswith('.json')):
                    target = root/name
                    target.parent.mkdir(exist_ok=True)
                    target.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
                else:
                    raise ValueError('Unknown journal stream')
            if time.monotonic()>=next_flush:
                for stream in files.values():stream.flush()
                next_flush=time.monotonic()+.2
    except BaseException:
        status.value = -1
        traceback.print_exc()
        raise
    finally:
        for stream in files.values():
            stream.close()
    if status.value == 1:
        status.value = 2


class ProcessJournal:
    """Nonblocking, lossless while healthy; a full queue requires disarming."""
    def __init__(self, directory, capacity=512):
        storage = prepare_local_logs(directory,
            ('control.jsonl', 'actuator_snapshots.jsonl', 'sequence_events.jsonl', 'initial_heading.json'),
            directories=('trajectory_plans',))
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue(maxsize=capacity)
        # Lock-free worker status also survives SIGKILL. A killed process can
        # leave an Event/Condition lock held forever during error cleanup.
        self.status = ctx.RawValue('i', 0)
        self.worker = ctx.Process(target=_journal_worker,
            args=(self.queue, self.status, str(storage)), name='mpc-journal', daemon=True)
        self.worker.start()
        # Interpreter/import/file creation can be slow on the workspace's
        # FUSE volume. This wait happens before arming, never during control.
        deadline = time.monotonic()+60.
        while self.status.value == 0 and self.worker.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        if self.status.value != 1 or not self.worker.is_alive():
            self.close()
            raise RuntimeError('Logging process initialization failed')
        self.pending_high_water = 0

    def check(self):
        if not self.worker.is_alive() or self.status.value != 1:
            raise RuntimeError('Logging process stopped; control disabled')

    def put(self, stream, record):
        self.check()
        try:
            self.queue.put_nowait((stream, record))
        except Full:
            raise RuntimeError('Logging queue full; control disabled') from None
        self.pending_high_water = max(self.pending_high_water, self.queue.qsize())

    def close(self):
        # Called only after the caller has requested disabled motor outputs.
        flushed = False
        if self.worker.is_alive():
            try:
                self.queue.put(None, timeout=.25)
                self.worker.join(2.)
                flushed = not self.worker.is_alive() and self.worker.exitcode == 0
            except Full:
                pass
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(.5)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(.5)
        self.queue.cancel_join_thread()
        self.queue.close()
        return flushed


def render_view(pygame, screen, font, view):
    screen.fill((15, 25, 36))
    lines = view['lines']
    if time.monotonic()-view['monotonic'] > .5:
        lines = ['Control status display is stale; check the control process']+lines[1:]
    for row, line in enumerate(lines):
        screen.blit(font.render(line, True, (222, 235, 242)), (20, 15+row*35))
    for i, value in enumerate(view['throttle']):
        y = 410+i*28
        screen.blit(font.render(f'CH{i+1}  {value:+.3f}', True, (200, 221, 235)), (30, y))
        pygame.draw.line(screen, (62, 82, 102), (230, y+13), (930, y+13), 4)
        pygame.draw.line(screen, (52, 194, 150), (580, y+13), (580+int(value/.30*340), y+13), 13)
    plan = view.get('trajectory')
    if plan is not None:
        message = 'Disarmed: R previews; Start descends; stabilize depth and vision; follow world trajectory. Joystick motion aborts'
        screen.blit(font.render(message, True, (255, 211, 111)), (20, 655))
        if plan['anchor'] is None:
            message = f'Press R near the surface to preview; target depth {plan["target_depth"]:.2f} m; check travel distance and bottom clearance first.'
            screen.blit(font.render(message, True, (230, 235, 240)), (20, 695))
        else:
            a, b = plan['anchor'], plan['endpoint']
            labels = [f'World start x/y=({a[0]:.3f}, {a[2]:.3f}) m -> end ({b[0]:.3f}, {b[2]:.3f}) m',
                f'Depth {plan["depth"]:.3f} m | Travel {plan["distance"]:.2f} m | Maximum reference speed {plan["speed"]:.3f} m/s',
                f'Phase {plan["phase"]} | Planar progress {plan["elapsed"]:.1f}/{plan["duration"]:.1f} s | blue corridor, yellow path, green vehicle']
            if plan.get('path'):
                stage=plan['collection_stage']
                labels[0]=f'Fixed-camera world region | route {stage["route"]} | blue box is the planning region'
                labels[1]=f'Depth {plan["depth"]:.3f} m | Maximum reference speed {plan["speed"]:.3f} m/s | route does not rotate with heading'
                if stage.get('continuous'):
                    labels[2]=f'Phase {plan["phase"]} | Continuous cycle {stage["cycle"]+1} | cumulative planar time {plan["elapsed"]:.1f} s | Start disarms'
            for row, line in enumerate(labels):
                screen.blit(font.render(line, True, (230, 235, 240)), (20, 690+row*32))
            if plan['active']:
                for row, line in enumerate(plan['blocking_reasons'][:2] or ['Target depth reached; vision and EKF ready; confirmation timer running']):
                    screen.blit(font.render('Waiting: '+line, True, (255, 211, 111)), (20, 790+row*30))
            scale = 65./max(plan['distance'], .05)
            def pixel(v):
                return int(920+scale*(v[0]-a[0])), int(815-scale*(v[2]-a[2]))
            if plan.get('path'):
                bounds=plan['bounds']
                xmin,xmax=min(p[0] for p in bounds),max(p[0] for p in bounds)
                ymin,ymax=min(p[2] for p in bounds),max(p[2] for p in bounds)
                scale=min(190/max(xmax-xmin,.05),115/max(ymax-ymin,.05))
                def pixel(v):return int(915+scale*(v[0]-xmin)),int(875-scale*(v[2]-ymin))
                pygame.draw.polygon(screen,(90,135,190),[pixel(v) for v in bounds],2)
                pygame.draw.lines(screen,(255,211,111),False,[pixel(v) for v in plan['path']],2)
            elif plan['phase'] == 'tracking':
                direction = plan['direction']
                side = [-direction[2], 0., direction[0]]
                corners = [[a[j]+along*direction[j]+cross*side[j] for j in range(3)]
                    for along, cross in ((-.15, -.15), (plan['distance']+.15, -.15),
                        (plan['distance']+.15, .15), (-.15, .15))]
                pygame.draw.polygon(screen, (90, 135, 190), [pixel(v) for v in corners], 2)
            if not plan.get('path'):
                pygame.draw.line(screen, (255, 211, 111), pixel(a), pixel(b), 4)
            for endpoint in (a, b):
                pygame.draw.circle(screen, (255, 211, 111), pixel(endpoint), 6)
            if view['position'] is not None:
                pygame.draw.circle(screen, (52, 194, 150), pixel(view['position']), 7)
    pygame.display.flip()


def _display_worker(channel, title, line_mode):
    import pygame
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((1140, 900 if line_mode else 650))
        pygame.display.set_caption(title)
        font = pygame.font.SysFont('notosanscjksc,wenquanyimicrohei,dejavusans', 21)
        pending = []
        latest = None
        last_heartbeat = 0.
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pending.append('quit')
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    pending.append('plan')
            now = time.monotonic()
            if pending or now-last_heartbeat >= .1:
                try:
                    channel.send(json.dumps(dict(monotonic=now, events=pending)).encode())
                    quitting = 'quit' in pending
                    pending = []
                    last_heartbeat = now
                    if quitting:
                        break
                except BlockingIOError:
                    pass
            changed = False
            for _ in range(32):
                try:
                    latest = json.loads(channel.recv(65536))
                    if latest.get('close'):
                        return
                    changed = True
                except BlockingIOError:
                    break
            if latest is not None and changed:
                render_view(pygame, screen, font, latest)
            time.sleep(.02)
    finally:
        pygame.quit()
        channel.close()


class ProcessDisplay:
    """Latest-value display IPC; keyboard R/close only, never motor authority."""
    def __init__(self, title, line_mode=False):
        ctx = mp.get_context('spawn')
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.channel.setblocking(False)
        child.setblocking(False)
        self.worker = ctx.Process(target=_display_worker,
            args=(child, title, line_mode), name='mpc-display', daemon=True)
        self.worker.start()
        child.close()
        self.last_heartbeat = None
        self.dropped_views = 0
        self.next_view = 0.
        if not select.select([self.channel], [], [], 60.)[0]:
            self.close()
            raise RuntimeError('Control window initialization timed out')
        self.poll(time.monotonic())

    def poll(self, now):
        events = []
        for _ in range(32):
            try:
                message = json.loads(self.channel.recv(65536))
            except BlockingIOError:
                break
            self.last_heartbeat = message['monotonic']
            events.extend(message['events'])
        if 'quit' in events:
            return events
        if not self.worker.is_alive():
            raise RuntimeError('Control window closed; control disabled')
        if self.last_heartbeat is None or now-self.last_heartbeat > 2.:
            raise RuntimeError('Control window unresponsive; control disabled')
        return events

    def due(self, now):
        if now < self.next_view:
            return False
        self.next_view = now+.1
        return True

    def publish(self, view):
        try:
            self.channel.send(json.dumps(view, allow_nan=False).encode())
        except BlockingIOError:
            # Old views may be dropped. Commands and authoritative state stay
            # in the controller process; a stalled display never queues output.
            self.dropped_views += 1

    def close(self):
        try:
            self.channel.send(b'{"close":true}')
        except OSError:
            pass
        self.worker.join(.5)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(.5)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(.5)
        self.channel.close()
