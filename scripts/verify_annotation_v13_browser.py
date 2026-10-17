#!/usr/bin/env python3
"""Verify public annotation UI without persisting test labels or drafts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--database', type=Path, required=True)
    ap.add_argument('--base-url', default='http://47.108.220.174:3002')
    ap.add_argument('--chromium', type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    report = json.loads(args.report.read_text())
    private = Path(report['private_backup_directory'])
    cookie = json.loads((private / 'verification_session.json').read_text())
    def labels():
        with sqlite3.connect(f'file:{args.database}?mode=ro', uri=True) as conn:
            return {table: {'count': len(rows), 'sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()}
                    for table in ('task_completion', 'tasks_annotationdraft')
                    for rows in [conn.execute(f'SELECT * FROM {table} ORDER BY id').fetchall()]}
    before = labels()
    assert before == report['legacy_annotation_tables'], 'User changed labels since deployment; refresh protected baseline before browser verification'
    browser_results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(args.chromium), headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1100})
        context.add_cookies([{**cookie, 'url': args.base_url, 'httpOnly': True, 'sameSite': 'Lax'}])
        guide = context.request.get(args.base_url + '/data/local-files/?d=artifacts/annotation_batch_v1/guides/annotation_protocol_v1_3.html')
        assert guide.ok and 'text/html' in guide.headers.get('content-type', '')
        assert '八个练习例子' in guide.text() and '一帧一个窗口' in guide.text()
        # GET every unique source image/video through the actual public path.
        media = []
        seen = set()
        for project in report['projects']:
            task_payload = context.request.get(args.base_url + f"/api/tasks/?project={project['project_id']}&page_size=1000").json()
            tasks = task_payload['tasks']
            assert len(tasks) == project['tasks']
            project['_browser_task'] = next(t for t in tasks if t['total_annotations'] == 0)
            for task in tasks:
                d = task['data']
                if d['video_id'] in seen:
                    continue
                seen.add(d['video_id'])
                image = context.request.get(args.base_url + d['image'])
                video = context.request.get(args.base_url + d['context_video'], headers={'Range': 'bytes=0-255'})
                assert image.status == 200 and video.status == 206
                media.append({'video_id': d['video_id'], 'image_status': image.status, 'mp4_range_status': video.status})
        mutations = []
        ui_view_mutations = []
        def intercept(route):
            req = route.request
            # A newly created project lazily creates its data-manager view
            # using POST. Permit that UI preference, while blocking every
            # annotation/draft write (and all other mutation endpoints).
            ui_view = urlsplit(req.url).path.startswith('/api/dm/views')
            if ui_view and req.method not in ('GET', 'HEAD', 'OPTIONS'):
                ui_view_mutations.append({'method': req.method, 'path': urlsplit(req.url).path})
            if req.method not in ('GET', 'HEAD', 'OPTIONS') and not ui_view:
                mutations.append({'method': req.method, 'url': req.url, 'body': req.post_data})
                route.abort()
            else:
                route.continue_()
        context.route('**/*', intercept)
        guide_page = context.new_page()
        guide_page.goto(args.base_url + '/data/local-files/?d=artifacts/annotation_batch_v1/guides/annotation_protocol_v1_3.html', wait_until='domcontentloaded')
        assert guide_page.get_by_role('heading', name='高光视频打标标准 V1.3', exact=True).is_visible()
        guide_page.close()
        def choose(page, value):
            pattern = re.compile('^' + re.escape(value) + r'(?:\[[^\]]+\])?$')
            page.locator('label').filter(has_text=pattern).first.click()
        for project in report['projects']:
            task = project.pop('_browser_task')
            page = context.new_page()
            page.goto(args.base_url + f"/projects/{project['project_id']}/data?task={task['id']}", wait_until='domcontentloaded', timeout=45000)
            try:
                page.locator('label').filter(has_text=re.compile(r'^0普通(?:\[[^\]]+\])?$')).wait_for(state='visible', timeout=25000)
                # HyperText renders each block inside an iframe. Inspect the
                # real frames, rather than treating missing main-DOM text as
                # a missing guide or player.
                deadline = time.monotonic() + 20
                help_frame = media_frame = None
                while time.monotonic() < deadline:
                    help_frame = next((f for f in page.frames if f.locator('a[href*="guides/"]').count()), None)
                    media_frame = next((f for f in page.frames if f.locator('video').count()), None)
                    if help_frame and media_frame:
                        break
                    page.wait_for_timeout(100)
                assert help_frame and media_frame, 'Guide/player iframe missing'
                assert help_frame.get_by_role('link', name='打开完整打标标准 V1.3（流程、例子与画框说明）').is_visible()
                assert not page.locator('label').filter(has_text=re.compile(r'^未看查询')).is_visible()
                video = media_frame.locator('video').first
                video.wait_for(state='visible')
                media_frame.wait_for_function('''() => {const v=document.querySelector('video'); return v && v.readyState>=2 && v.duration>29;}''', timeout=30000)
                playback = video.evaluate('''async v => {v.muted=false; v.volume=1; v.currentTime=14; await v.play(); return {duration:v.duration,muted:v.muted,source:v.currentSrc};}''')
                media_frame.wait_for_function("document.querySelector('video').currentTime>14.2", timeout=10000)
                video.evaluate('v => v.pause()')
                if project['role'] == 'A':
                    choose(page, '0普通')
                    choose(page, '无事件或无法判断')
                    choose(page, '无事件')
                    choose(page, '高')
                    expected = {'generic_highlight','event_phase','event_type','confidence'}
                else:
                    choose(page, '2值得保留')
                    choose(page, '进行中')
                    choose(page, '动作')
                    choose(page, '高')
                    choose(page, '需要前后过程')
                    choose(page, '低或无合理框')
                    page.locator('textarea').last.fill('界面验证示例：相关主体分散，无法给出稳定窗口；不会保存此结果')
                    expected = {'generic_highlight','event_phase','event_type','confidence','needs_context','composition_confidence','annotator_note'}
                marker = len(mutations)
                page.locator('button').filter(has_text=re.compile(r'^Submit$')).click(timeout=10000)
                deadline = time.monotonic() + 10
                submitted = None
                while time.monotonic() < deadline:
                    submitted = next((m for m in mutations[marker:] if m['method']=='POST' and '/annotations' in m['url']), None)
                    if submitted:
                        break
                    page.wait_for_timeout(100)
                assert submitted is not None, 'No annotation POST reached; inspect validation and screenshot'
                fields = {r['from_name'] for r in json.loads(submitted['body'])['result']}
                assert expected == fields, fields
                screenshot = root / f"artifacts/annotation_batch_v1/v13_browser_{project['role']}.png"
                page.screenshot(path=str(screenshot), full_page=True)
                browser_results.append({'project_id':project['project_id'],'task_id':task['id'],
                                        'case':'nonhighlight_skip_dependencies' if project['role']=='A' else 'highlight_with_no_stable_crop',
                                        'submitted_fields':sorted(fields),'frontend_validation':'PASS',
                                        'playback':playback,'screenshot':str(screenshot.relative_to(root))})
            except Exception:
                page.screenshot(path=str(root / 'artifacts/annotation_batch_v1/v13_browser_debug.png'), full_page=True)
                print(page.locator('body').inner_text()[-7000:])
                print(json.dumps({'blocked_endpoints': sorted({(m['method'], urlsplit(m['url']).path) for m in mutations})}))
                raise
            finally:
                page.close()
        context.close()
        browser.close()
    assert labels() == before, 'Existing labels or drafts changed; inspect concurrent edits'
    report.update(browser_verification='PASS', browser_results=browser_results,
                  media_checks=media, public_guide_status=guide.status,
                  annotation_and_draft_mutations_blocked=True, test_annotations_saved=False,
                  allowed_data_manager_view_mutations=ui_view_mutations,
                  browser_command=[sys.executable, *sys.argv],
                  browser_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  playwright_version=importlib.metadata.version('playwright'),
                  browser_wall_seconds=time.perf_counter()-started,
                  annotation_and_draft_hashes_rechecked=True)
    report.pop('all_browser_mutations_blocked', None)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'browser_verification':'PASS','cases':browser_results,'unique_media_sources':len(media)},ensure_ascii=False))


if __name__ == '__main__':
    main()
