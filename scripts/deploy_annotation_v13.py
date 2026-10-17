#!/usr/bin/env python3
"""Expand two blind annotation projects through the local Label Studio API.

Use the service's Python venv. Existing labels/drafts are hashed before and
after, never written. Import is idempotent on (video_id, time_sec). An own
temporary admin session is stored only in a private backup directory for
browser verification; remove it after verification. No credentials are logged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--database', type=Path, required=True)
    ap.add_argument('--service-environment', type=Path, required=True)
    ap.add_argument('--assignments', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:18081')
    args = ap.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.assignments.read_text())
    db = args.database.resolve()
    private = db.parent / 'backups' / ('v13_api_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    private.mkdir(parents=True, mode=0o700)
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn, sqlite3.connect(private / db.name) as backup:
        conn.backup(backup)
    os.chmod(private / db.name, 0o600)

    def read(sql: str, values=()):
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
            return conn.execute(sql, values).fetchall()

    def annotations():
        return {table: {'count': len(rows), 'sha256': digest(json.dumps(rows, ensure_ascii=False).encode())}
                for table in ('task_completion', 'tasks_annotationdraft')
                for rows in [read(f'SELECT * FROM {table} ORDER BY id')]}

    protected_labels = annotations()
    old_tasks = read('SELECT id,data,project_id FROM task ORDER BY id')
    # Load the same settings as the running service, without displaying them.
    logging.disable(logging.CRITICAL)
    for line in args.service_environment.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key] = shlex.split(value)[0] if value else ''
    os.environ.update(DJANGO_SETTINGS_MODULE='core.settings.label_studio', DATABASE_NAME=str(db), LOG_LEVEL='ERROR')
    import label_studio
    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    import django
    django.setup()
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.test import Client
    from projects.models import Project
    import requests
    from markdown_it import MarkdownIt

    xml_path = root / 'configs/label_studio_annotation_v1.xml'
    xml = xml_path.read_text()
    Project.validate_label_config(xml)
    owner = read('SELECT created_by_id FROM project WHERE id=?', (cfg['projects'][0]['existing_project_id'],))[0][0]
    user = get_user_model().objects.get(pk=owner)
    client = Client()
    client.force_login(user, backend='django.contrib.auth.backends.ModelBackend')
    session = client.session
    session['last_login'] = time.time()
    session.save()
    cookie = {'name': settings.SESSION_COOKIE_NAME, 'value': session.session_key}
    client.cookies[cookie['name']] = cookie['value']
    assert client.get('/api/projects/1/').status_code == 200
    http = requests.Session()
    http.cookies.set(cookie['name'], cookie['value'])
    http.get(args.base_url + '/user/login/', timeout=20).raise_for_status()
    if 'csrftoken' in http.cookies:
        http.headers['X-CSRFToken'] = http.cookies['csrftoken']
    cookie_path = private / 'verification_session.json'
    cookie_path.write_text(json.dumps(cookie))
    os.chmod(cookie_path, 0o600)

    def api(method: str, path: str, **kwargs):
        response = http.request(method, args.base_url + path, timeout=45, **kwargs)
        if not response.ok:
            # Avoid printing whole responses that could contain sensitive
            # project or session fields. Validation details stay private.
            error_path = private / 'api_error.txt'
            error_path.write_text(response.text)
            os.chmod(error_path, 0o600)
            raise RuntimeError(f'{method} {path} returned {response.status_code}; details in private backup')
        return response.json()

    protocol_path = root / 'reports/annotation_protocol_v1_3.md'
    instructions = MarkdownIt('commonmark').enable('table').render(protocol_path.read_text())
    report = {
        'date': datetime.now(timezone.utc).isoformat(), 'protocol_version': '1.3',
        'git_commit_before_changes': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'private_backup_directory': str(private),
        'config_sha256': digest(xml_path.read_bytes()),
        'protocol_sha256': digest(protocol_path.read_bytes()),
        'assignments_sha256': digest(args.assignments.read_bytes()),
        'script_sha256': digest(Path(__file__).read_bytes()),
        'legacy_annotation_tables': protected_labels,
        'command': [sys.executable, *sys.argv],
        'python': sys.version.split()[0], 'label_studio': importlib.metadata.version('label-studio'),
        'projects': [], 'backend_config_validation': 'PASS',
    }
    keysets = []
    for assignment in cfg['projects']:
        role = assignment['role']
        title = assignment['title']
        if 'existing_project_id' in assignment:
            project_id = assignment['existing_project_id']
        else:
            found = read('SELECT id FROM project WHERE title=?', (title,))
            assert len(found) <= 1, 'Duplicate projects already exist'
            project_id = found[0][0] if found else None
        fields = {
            'title': title, 'label_config': xml, 'expert_instruction': instructions,
            'show_instruction': True, 'show_annotation_history': False,
            'show_collab_predictions': False, 'maximum_annotations': 1,
            'enable_empty_annotation': False, 'sampling': 'Sequential sampling',
            'description': f'标注员{ "甲" if role == "A" else "乙" }专用。V1.3：先有声完整看30秒，再逐秒判断。新增240题。只做本项目，不看另一项目结果。每帧最多一个完整保留窗口，无高光配额。',
        }
        if project_id is None:
            project_id = api('POST', '/api/projects/', json=fields)['id']
        else:
            api('PATCH', f'/api/projects/{project_id}/', json=fields)
        check = api('GET', f'/api/projects/{project_id}/')
        assert check['label_config'].strip() == xml.strip()
        assert '高光视频打标标准 V1.3' in check['expert_instruction']
        # Serving-only connections: never invoke storage sync, which would
        # create unintended image/video tasks.
        stores = api('GET', f'/api/storages/localfiles/?project={project_id}')
        for directory in ('frames', 'videos', 'guides'):
            path = str(root / 'artifacts/annotation_batch_v1' / directory)
            if not any(s['path'] == path for s in stores):
                api('POST', '/api/storages/localfiles/', json={
                    'project': project_id, 'path': path, 'title': f'V1.3 {directory} (serve only)',
                    'use_blob_urls': True, 'regex_filter': '(?!)',
                })
        task_path = root / f'artifacts/annotation_batch_v1/label_studio_tasks_v13_{role}.json'
        command = [sys.executable, str(root / 'scripts/build_annotation_tasks.py'),
                   '--config', str(root / cfg['source_manifest']),
                   '--frames', str(root / 'artifacts/annotation_batch_v1/frames'),
                   '--videos', str(root / 'artifacts/annotation_batch_v1/videos'),
                   '--image-mode', 'label-studio-local-files', '--document-root', str(root),
                   '--output', str(task_path)]
        for vid in assignment['video_ids']:
            command.extend(['--video-id', vid])
        subprocess.run(command, check=True, capture_output=True)
        desired = json.loads(task_path.read_text())
        def key(data):
            return data['video_id'], float(data['time_sec'])
        allowed = {'image','context_video','context_player','video_id','time_sec','source_offset_sec','source_time_sec'}
        assert all(set(t['data']) == allowed and set(t) == {'data'} for t in desired)
        desired_by_key = {key(t['data']): t for t in desired}
        assert len(desired_by_key) == len(desired) == assignment['expected_tasks']
        existing = read('SELECT id,data FROM task WHERE project_id=? ORDER BY id', (project_id,))
        have = {key(json.loads(data)) for _, data in existing}
        assert len(have) == len(existing) and have <= desired_by_key.keys()
        missing = [t for t in desired if key(t['data']) not in have]
        if missing:
            api('POST', f'/api/projects/{project_id}/import', json=missing)
        actual = read('SELECT id,data FROM task WHERE project_id=? ORDER BY id', (project_id,))
        actual_keys = {key(json.loads(data)) for _, data in actual}
        assert len(actual_keys) == len(actual) == len(desired) and actual_keys == desired_by_key.keys()
        keysets.append(actual_keys)
        table_counts = dict(read('SELECT is_labeled,count(*) FROM task WHERE project_id=? GROUP BY is_labeled', (project_id,)))
        completed = read('SELECT count(*) FROM task_completion WHERE project_id=? AND was_cancelled=0', (project_id,))[0][0]
        predictions = read('SELECT sum(total_predictions) FROM task WHERE project_id=?', (project_id,))[0][0]
        if role == 'B':
            assert completed == 0 and predictions == 0, 'B must start without annotations or predictions'
        report['projects'].append({
            'role': role, 'project_id': project_id, 'title': title,
            'tasks': len(actual), 'new_tasks_imported_this_run': len(missing),
            'completed_annotations': completed, 'remaining_tasks': table_counts.get(0, 0),
            'video_count': len({k[0] for k in actual_keys}),
            'task_file_sha256': digest(task_path.read_bytes()), 'task_file': str(task_path.relative_to(root)),
            'predictions': predictions,
            'new_task_protocol': '1.3', 'legacy_smoke_protocol': cfg['existing_smoke_protocol'] if role == 'A' else None,
        })
    union, intersection = keysets[0] | keysets[1], keysets[0] & keysets[1]
    assert len(union) == cfg['expected_unique_frames']
    assert len(intersection) == cfg['expected_overlap_frames']
    overlap_ids = set(json.loads((root / cfg['source_manifest']).read_text())['annotators']['overlap_video_ids'])
    assert {k[0] for k in intersection} == overlap_ids
    after = annotations()
    assert protected_labels == after, 'Labels or drafts changed concurrently; compare private backup, do not restore over user edits'
    assert old_tasks == read('SELECT id,data,project_id FROM task WHERE id<=? ORDER BY id', (max(t[0] for t in old_tasks),))
    report.update(
        unique_frames=len(union), unique_videos=len({k[0] for k in union}),
        overlap_frames=len(intersection), overlap_videos=len(overlap_ids),
        existing_results_unchanged=True, existing_task_data_unchanged=True,
        drafts_merged=False, blind_task_allowlist_pass=True,
        account_access_isolation=False, browser_verification='pending',
        wall_seconds=time.perf_counter()-started,
        environment={'device': 'CPU', 'gpu_used': False, 'formal_training': False},
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['projects','unique_frames','overlap_frames','existing_results_unchanged','wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
