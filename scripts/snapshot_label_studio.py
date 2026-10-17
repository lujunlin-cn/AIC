#!/usr/bin/env python3
"""Read a consistent, credential-free annotation snapshot; never merge drafts."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--database', type=Path, required=True)
    ap.add_argument('--project', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--private-backup', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True) as conn:
        if args.private_backup:
            args.private_backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if args.private_backup.exists():
                raise SystemExit('Backup already exists; refusing to overwrite it')
            with sqlite3.connect(args.private_backup) as backup:
                conn.backup(backup)
            os.chmod(args.private_backup, 0o600)
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        def rows(sql: str):
            return [dict(row) for row in conn.execute(sql, (args.project,))]
        tasks = rows('SELECT id,data,project_id,created_at,updated_at FROM task WHERE project_id=? ORDER BY id')
        annotations = rows('SELECT id,task_id,result,was_cancelled,created_at,updated_at,lead_time,completed_by_id,ground_truth FROM task_completion WHERE project_id=? ORDER BY id')
        drafts = rows('SELECT d.id,d.task_id,d.result,d.created_at,d.updated_at,d.user_id,d.annotation_id,d.was_postponed FROM tasks_annotationdraft d JOIN task t ON t.id=d.task_id WHERE t.project_id=? ORDER BY d.id')
        label_config = conn.execute('SELECT label_config FROM project WHERE id=?', (args.project,)).fetchone()[0]
        for row in tasks:
            row['data'] = json.loads(row['data'])
        for row in annotations + drafts:
            row['result'] = json.loads(row['result'])
            for field in ('was_cancelled', 'ground_truth', 'was_postponed'):
                if field in row:
                    row[field] = bool(row[field])
        payload = {
            'exported_at_utc': datetime.now(timezone.utc).isoformat(),
            'method': 'SQLite read-only transaction; completed annotations and drafts separate',
            'tasks': tasks, 'annotations': annotations, 'drafts': drafts,
        }
    target = args.output / 'raw_export.json'
    if target.exists():
        raise SystemExit('Snapshot exists; choose a new output directory')
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'label_config.xml').write_text(label_config)
    print(json.dumps({'project': args.project, 'tasks': len(tasks), 'annotations': len(annotations), 'drafts': len(drafts), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
