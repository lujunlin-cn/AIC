#!/usr/bin/env python3
"""Render the shared V1.3 guide and an interface retaining legacy controls.

Run with the Label Studio venv (markdown-it-py is already installed there).
No saved annotation or task data is modified by this builder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from markdown_it import MarkdownIt


GUIDE_URL = '/data/local-files/?d=artifacts/annotation_batch_v1/guides/annotation_protocol_v1_3.html'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--base-config', type=Path, required=True)
    ap.add_argument('--output-config', type=Path, required=True)
    ap.add_argument('--output-html', type=Path, required=True)
    args = ap.parse_args()
    body = MarkdownIt('commonmark').enable('table').render(args.protocol.read_text())
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>高光视频打标标准 V1.3</title><style>'
            'body{font:17px/1.8 system-ui,sans-serif;max-width:960px;margin:32px auto;padding:0 20px;color:#17202a}'
            'h1,h2,h3{line-height:1.4}h2{margin-top:2em}table{border-collapse:collapse;width:100%}'
            'th,td{border:1px solid #cbd5e1;padding:9px;text-align:left;vertical-align:top}'
            'pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}a{color:#075fbd}'
            '@media print{body{font-size:12pt}h2{break-after:avoid}tr{break-inside:avoid}}'
            '</style><body>' + body + '</body></html>\n')
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html)
    old = ET.parse(args.base_config).getroot()
    controls = {e.attrib['name']: e for e in old.iter() if 'name' in e.attrib}
    root = ET.Element('View')

    def header(value: str) -> None:
        ET.SubElement(root, 'Header', value=value)

    def control(name: str) -> None:
        root.append(deepcopy(controls[name]))

    header('V1.3｜$video_id｜图片在片段第 $time_sec 秒｜源素材时间 $source_time_sec 秒')
    ET.SubElement(root, 'HyperText', name='protocol_help', value=(
        f'<p><a href="{GUIDE_URL}" target="_blank" rel="noopener noreferrer">'
        '<strong>打开完整打标标准 V1.3（流程、例子与画框说明）</strong></a></p>'
        '<p><strong>先完整看30秒视频并开声音 → 给当前时刻等级 → 填阶段、类型、把握 → '
        '2/3再填依赖和一个构图窗口 → 提交。</strong><br>'
        '同一视频看完一次即可，换视频后重看。只进自己的项目，不看另一人的答案。'
        '没有数量/比例要求；连续精彩可以连续2/3，整段普通也可以全0/1。</p>'))
    control('context_player')
    header('判断：证据不够且影响等级→不确定；确定普通→0；背景/普通过程→1；明确值得保留→2；关键动作/结果/信息→3。不要为了去重把相邻高光降级。')
    control('image')
    header('1. 高光等级：这一时刻有没有明确值得保留的事件或信息？不因清晰、居中、运动大或时间位置单独加分。')
    control('generic_highlight')
    header('2. 事件阶段：普通动作也有阶段。「结果后反应」指表情/庆祝/反馈，不是结果之后的所有画面。')
    control('event_phase')
    header('3. 事件类型：选当前主要的一类。没有高光不等于没有事件；普通走路仍可选动作。')
    control('event_type')
    header('4. 判断把握：这是对判断有多确定，不是精彩程度。等级为不确定时选低，并备注缺什么。')
    control('confidence')
    header('5. 信息依赖：0/1可跳过；2/3请填。问判断价值需要什么证据，不是问有没有看过视频。')
    header('图中已能说明价值→当前帧足够；必须对比前后状态/动作结果→前后过程；关键含义在台词/语气/声音→声音或语言；仍缺证据→无法判断。过程与声音都需要时选声音，并备注。')
    control('needs_context')
    hidden = ET.SubElement(root, 'View', style='display: none;')
    hidden.append(deepcopy(controls['query_relevance']))
    header('6. 构图：仅2/3考虑。一帧最多一个完整保留窗口，像观众最后看到的画面；不要分别框人、物、字幕，不要紧紧圈脸。可以接近整图。本批不限制比例。')
    header('下方四种框标签只选一个主要理由，再画一个窗口；多人/人和物/字幕需要一起保留时放进同一个框。误画第二个要删除。')
    control('crop_box')
    header('画框理由：有框时填写，可多选。')
    control('crop_reason')
    header('构图把握：2/3有框请填；无合理窗口则不画，选低或无合理框并备注。不要因难裁剪降低高光等级。')
    control('composition_confidence')
    header('备注：不确定、听不懂、事件超出片段、无合理窗口或另有合理构图时，写一句具体原因。')
    note = deepcopy(controls['annotator_note'])
    note.set('placeholder', '例如：关键结果被挡住 / 听不懂语言 / 同时需要前后过程 / 主体分散难裁剪；音频不可用写 audio_unavailable')
    root.append(note)
    ET.indent(root, space='  ')
    args.output_config.write_text(ET.tostring(root, encoding='unicode') + '\n')
    # The new help block adds no labels; all old control names, types, and
    # option values remain unchanged, so saved results stay representable.
    new_controls = {e.attrib['name']: e for e in root.iter() if 'name' in e.attrib}
    for name, element in controls.items():
        assert name in new_controls and element.tag == new_controls[name].tag
        for tag in ('Choice', 'Label'):
            assert [x.attrib['value'] for x in element.findall(tag)] == [x.attrib['value'] for x in new_controls[name].findall(tag)]
    print(f'Built {args.output_config} and {args.output_html}; legacy controls preserved')


if __name__ == '__main__':
    main()
