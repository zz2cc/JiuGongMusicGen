"""
工尺谱 → MusicXML 转换模块。

将生成的工尺谱旋律序列转换为标准 MusicXML 3.1 格式文件，
可直接用 MuseScore 等软件打开、编辑和播放。

生成的 MusicXML 格式与原始数据集保持一致:
- divisions: 10080 (每个四分音符=10080 ticks)
- 拍号: 2/4
- 无调号
- G 谱号
- 歌词通过 <lyric> 元素附加
- 拖腔用 <slur> 连接
"""

import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import date
from lxml import etree

from .gongche_vocab import GONGCHE_TO_PITCH, PITCH_TO_STEP_OCTAVE
from .data_loader import NoteEvent, LyricNoteGroup


# ============================================================================
# MusicXML 常量
# ============================================================================

DIVISIONS = 10080          # 每个四分音符的 tick 数
BEATS_PER_MEASURE = 2      # 拍号分子
BEAT_TYPE = 4              # 拍号分母
MAX_DIVISIONS_PER_MEASURE = DIVISIONS * BEATS_PER_MEASURE  # 20160



def _make_element(tag: str, text: Optional[str] = None,
                  attrib: Optional[dict] = None) -> etree.Element:
    """创建 XML 元素（无命名空间）"""
    el = etree.Element(tag)
    if attrib:
        for k, v in attrib.items():
            el.set(k, str(v))
    if text is not None:
        el.text = str(text)
    return el


def _sub_element(parent: etree.Element, tag: str,
                 text: Optional[str] = None,
                 attrib: Optional[dict] = None) -> etree.Element:
    """创建子元素"""
    el = etree.SubElement(parent, tag)
    if attrib:
        for k, v in attrib.items():
            el.set(k, str(v))
    if text is not None:
        el.text = str(text)
    return el


# ============================================================================
# 核心转换
# ============================================================================


def gongche_to_musicxml(
    lyric_groups: List[LyricNoteGroup],
    output_path: str,
    title: str = "",
    work_title: str = "",
    qupai: str = "",
    source: str = "",
) -> str:
    """
    将工尺谱旋律序列写入 MusicXML 文件。

    Args:
        lyric_groups: 歌词-音符分组列表
        output_path: 输出 .musicxml 文件路径
        title: 曲目标题
        work_title: 作品标题
        qupai: 曲牌名
        source: 来源

    Returns:
        生成的 MusicXML 文件路径
    """
    # ---- 计算每小节的音符分组 ----
    measures = _partition_into_measures(lyric_groups)

    # ---- 构建 MusicXML 文档 ----
    score = _make_element("score-partwise", attrib={"version": "3.1"})

    # 文档头
    _write_header(score, title, work_title, qupai, source)

    # Part list
    part_list = _sub_element(score, "part-list")
    part_id = "P" + hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:30]
    score_part = _sub_element(part_list, "score-part", attrib={"id": part_id})
    _sub_element(score_part, "part-name")

    # Part（音符数据）
    part = _sub_element(score, "part", attrib={"id": part_id})

    # ---- 逐小节写入 ----
    slur_counter = [0]  # 用 list 是可变的，跨小节共享计数

    for measure_idx, measure_events in enumerate(measures):
        measure = _sub_element(part, "measure", attrib={"number": str(measure_idx + 1)})

        # 第一小节：属性
        if measure_idx == 0:
            _write_attributes(measure)

        # 写入音符
        _write_measure_notes(measure, measure_events, measure_idx == len(measures) - 1, slur_counter)

    # ---- 写入文件 ----
    tree = etree.ElementTree(score)
    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
        doctype='<!DOCTYPE score-partwise  PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        pretty_print=True,
    )

    return str(output_path)


# ============================================================================
# 小节划分
# ============================================================================


@dataclass
class _MeasureEvent:
    """小节内的音符事件"""
    lyric: str
    step: str
    octave: int
    xml_duration: int       # divisions 单位的时长
    musicxml_type: str      # "quarter", "eighth", "half", "16th"
    is_dotted: bool
    has_lyric: bool         # 是否承载歌词（拖腔后续音=否）
    is_melisma_start: bool  # 拖腔起始
    is_melisma_continue: bool  # 拖腔延续（跨小节中间段）
    is_melisma_end: bool    # 拖腔结束
    is_tie_start: bool      # 同音拖腔 — 用 tie 不用 slur
    is_tie_stop: bool
    beam: Optional[str]     # "begin", "continue", "end", None
    stem: str               # "up", "down", None


def _csv_to_xml_duration(csv_duration: float) -> Tuple[int, str, bool]:
    """
    CSV时长 (四分音符=0.25) → MusicXML duration (divisions)

    Returns: (xml_duration, musicxml_type, is_dotted)
    """
    # 1 quarter = 0.25 csv = 10080 divisions
    # csv_duration * 40320 = xml_duration
    raw = csv_duration * 40320
    xml_dur = int(round(raw))

    # 确定 MusicXML type
    dur_map = {
        10080: "quarter",
        5040: "eighth",
        20160: "half",
        2520: "16th",
        15120: "quarter",  # dotted quarter
        7560: "eighth",    # dotted eighth
    }
    is_dotted = abs(raw - 15120) < 10 or abs(raw - 7560) < 10

    musicxml_type = dur_map.get(xml_dur, "quarter")
    return xml_dur, musicxml_type, is_dotted


def _partition_into_measures(
    lyric_groups: List[LyricNoteGroup]
) -> List[List[_MeasureEvent]]:
    """
    将旋律序列划分为小节（每小节 2/4 = 2 quarters = 20160 divisions）。

    策略：
    - 累积音符时长，当达到 20160 时换小节
    - 如果单个音符跨小节，使用 tie 连接
    """
    measures = []
    current_measure = []
    current_div_total = 0

    all_events = []  # 先展平所有事件

    for group in lyric_groups:
        notes = group.notes
        # 如果组内所有音符同音高 → 不需要弧线（slur/tie），直接按独立音符处理
        pitches_in_group = [GONGCHE_TO_PITCH.get(n.gongche, 0.0) for n in notes]
        all_same_pitch = len(set(pitches_in_group)) == 1
        is_melisma = (len(notes) > 1) and not all_same_pitch

        for i, note in enumerate(notes):
            xml_dur, mtype, is_dotted = _csv_to_xml_duration(note.duration)

            is_first = (i == 0)
            is_last = (i == len(notes) - 1)

            # Beam: 八分音符需要 beam
            beam = None
            if mtype == "eighth" or xml_dur <= 5040:
                beam = "begin"  # 先标记，后面调整

            # Stem
            pitch = pitches_in_group[i]
            stem = "up" if pitch <= 2.0 else "down"

            all_events.append(_MeasureEvent(
                lyric=note.lyric if is_first else "",
                step=PITCH_TO_STEP_OCTAVE.get(pitch, ("C", 4))[0],
                octave=PITCH_TO_STEP_OCTAVE.get(pitch, ("C", 4))[1],
                xml_duration=xml_dur,
                musicxml_type=mtype,
                is_dotted=is_dotted,
                has_lyric=is_first,
                is_melisma_start=is_melisma and is_first,
                is_melisma_continue=is_melisma and not is_first and not is_last,
                is_melisma_end=is_melisma and is_last,
                is_tie_start=False,  # 跨小节 tie 稍后处理
                is_tie_stop=False,
                beam=beam,
                stem=stem,
            ))

    # ---- 分配到小节 ----
    i = 0
    while i < len(all_events):
        ev = all_events[i]
        needed = current_div_total + ev.xml_duration

        if needed <= MAX_DIVISIONS_PER_MEASURE:
            current_measure.append(ev)
            current_div_total = needed

            if abs(current_div_total - MAX_DIVISIONS_PER_MEASURE) < 10:
                # 小节正好填满
                _fix_beams(current_measure)
                measures.append(current_measure)
                current_measure = []
                current_div_total = 0
            i += 1
        else:
            # 音符跨小节：拆分为两个音符 + tie
            remaining = MAX_DIVISIONS_PER_MEASURE - current_div_total
            if remaining > 0:
                # 前小节部分
                first_part = _MeasureEvent(
                    lyric=ev.lyric,
                    step=ev.step,
                    octave=ev.octave,
                    xml_duration=remaining,
                    musicxml_type=_duration_to_type(remaining),
                    is_dotted=False,
                    has_lyric=ev.has_lyric,
                    is_melisma_start=ev.is_melisma_start,
                    is_melisma_continue=False,
                    is_melisma_end=False,
                    is_tie_start=True,
                    is_tie_stop=False,
                    beam=ev.beam,
                    stem=ev.stem,
                )
                current_measure.append(first_part)

            _fix_beams(current_measure)
            measures.append(current_measure)
            current_measure = []
            current_div_total = 0

            # 后小节部分
            second_part = _MeasureEvent(
                lyric="",
                step=ev.step,
                octave=ev.octave,
                xml_duration=ev.xml_duration - remaining,
                musicxml_type=_duration_to_type(ev.xml_duration - remaining),
                is_dotted=False,
                has_lyric=False,
                is_melisma_start=False,
                is_melisma_continue=True,
                is_melisma_end=ev.is_melisma_end,
                is_tie_start=False,
                is_tie_stop=True,
                beam=ev.beam,
                stem=ev.stem,
            )
            current_measure.append(second_part)
            current_div_total = ev.xml_duration - remaining
            i += 1

    # 最后一个小节（补齐）
    if current_measure:
        _fix_beams(current_measure)
        measures.append(current_measure)

    return measures


def _fix_beams(measure_events: List[_MeasureEvent]):
    """修正 beam 标记（begin/continue/end）"""
    beamed_idx = [j for j, e in enumerate(measure_events)
                  if e.beam is not None and e.xml_duration <= 5040]
    for k, idx in enumerate(beamed_idx):
        if k == 0:
            measure_events[idx].beam = "begin"
        elif k == len(beamed_idx) - 1:
            measure_events[idx].beam = "end"
        else:
            measure_events[idx].beam = "continue"


def _duration_to_type(xml_dur: int) -> str:
    """XML duration → MusicXML type"""
    mapping = {10080: "quarter", 5040: "eighth", 20160: "half",
               2520: "16th", 15120: "quarter"}
    return mapping.get(xml_dur, "quarter")


# ============================================================================
# MusicXML 元素写入
# ============================================================================


def _write_header(score: etree.Element, title: str, work_title: str,
                  qupai: str, source: str):
    """写入 MusicXML 文档头"""
    work = _sub_element(score, "work")
    wt = work_title or title or f"{qupai}"
    if source:
        wt = f"{source}《{qupai}》" if qupai else f"{source}"
    _sub_element(work, "work-title", wt)

    _sub_element(score, "movement-title", title or wt)

    ident = _sub_element(score, "identification")
    enc = _sub_element(ident, "encoding")
    _sub_element(enc, "encoding-date", date.today().isoformat())
    _sub_element(enc, "software", "JiuGong AI Music Generator v.1.0")

    defaults = _sub_element(score, "defaults")
    scaling = _sub_element(defaults, "scaling")
    _sub_element(scaling, "millimeters", "7")
    _sub_element(scaling, "tenths", "40")


def _write_attributes(measure: etree.Element):
    """写入第一小节的属性（拍号、谱号、divisions）"""
    attrs = _sub_element(measure, "attributes")
    _sub_element(attrs, "divisions", str(DIVISIONS))

    time = _sub_element(attrs, "time")
    _sub_element(time, "beats", str(BEATS_PER_MEASURE))
    _sub_element(time, "beat-type", str(BEAT_TYPE))

    clef = _sub_element(attrs, "clef")
    _sub_element(clef, "sign", "G")
    _sub_element(clef, "line", "2")


def _write_measure_notes(measure: etree.Element,
                          events: List[_MeasureEvent],
                          is_last_measure: bool,
                          slur_counter: list):
    """写入一个完整小节的音符。slur_counter 是可变的列表 [num]，跨小节共享。"""
    for ev in events:
        note_el = _sub_element(measure, "note")

        # 音高
        pitch_el = _sub_element(note_el, "pitch")
        _sub_element(pitch_el, "step", ev.step)
        _sub_element(pitch_el, "octave", str(ev.octave))

        # 时长
        _sub_element(note_el, "duration", str(ev.xml_duration))
        _sub_element(note_el, "type", ev.musicxml_type)

        if ev.is_dotted:
            _sub_element(note_el, "dot")

        # Tie (跨小节连线)
        if ev.is_tie_start:
            _sub_element(note_el, "tie", attrib={"type": "start"})
        if ev.is_tie_stop:
            _sub_element(note_el, "tie", attrib={"type": "stop"})

        # Notations
        notations = _sub_element(note_el, "notations")

        if ev.is_tie_start:
            _sub_element(notations, "tied", attrib={"type": "start"})
        if ev.is_tie_stop:
            _sub_element(notations, "tied", attrib={"type": "stop"})

        # Slur (拖腔连线) — 用 slur_counter 确保跨小节 number 一致
        if ev.is_melisma_start:
            slur_counter[0] += 1
            _sub_element(notations, "slur",
                         attrib={"number": str(slur_counter[0]), "type": "start"})
        if ev.is_melisma_end:
            _sub_element(notations, "slur",
                         attrib={"number": str(slur_counter[0]), "type": "stop"})

        # 歌词
        if ev.has_lyric and ev.lyric:
            lyric_el = _sub_element(note_el, "lyric",
                                     attrib={"name": "1", "number": "1"})
            _sub_element(lyric_el, "syllabic", "single")
            _sub_element(lyric_el, "text", ev.lyric)

        # Stem
        if ev.musicxml_type in ("eighth", "16th"):
            _sub_element(note_el, "stem", ev.stem)

        # Beam
        if ev.beam:
            _sub_element(note_el, "beam", attrib={"number": "1"}, text=ev.beam)

    # 终小节线
    if is_last_measure:
        barline = _sub_element(measure, "barline", attrib={"location": "right"})
        _sub_element(barline, "bar-style", "light-heavy")


# ============================================================================
# 便捷函数
# ============================================================================


def notes_to_musicxml(
    notes: List[NoteEvent],
    output_path: str,
    title: str = "",
) -> str:
    """
    将扁平音符列表直接转为 MusicXML（不按歌词分组，每个音符独立）。

    适用于没有歌词信息的纯旋律导出。
    """
    # 构造简单的 LyricNoteGroup（每个音符独立）
    groups = []
    for note in notes:
        from .data_loader import LyricNoteGroup
        groups.append(LyricNoteGroup(
            lyric=note.lyric if note.lyric else "",
            lyric_modern_tone=note.lyric_modern_tone,
            lyric_is_entering=note.lyric_is_entering,
            lyric_guangyun_tone=note.lyric_guangyun_tone,
            notes=[note],
        ))

    return gongche_to_musicxml(groups, output_path, title=title)


def song_to_musicxml(
    song: "Song",
    output_path: str,
) -> str:
    """
    将数据集中的 Song 对象导出为 MusicXML。

    可选的 round-trip 测试用。
    """
    title = f"{song.polyu_id}:{song.source}《{song.qupai_bare}》"
    return gongche_to_musicxml(
        song.lyric_note_groups,
        output_path,
        title=title,
        work_title=title,
        qupai=song.qupai_bare,
        source=song.source,
    )

# Need to import at bottom to avoid circular imports
from .data_loader import NoteEvent, LyricNoteGroup, Song
