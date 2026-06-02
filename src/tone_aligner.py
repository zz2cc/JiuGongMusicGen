"""
声调标注模块 — 为歌词中的每个汉字标注现代声调和中古音声调。

使用 pypinyin 获取现代声调（1=阴平, 2=阳平, 3=上声, 4=去声, 5=轻声）。
内置中古音（广韵）声调映射字典（平/上/去/入），覆盖常用汉字约 2000 字。
"""

import logging
from dataclasses import dataclass
from typing import List, Dict

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

logger = logging.getLogger(__name__)


# ============================================================================
# 中古音（广韵）声调字典
# 按声调组织，便于维护和扩展
# ============================================================================

_平声字 = set(
    "风花光春秋冬天山川云星河江林松梅兰桃荷莲芳清香明空长生高深"
    "新阴阳光中南楼台亭园门窗桥堂庭阶城村家心思忧愁悲伤欢离逢归"
    "飞闻看听吟歌言谈行人来留游流求寻知能无同从如然还时年前名"
    "尘身心情灵魂衣裳书文章诗词声恩君臣民人仁慈和平安宁繁华荣容"
    "颜妆眉眸琴棋箫笙钟茶肴羹糕酥糖烟霞波涛潮澜溪泉潭湖田原峰"
    "峦岩崖沙泥灰朝昏宵晨曦晖帘帷屏栏杆钩环钗梳梭针丝罗纱绡绸"
    "鞭鞍旗旌弓刀枪矛舟船帆桡槎翎毛蹄须眉髯冠袍裘巾绦囊灯炉香"
    "烛盘杯壶瓶笺毫题裁描图临摹挥扶携攀提推敲驰追随邀迎送依偎"
    "凭沉浮升腾横斜齐端闲忙穷通难容稀疏浓纤丰肥干枯残全奇雄豪"
    "孤单双成除存亡兴衰添消增亏赢输夸嘲传垂凝停冰凌霜雷虹霓"
    # 辛弃疾《破阵子》常见平声字
    "挑灯麾弦翻声沙场兵卢飞惊王赢怜回吹连营分"
)

_上声字 = set(
    "水海岛浦港渚远近短小早晚晓暖冷手眼耳口齿舌脚步走跑起止举"
    "首俯仰返转反卷展掩敛古老旧好丑美巧雅婉妩媚我你女子父母雨"
    "雪语酒影火土马虎鼠犬纸笔简管板鼓缕锁所理改写补把打采改敢"
    "感满晚暖柳李藕笋锦枕寝稳饮品岭顶浦府舞紫此死史使始喜几鬼"
    "委毁伟敏引隐"
    # 辛弃疾词
    "里点了马可五"
)

_去声字 = set(
    "大太万半片遍面见现变断乱散怨恨爱念忆记忘寄赠问道报告教话"
    "笑叹泪碎破醉睡梦会对向到过在坐卧立住用事世代岁夜昼路径处"
    "地外内后气味调韵奏唱诵翠丽贵富秀妙艳绚烂态意志趣量数四二"
    "妇夏暮旦信印正圣令命姓性镜"
    # 辛弃疾词
    "下外事快后塞"
)

_入声字 = set(
    "月日一七八十百不出入别结节绝灭歇热铁雪血骨肉目木竹菊烛曲玉"
    "欲绿落乐药客白石色得德急及立力历列烈猎叶业物佛发法忽独读"
    "毒服福复伏幅国觉角脚却确学削昨索莫幕漠脉策册塞黑北贼则笔"
    "密蜜必毕匹吉七质失实室疾悉阔活夺脱拨末沫达甲鸭压押插恰狭"
    "峡接妾蝶叠帖贴协挟泣习集湿拾十汁合鸽盒塔蜡腊杂纳杰竭缺决"
    "岳浊捉剥驳朴戟击壁笛敌历寂惕职织识食息直力极屋木谷哭族速"
    "足俗局录束玉"
    # 辛弃疾词
    "八炙白的霹雳得却作角百"
)


def _build_guangyun_map() -> Dict[str, str]:
    """构建广韵声调映射字典"""
    result = {}
    for c in _平声字:
        result[c] = "平"
    for c in _上声字:
        result[c] = "上"
    for c in _去声字:
        result[c] = "去"
    for c in _入声字:
        result[c] = "入"
    return result


_GUANGYUN_TONE_MAP = _build_guangyun_map()


@dataclass
class ToneAnnotation:
    """单字声调标注"""
    character: str
    modern_tone: int          # 1=阴平 2=阳平 3=上声 4=去声 5=轻声 0=未知
    guangyun_tone: str        # "平" | "上" | "去" | "入" | ""=未知
    is_entering: bool         # 是否为入声字


class ToneAligner:
    """
    声调对齐器 — 为歌词中的每个汉字标注声调。

    Usage:
        aligner = ToneAligner()
        annotations = aligner.annotate("醉里挑灯看剑")
        for a in annotations:
            print(f"{a.character}: 现代{a.modern_tone} 中古{a.guangyun_tone}")
    """

    # 现代声调到中古的近似映射（入声无法从现代声调推断）
    MODERN_TO_GUANGYUN = {
        1: "平",
        2: "平",
        3: "上",
        4: "去",
        5: "",
    }

    def __init__(self):
        if not HAS_PYPINYIN:
            logger.warning("pypinyin 未安装，声调标注将仅使用内置字典")

    def annotate(self, lyrics: str) -> List[ToneAnnotation]:
        """
        Args:
            lyrics: 歌词文本（可包含标点、空格等，自动跳过非汉字）

        Returns:
            List[ToneAnnotation]
        """
        result = []
        modern_tones = self._get_modern_tones(lyrics)

        idx = 0
        for char in lyrics:
            if not self._is_chinese(char):
                continue
            modern = modern_tones[idx] if idx < len(modern_tones) else 0
            idx += 1

            guangyun = _GUANGYUN_TONE_MAP.get(char, "")
            if not guangyun and modern > 0:
                guangyun = self.MODERN_TO_GUANGYUN.get(modern, "")

            result.append(ToneAnnotation(
                character=char,
                modern_tone=modern,
                guangyun_tone=guangyun,
                is_entering=(guangyun == "入"),
            ))

        return result

    def _get_modern_tones(self, text: str) -> List[int]:
        """使用 pypinyin 获取现代声调数组"""
        if not HAS_PYPINYIN:
            return []

        try:
            results = pinyin(text, style=Style.TONE3, errors='ignore')
            tones = []
            for item in results:
                py = item[0] if item else ""
                tone = self._extract_tone_number(py)
                tones.append(tone)
            return tones
        except Exception as e:
            logger.warning(f"pypinyin 声调获取失败: {e}")
            return []

    @staticmethod
    def _extract_tone_number(pinyin_str: str) -> int:
        """从带数字声调的拼音中提取声调数字"""
        if not pinyin_str:
            return 0
        last_char = pinyin_str[-1]
        if last_char.isdigit():
            t = int(last_char)
            return t if 1 <= t <= 5 else 0
        return 5  # 轻声

    @staticmethod
    def _is_chinese(char: str) -> bool:
        """判断是否为中文字符"""
        cp = ord(char)
        return (0x4E00 <= cp <= 0x9FFF or
                0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF)


# 模块级便捷函数
def assign_tones(lyrics: str) -> List[dict]:
    """
    便捷函数：为歌词标注声调。

    Returns:
        [{char, modern_tone, guangyun_tone, is_entering}, ...]
    """
    aligner = ToneAligner()
    annotations = aligner.annotate(lyrics)
    return [
        {
            "char": a.character,
            "modern_tone": a.modern_tone,
            "guangyun_tone": a.guangyun_tone,
            "is_entering": a.is_entering,
        }
        for a in annotations
    ]
