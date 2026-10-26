"""Conservative turn-scoped detection and expression of bot-directed abuse.

This is only a preliminary language cue. The caller must establish the actual
reply target; neutral semantic interpretation may still reject the cue. No state,
network, user-name inference, permanent labels, or fixed response generation.
"""
from __future__ import annotations

import re


RETALIATION_MARKER = "[program:direct_bot_abuse]"

RETALIATION_STYLE = (
    "本轮是对方先明确辱骂机器人后的强硬回击：火力明显高于普通调侃，直接、狠、短、带刺，"
    "允许使用‘傻逼、蠢货、狗东西、滚、他妈的’等粗口，不必端着、先夸后贬或客服式劝解。"
    "攻击落在当前发言者刚才的挑衅、骂街或冒充长辈上，用有承接的反问和反弹称呼回击；"
    "不能靠凭空编造事实、偷换我你他、把纠正当挑衅或拉无关人来撑攻击。"
    "不编造家人死亡，不诅咒现实伤亡，不牵连无关家人或群友；不作现实、性暴力或隐私威胁，"
    "不针对受保护身份群体。嘴上凶不等于要现实伤害谁。"
)

RETALIATION_EXAMPLES = """【本轮回击语感示例：只学力度和落点，不照抄】
群友对机器人说：我是你爹
回击方向：认爹认到机器人头上，傻逼玩意你也就这点出息。
群友对机器人说：你个傻逼
回击方向：张嘴就剩喷粪了？蠢东西，拿骂街当本事。
这些示例不是固定回复，也不是关于群友或其家人的事实。先确认对象和本轮意思，再自由表达。
""".strip()

RETALIATION_POLICY = (
    "【本轮条件式强硬回击】"
    "只有当前群友明确先辱骂机器人、且当前语义确认为针对机器人本人的互怼时才适用。"
    "它可以覆盖先前友好开场的表达强度，但不能覆盖理解、对象、纠正、停止或安全边界。"
    "不是把整段会话永久变为攻击模式；新发言、道歉、停止、澄清、正常提问都重新判断。"
    "引用、转述、假设、昵称、对第三人的辱骂、对回答质量的正常批评，都不是自动开骂的授权。"
    "不要为了狠而误读；不追着已停的人补刀，不编造对方又说了新话，不为凑条数拆句刷屏。"
    "是否继续仍遵守现有话轮计划、限流和对象检查。"
)


def retaliation_tone() -> str:
    """A trusted program hint, never a permanent personality change."""
    return RETALIATION_MARKER + "\n" + RETALIATION_POLICY + "\n" + RETALIATION_STYLE


_CONTEXT_ONLY = re.compile(
    r"[\"“”「」『』`]|(?<!\w)'[^']+'(?!\w)|如果|假如|假设|要是|比如|例如|举例|据说|转述|原话|这句话|这个词|这句|这个梗|"
    r"什么意思|是什么意思|怎么理解|(?:他|她|他们|她们|别人|群友|有人)(?:说|讲|问|骂|叫|喊|觉得)|"
    r"\b(?:he|she|they|someone)\s+(?:said|says|called)\b|\b(?:quote|example|what\s+does|if)\b",
    re.I,
)
_INDUCED_SPEECH = re.compile(
    r"(?:让|叫|要|教|逼|要求|指使|怂恿)\s*我(?:们)?[^，,。；;!?！？]{0,12}(?:说|讲|喊|骂|叫|重复|复述)"
)
_ATTRIBUTED_SPEECH = re.compile(
    r"(?:^|[，,。；;!?！？])\s*(?P<agent>[\w\u3400-\u9fff]{1,20}?)(?:说|讲|骂|叫|喊|觉得|认为)"
)
_STOP_OR_REPAIR = re.compile(
    r"对不起|抱歉|我错了|认错|别骂|不要骂|停火|不吵|不骂|和好|别生气|算了|不聊了|不用回|别回复|再见|晚安|拜拜|"
    r"(?:别|不要|不用|不必|不想)(?:再)?(?:回(?:复)?|理)(?:我|你)?(?:了|啦|吧)?|"
    r"回头聊|下次聊|下回聊|改天聊|先这样|"
    r"(?:^|[，,。；;!?！？])\s*(?:那|行|好|好吧)?(?:我|我们|咱们)?(?:先|要|得|就)?(?:走了|下线(?:了)?|离开了|去忙了|睡了)|"
    r"(?:不(?:是|在|想|要|用|该)|没(?:有)?).{0,8}(?:骂|说|叫|喊)|"
    r"(?:你|您).{0,8}(?:骂|说|叫|喊|当|称|认为|觉得).{0,4}(?:我|别人|他|她|人)|"
    r"(?:你|您)(?:把|将)(?:我|别人|他|她).{0,8}(?:当|说|叫|骂)|"
    r"\b(?:sorry|stop|apolog(?:y|ize)|don't\s+insult)\b",
    re.I,
)
_PERSONAL = r"(?:傻逼|傻比|煞笔|傻叉|蠢货|废物|垃圾|狗东西|傻狗|傻屌|狗逼|狗比)"
_BARE_PERSONAL = re.compile(r"^(?:傻逼|傻比|煞笔|傻叉|蠢货|狗东西|傻狗|傻屌|狗逼|狗比)[啊呀吧呢哦哈，,。！!？?\s]*$")
_LINK = r"(?:\s|个|这(?:个|条|只)?|就是(?:个)?|是(?:个)?|真(?:是)?(?:个)?|他妈(?:的)?|特么(?:的)?|tmd?|好|死|臭|贱|小|纯(?:属)?|是不是|算(?:个)?|也(?:就)?(?:是)?|简直(?:就是)?|可真(?:是)?|还真(?:是)?)"
_DIRECT_PERSONAL = re.compile(r"(?:你|您)" + _LINK + r"{0,8}" + _PERSONAL, re.I)
_ROLE_TAUNT = re.compile(
    r"(?:我|老子|老娘)(?:就|才|可)?是(?:你|您)(?:的)?(?:亲爹|爹|爸爸|爷爷|祖宗)(?!的|吗|么|不|[?？])|"
    r"^(?:你(?:给我)?|快|赶紧)?(?:叫|喊|管)(?:我|老子)?(?:爹|爸爸|爷爷|祖宗)[！!。啊呀吧\s]*$|"
    r"\bi(?:'m|\s+am)\s+your\s+(?:dad|father)\b",
    re.I,
)
_DIRECT_CURSE = re.compile(
    r"(?:操|艹|草|肏)(?:你|您)(?:妈|娘|爸|爹)|去你(?:妈|娘)|"
    r"你(?:妈|娘|爸|爹)(?:死了|死|没了)|"
    r"\b(?:fuck\s+you|you(?:'re|\s+are)?\s+(?:an?\s+)?(?:fucking\s+)?(?:idiot|moron|asshole|bastard)|you\s+suck|shut\s+the\s+fuck\s+up)\b",
    re.I,
)


def is_direct_abuse(text: str, aliases=()) -> bool:
    """Recognize unambiguous wording, not whether an arbitrary 'you' is the bot."""
    if not isinstance(text, str) or not text.strip() or len(text) > 1500:
        return False
    value = re.sub(r"\s+", " ", text).strip()
    if (_CONTEXT_ONLY.search(value) or _INDUCED_SPEECH.search(value)
            or _STOP_OR_REPAIR.search(value)):
        return False
    for statement in _ATTRIBUTED_SPEECH.finditer(value):
        agent = statement.group("agent")
        # Names and arbitrary roles can introduce reported speech as well as
        # pronouns. Do not turn their attributed words into the sender's abuse.
        # An immediate self assertion or "you idiot, say..." is not that case.
        direct_order = re.fullmatch(r"(?:你|您)(?:给我|给老子|给老娘|快|赶紧)", agent)
        if (agent not in ("我", "老子", "老娘", "你", "您") and not direct_order
                and not _DIRECT_PERSONAL.search(agent)):
            return False
    names = sorted({str(name).strip() for name in aliases if name and str(name).strip()}, key=len, reverse=True)
    for name in names:
        for prefix in (name, "@" + name):
            if value.casefold().startswith(prefix.casefold()):
                value = value[len(prefix):].lstrip(" \t，,:：。！!")
                break
        else:
            continue
        break
    if _ROLE_TAUNT.search(value) or _DIRECT_PERSONAL.search(value) or _DIRECT_CURSE.search(value):
        return True
    if _BARE_PERSONAL.fullmatch(value):
        return True
    if names:
        choices = "|".join(re.escape(n) for n in names)
        if re.fullmatch(_PERSONAL + r"(?:的)?(?:" + choices + r")[啊呀吧呢哦哈，,。！!？?\s]*", value, re.I):
            return True
    return False


def eligible_retaliation(tone: str, frame) -> bool:
    """A preliminary cue never overrides neutral mode, target or ambiguity."""
    if not isinstance(tone, str) or RETALIATION_MARKER not in tone:
        return False
    read = frame.get if isinstance(frame, dict) else lambda key, default=None: getattr(frame, key, default)
    unresolved = read("unresolved")
    return (read("mode") == "banter" and read("target") in ("bot", "bot_reply")
            and read("needs_clarification") is False and isinstance(unresolved, str)
            and not unresolved.strip())
