# 参数手册（bot.py · DEFAULTS）

这份文档由 `bot.py` 里的 `DEFAULTS` 整理而来，包含本轮新增的合批与续聊参数。

**优先级**：`config.json` 的同名键会覆盖这里的默认值，运行时以 `config.json` 为准。
查当前实际生效值，看 `config.json`；查“出厂设置”，看本文。

---

## 目录

- [一、连接层 onebot](#一连接层-onebot)
- [二、模型层 llm](#二模型层-llm)
- [三、人设 persona](#三人设-persona)
- [四、行为 behavior](#四行为-behavior)
- [五、范围控制 groups / users](#五范围控制-groups--users)
- [六、安全 safety](#六安全-safety)
- [七、运行时 runtime](#七运行时-runtime)
- [八、失效与休眠参数](#八失效与休眠参数)
- [九、最常调的六个](#九最常调的六个)
- [十、最小配置示例](#十最小配置示例)

---

## 一、连接层 onebot

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `ws_url` | str | `ws://127.0.0.1:3001` | NapCat 的 WebSocket 地址 |
| `access_token` | str | `""` | 连接 NapCat 的 token，需与 NapCat 端一致；空则不鉴权 |
| `heartbeat` | int | `25` | 心跳间隔（秒），0 = 关闭 |
| `reconnect_min` | int | `3` | 断线后首次重连等待（秒） |
| `reconnect_max` | int | `60` | 重连间隔上限，指数退避 |
| `request_timeout` | int | `15` | 调 OneBot 接口（发消息等）的超时（秒） |

## 二、模型层 llm

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `provider` | str | `deepseek` | 模型来源：`deepseek` 或官方套餐授权 `chatgpt_plan`；不自动回退计费方式 |
| `chatgpt_model` | str | `auto` | 仅套餐模式：选账号可用列表首项，或指定列表中的精确 slug |
| `chatgpt_profile` | str | `""` | 仅套餐模式：空值使用 Windows LocalAppData 的独立加密授权目录，不放仓库 |
| `chatgpt_timeout` | float | `90` | 仅套餐模式：单次 HTTP 超时秒数 |
| `semantic_review_policy` | str | `adaptive` | 普通低风险两步，风险三步；`all`恢复每条独立模型复核 |
| `base_url` | str | `https://api.deepseek.com/v1` | 任何 OpenAI 兼容端点 |
| `api_key` | str | `env:DEEPSEEK_API_KEY` | 支持 `env:变量名` 写法，避免把 key 写进文件 |
| `model` | str | `deepseek-chat` | 模型名 |
| `temperature` | float | `1.15` | 越高越跳脱；1.2~1.35 适合“敢说” |
| `max_tokens` | int | `100` | 单条回复上限，一句话足够 |
| `timeout` | int | `30` | 单次请求超时（秒） |
| `retries` | int | `1` | 失败重试次数；另有 2 次“重复回复换说法”重试 |

## 三、人设 persona

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `name` | str | `大肥鱼` | 人设名，用于提示词与日志 |
| `style` | str | `sharp` | `mild` / `sharp` / `savage`，对应第三层表达风格，不覆盖理解底座 |
| `extra` | str | `""` | 群内黑话、追加风格设定，进第三层，不覆盖对象、语义、协议与边界 |

## 四、行为 behavior

### A. 触发判定

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `at_reply` | bool | `True` | 开启 @ / 引用机器人消息的直接回复候选；仍受过滤与限流约束 |
| `keyword_reply` | bool | `True` | 命中机器人关键词可直接触发；明确面向他人时不抢答 |
| `keywords` | list | 大肥鱼 / 肥鱼 / 若凡 / 若帆 / deepseek / Deepseek / ds | 触发词表 |
| `question_reply` | bool | `False` | 问号是否触发（关闭，避免盖过概率体系） |
| `question_rate` | float | `0.35` | 开启后问号的接话概率 |
| `random_rate` | float | `0.03` | 旧版随机率，仅在未配置 `topic_rates` 时生效 |
| `topic_rates` | dict | 见下 | 按话题的随机接话概率 |

```json
"topic_rates": {
  "daily": {"friendly": 0.10, "attack": 0.05},
  "game":  {"friendly": 0.0,  "attack": 0.10}
}
```

- `daily`（日常）：10% 友好接话，5% 攻击
- `game`（游戏）：10% 攻击，0 友好
- 先判攻击档，再判友好档；剩下的概率不接话

### B. 话题与语气

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `game_keywords` | list | 35 个词（英雄/出装/排位/三角洲…） | 命中即判为“游戏局”，否则“日常” |
| `friendly_tone` | str | 友好接话指令 | “日常·友好”那条的语气覆盖 |
| `attack_tone` | str | 可刻薄 | “日常·攻击”的语气覆盖 |
| `attack_game_tone` | str | 必须贴游戏骂 | “游戏·攻击”的语气覆盖 |
| `greeting_keywords` | list | 早上好/早安/晚安/你好/在吗… | 寒暄识别词 |
| `greeting_tone` | str | 热情、不攻击 | 寒暄时的语气覆盖（优先级高于人设） |

### C. 特殊交互

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `mute_keywords` | list | `["肥鱼闭嘴"]` | 永久忽略这个人，直到他下次 @ |
| `mute_reply` | list | `["呜呜呜，接下来我再也不会理你了"]` | 收到闭嘴指令时的回执 |
| `fixed_replies` | dict | 华为 → 两句随机 | 命中即回固定句，不走模型 |
| `image_replies` | dict | `{}` | 关键词 → 图片规则；支持 `{text,image}` 单条，或 `{variants:[{text,image,weight}]}` 按权重随机，且相邻两次不重复 |
| `image_reply_per_minute` | int | `3` | 图片回复的单群每分钟上限 |
| `combo_replies` | list | `[]` | 组合规则：一条消息同时命中 `keywords` 全部词时优先回复；每项 `{keywords, text, image}` |
| `combo_reply_per_minute` | int | `3` | 组合回复的单群每分钟上限 |
| `sleep_wake_hours` | int | `5` | 多久没互动算“睡着了” |
| `wake_replies` | list | 7 句赖床文案 | 睡醒被 @ 时的随机回复 |
| `quiet_keywords` | list | 破绷/绷不住/哈哈哈/笑死/破防… | 命中就不接话（不触发任何回复） |

### D. 节奏与冷却

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `min_gap` | float | `1.5` | 全局最小发言间隔（秒） |
| `group_cooldown` | int | `25` | 同群两次**主动**发言间隔 |
| `direct_cooldown` | int | `3` | 被 @ / 关键词回复的群内冷却 |
| `user_cooldown` | int | `60` | 同群同一个人（非直接对话）的冷却 |
| `direct_user_cooldown` | int | `3` | 被 @ / 续聊时同群同一人的冷却 |
| `per_group_per_minute` | int | `18` | 单群每分钟上限 |
| `global_per_minute` | int | `25` | 所有群合计上限 |
| `max_chain` | int | `3` | 自己连说 3 句没人接就闭嘴 |
| `merge_delay` | float | `1.6` | 普通分段消息合批等待（秒） |
| `merge_at_delay` | float | `0.6` | 被 @ 时采用的短等待上限（秒） |
| `merge_max_wait` | float | `5.0` | 单批最长合批等待（秒），防止持续发言无限推迟 |
| `merge_max_messages` | int | `20` | 单批最多消息数 |
| `pending_message_limit` | int | `100` | 每群待处理消息上限，限制积压 |

这些时间只控制**合批等待**，不是模型总耗时或从接收至发送的总延迟。同一人的连续分段可合并，交叉发言仍保留各自身份。同群串行、不同群独立。模型生成期间收到当前对象的新补充分段，最多额外重判一次，避免无限重做；不能保证每次都能纳入所有后续消息。

### E. 上下文

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `history_size` | int | `14` | 送入模型的最近消息条数（JSONL） |
| `history_ttl` | int | `900` | 消息在内存里保留的秒数 |

### F. 输出

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `max_chars` | int | `120` | 回复硬截断长度 |
| `prefix_at` | bool | `False` | 回复时是否 @ 对方 |
| `quote_mode` | str | `auto` | `auto` / `always` / `never` |
| `quote_busy_per_minute` | int | `12` | auto 模式下，群消息超过这个数才引用 |
| `quote_reply` | bool | `True` | **已废弃**，代码里不再读取 |

### G. 杂项

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `reply_poke` | bool | `True` | 被戳一戳是否回 |
| `ignore_at_all` | bool | `True` | 忽略 @全体成员 |
| `followup_window` | int | `120` | 秒；同群同对象的隐式续聊总窗口，从机器人上次成功回复起算 |
| `followup_rate` | float | `1.0` | 续聊判定成立后的接话概率；1 表示不随机漏接，仍受限流和安全规则约束 |
| `followup_max_turns` | int | `6` | 连续隐式续聊的最大轮数；明确 @ / 呼叫可重新建立会话 |
| `followup_grace_window` | float | `30` | 秒；同一对象、无人插话时不要求话题共词，允许自然接话和换聊 |
| `taunt_keywords` | list | 你妈/傻逼/废物/滚/草… | 只有对象和话题证据成立时才作为续聊候选，仍受冷却限流 |

**对象判断优先于概率**：明确 @ / 引用其他成员时默认不插嘴；引用作者未知时保守不主动。机器人成功回复后，同一 QQ 在 30 秒内且无人插话，可以自然承接或换聊，不必重复关键词。超过 30 秒但未超过 120 秒，需要话题词重合或“找你聊天 / 你刚才说”等承接线索。其他成员插话、当前对象转向别人，或明确“不聊了 / 不用回 / 换个话题”时结束隐式续聊；明确 @ / 引用机器人可重新建立会话。默认最多连续 6 轮，命中后不随机漏接。续聊遇到 3 秒直聊冷却会等待而非丢弃；等待或生成期间换对象，发送前取消旧续聊。每群每分钟等硬限流仍保留，故并非任何消息都保证回答。所有判断使用本地规则，不新增模型分类调用；隐含对象仍可能误判。

回归检查：`python selftest.py` 和 `python dialogue_selftest.py`。图片与固定回复设置保留。本轮不修改模型接口兼容性。


### H. 兼容模式：随机补发（仅 model_reply_planning=false）

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `second_reply_rate` | float | `0.6` | 仅旧兼容模式生效；规划模式忽略此随机概率 |
| `second_reply_delay_min` | float | `0.8` | 秒；补发生成前的最短等待 |
| `second_reply_delay_max` | float | `0.8` | 秒；补发生成前的最长等待 |

只抽一次、最多追加一条，不会递归第三条。沿用首条的人设与临时语气，原本友好时不会突然转攻击；互怼时可以补一句不同角度的短调侃。与首条完全重复、模型表示无合适补充、输入明确结束、首条发送失败，均不补发。固定回复、图片、静音确认和唤醒语不走此机制。

同群收到新成员消息（含图片/表情/转发）或有效戳一戳时取消旧补发，其他群不影响；生成中也会取消旧生成任务，发送前再次核对对象。补发占用同一每群 18 条/分钟、全局 25 条/分钟限额，也受 `max_chain` 约束，不刷新隐式续聊轮数或窗口。60% 是资格满足后的触发概率，不保证实际发送比例恰好 60%。

固定等待 0.8 秒后才开始补充生成，模型耗时与发送间隔另计。补发会额外进行一次模型回复生成（内部重试仍可能发生），因此会增加 token 开销；新消息在生成期间到达前，已经产生的调用也可能计费。


### I. 模型自主话轮规划（默认启用）

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `model_reply_planning` | bool | `true` | 用模型同次生成的发言决定替代随机补发；false 回到 H 节兼容模式 |
| `model_burst_max_messages` | int | `3` | 一次选中话轮的发言保险上限，含首条；不是每次必须发满三条 |

程序通过本地对象、输入、冷却与限流门禁后，模型返回 `action/text/next`：`reply` 发一条短消息，`wait` 等群友新消息，`stop` 结束这次主动发言。`next=reply` 只表示有意继续；等 0.8 秒后，模型仍可重新选择 wait/stop。每条续说都重新裁决，已有内容说完整就停，不能为了凑条数重复、自问自答或编造群友回复。wait/stop 不会触发定时轮询。

规划模式不读取旧 60% 抽签；可说 0、1、2、3 条。总条数同时受 `model_burst_max_messages` 和 `max_chain` 限制。新群消息/有效戳一戳取消旧链，发送锁等待后仍复核对象；每次发送尝试按单群 18、全局 25 条/分钟原子扣额，模型选择等待不占发送配额。接口失败的发送尝试可能保守占额，但不写入成功历史。续条不重置隐式续聊计数和窗口。

每次请求同时生成短回复和发言决定，不另外调用一个分类模型；后续真要说话时仍会增加生成调用。`llm.planning_max_tokens=320` 为 JSON 协议留出预算，规划请求使用320、它和原 `llm.max_tokens` 中的最大值（旧配置192也会保底320），避免短回复被协议包络挤断；普通非规划请求仍用原值。上限不是必然消耗量。纯文本或 reply 缺 next 时先纠正一次，仍不合协议才警告并保守退回单条、next=wait（可能额外请求一次）；畸形 JSON 不直接发到群里，内部动作字段也不对群友展示。INFO 决策日志包含 source/action/next、已发条数与停止原因，格式降级及输出截断会警告，不额外记录正文或密钥。

此机制只对选中进入模型的正常回复生效，不等于对每条群消息都调用模型。固定/图片/静音确认/唤醒设置保留。普通群聊的本地筛选、冷却、明确对他人说话和结束信号仍先处理。

## 五、范围控制 groups / users

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `groups.whitelist` | list[int] | `[]` | 非空则只在这些群工作 |
| `groups.blacklist` | list[int] | `[]` | 永不发言的群 |
| `users.blacklist` | list[int] | `[]` | 永不回复的人 |

## 六、安全 safety

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `blocked_input` | list[str] | 裸聊 / 约炮 | 消息命中直接吞掉，不回复、不花钱 |
| `blocked_output` | list[str] | 习近平 / 共产党 / 六四 / 台独 / 法轮功 / 自杀 | 模型输出命中则改发兜底句 |
| `fallback_reply` | str | `君子谨言慎行` | 兜底句 |

匹配方式是**纯子串匹配**，不分词、不区分大小写感知（英文区分大小写）。
「开 盒」这种中间加空格的写法挡不住。

## 七、运行时 runtime

| 参数 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `dry_run` | bool | `False` | 只打印不发送，调人设时用 |
| `log_file` | str | `logs/bot.log` | 日志路径（5MB × 3 轮转） |
| `log_level` | str | `INFO` | DEBUG / INFO / WARNING |

## 八、失效与休眠参数

| 参数 | 状态 | 说明 |
|---|---|---|
| `quote_reply` | 死参数 | 只剩定义，代码不读（被 `quote_mode` 取代） |
| `random_rate` | 休眠 | 配了 `topic_rates` 就用不到它 |
| `question_rate` | 休眠 | `question_reply=False` 时不生效 |
| `focus_size` | 死参数 | 只存在于 config.json，`bot.py` 无此默认值也不读取 |

## 九、最常调的六个

1. **`topic_rates`** —— 改随机接话概率（日常/游戏分开）
2. **`group_cooldown`** —— 改“话多不多”的第一旋钮
3. **`history_size`** —— 改上下文长度；JSONL 下 14 条约 700~900 token
4. **`persona.style`** —— 整体凶不凶（mild / sharp / savage）
5. **`greeting_tone` / `attack_game_tone`** —— 最有效的两个语气开关
6. **`mute_keywords` / `quiet_keywords`** —— 什么时候该闭嘴

## 十、最小配置示例

```json
{
  "onebot": {
    "ws_url": "ws://127.0.0.1:3001",
    "access_token": ""
  },
  "llm": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "env:DEEPSEEK_API_KEY",
    "model": "deepseek-chat",
    "temperature": 1.2
  },
  "persona": {
    "name": "大肥鱼",
    "style": "sharp"
  },
  "behavior": {
    "topic_rates": {
      "daily": {"friendly": 0.10, "attack": 0.05},
      "game":  {"friendly": 0.0,  "attack": 0.10}
    },
    "group_cooldown": 25,
    "history_size": 12
  },
  "groups": {
    "whitelist": [123456789],
    "blacklist": []
  },
  "safety": {
    "fallback_reply": "君子谨言慎行"
  }
}
```

> 注意：`config.json` 里不要提交真实 `api_key`。用 `"env:DEEPSEEK_API_KEY"`，或把 `config.json` 加进 `.gitignore`。


### 不确定语义与回复依据

规划协议的 understanding 在 act/target 之外增加可选 meaning（短含义摘要，最多120字）和 unresolved（未明确关系，最多100字），旧协议仍兼容。回复不把未确认的因果、时间、对象当事实；能自然承接共有意思时直接聊，只有歧义实质影响回答才询问。摘要不发送到QQ，同轮补发参考首次可用的摘要，新用户发言及明确纠正重新理解。不新增独立分类请求；额外字段占用同次生成预算。离线测试只证明协议与传递机制，不证明真实模型每次理解正确。


## 中性理解优先流水线（默认启用）

`llm.semantic_pipeline=true` 始终先独立做不含人格的中性理解。新增 `llm.semantic_review_policy="adaptive"`（默认）：明确、低风险的普通聊天在理解后生成回复，再通过本地检查放行，顺利首条通常2次模型调用；纠正、歧义、互怼、第三人／引用／戳动作、近期夹话、可疑草稿或格式修复仍做独立模型复核，顺利首条通常3次。续条复用理解，仍保留表达＋复核，一般2次。失败或重写可能增加调用；不为卡死次数而放行坏格式或被否定草稿。

要恢复每条都复核，设置 `llm.semantic_review_policy="all"`；未知策略值也保守按严格模式处理，旧显式 `semantic_review=true` 保持严格。快速路径的 `review=local` 不等于模型已经复核，也不是语义准确性的证明：它是速度与质量的折中；需要用实际旧失败样本持续验证，不能保证每句都能少一次请求。原文和上下文会独立检查，不只相信理解摘要里的normal标签。

理解阶段温度0.2，复核0，表达温度最多0.8且不随重试升高；接口可能忽略采样参数，降低温度不等于保证理解正确。理解摘要不作为永久记忆，新用户原话与明确纠正重新理解。普通/纠错/澄清模式不加载攻击性附加人格，互怼模式仍可保留机灵反击。不要再配置“每句必须下结论、攻击对方本人”等旧强指令。

戳一戳是动作事件，不冒充文字；仅在同一成员最后一条原话近期未被回答时关联，`behavior.poke_context_window=45`秒。其他成员插话、已回答、缺失身份/消息ID、超时则不关联旧话。最近上下文默认28条（含机器人），仍非无限记忆。

`unresolved` 中已标记不确定的时序/因果会接受保守关系检查，防止摘要、表达及模型复核同时把猜测当事实。该检查跨主题但只是启发式，可能误拦，不能替代真实模型评测。日志记录每阶段 elapsed_ms、选中模型标识（若已知）、review=local/model、分流原因、调用次数及流程 total_ms；不额外记录原文、摘要正文、令牌或密钥。total_ms是语义流程耗时，不含消息合并／排队，也不是完整消息端到端延迟。

离线运行：`python selftest.py`、`python dialogue_selftest.py`、`python semantic_pipeline_tests.py .`、`python adaptive_review_tests.py .`。旧控制器回归明确使用兼容路径；自检假HTTP及新流水线测试覆盖默认开启路径。离线通过不等于语义全对；应使用匿名样本对照真实接口，并保留失败样本。


## 通知与提醒（notifications）

`enabled=true` 默认启用；`database="logs/reminders.sqlite3"` 为本地持久化数据库；`poll_interval=1.0` 秒检查一次，不调用模型；`grace_seconds=300` 为迟到窗口（0～900 秒）。立即/定时 @ 均有正文，最多5个真实 @ 对象，不允许 @all；真实身份只取当前发送者和 OneBot @ 段。限流与普通聊天共用单群/全局额度；定时通知不更改当前对话对象。详见 @与定时提醒（独立说明已移除）。
