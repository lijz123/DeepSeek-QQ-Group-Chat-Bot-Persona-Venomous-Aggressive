# 参数手册（bot.py · DEFAULTS）

这份文档由 `bot.py` 里的 `DEFAULTS` 整理而来，共 **68 个参数**。

**优先级**：`config.json` 的同名键会覆盖这里的默认值，运行时以 `config.json` 为准。#所以你改的时候，记得json文件里的参数也要同时改掉

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

| 参数                | 类型  | 默认                    | 作用                                    |
| ----------------- | --- | --------------------- | ------------------------------------- |
| `ws_url`          | str | `ws://127.0.0.1:3001` | NapCat 的 WebSocket 地址                 |
| `access_token`    | str | `""`                  | 连接 NapCat 的 token，需与 NapCat 端一致；空则不鉴权 |
| `heartbeat`       | int | `25`                  | 心跳间隔（秒），0 = 关闭                        |
| `reconnect_min`   | int | `3`                   | 断线后首次重连等待（秒）                          |
| `reconnect_max`   | int | `60`                  | 重连间隔上限，指数退避                           |
| `request_timeout` | int | `15`                  | 调 OneBot 接口（发消息等）的超时（秒）               |

## 二、模型层 llm

| 参数            | 类型    | 默认                            | 作用                           |
| ------------- | ----- | ----------------------------- | ---------------------------- |
| `base_url`    | str   | `https://api.deepseek.com/v1` | 任何 OpenAI 兼容端点               |
| `api_key`     | str   | `env:DEEPSEEK_API_KEY`        | 支持 `env:变量名` 写法，避免把 key 写进文件 |
| `model`       | str   | `deepseek-chat`               | 模型名                          |
| `temperature` | float | `1.15`                        | 越高越跳脱；1.2~1.35 适合“敢说”        |
| `max_tokens`  | int   | `100`                         | 单条回复上限，一句话足够                 |
| `timeout`     | int   | `30`                          | 单次请求超时（秒）                    |
| `retries`     | int   | `1`                           | 失败重试次数；另有 2 次“重复回复换说法”重试     |

## 三、人设 persona

| 参数      | 类型  | 默认      | 作用                                    |
| ------- | --- | ------- | ------------------------------------- |
| `name`  | str | `大肥鱼`   | 人设名，用于提示词与日志                          |
| `style` | str | `sharp` | `mild` / `sharp` / `savage`，对应第一层核心人格 |
| `extra` | str | `""`    | 群内黑话、追加设定，进第三层                        |

## 四、行为 behavior

### A. 触发判定

| 参数               | 类型    | 默认                                            | 作用                            |
| ---------------- | ----- | --------------------------------------------- | ----------------------------- |
| `at_reply`       | bool  | `True`                                        | 被 @ 必回                        |
| `keyword_reply`  | bool  | `True`                                        | 命中关键词必回                       |
| `keywords`       | list  | 大肥鱼 / 肥鱼 / 若凡 / 若帆 / deepseek / Deepseek / ds | 触发词表                          |
| `question_reply` | bool  | `False`                                       | 问号是否触发（关闭，避免盖过概率体系）           |
| `question_rate`  | float | `0.35`                                        | 开启后问号的接话概率                    |
| `random_rate`    | float | `0.03`                                        | 旧版随机率，仅在未配置 `topic_rates` 时生效 |
| `topic_rates`    | dict  | 见下                                            | 按话题的随机接话概率                    |

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

| 参数                  | 类型   | 默认                   | 作用                |
| ------------------- | ---- | -------------------- | ----------------- |
| `game_keywords`     | list | 35 个词（英雄/出装/排位/三角洲…） | 命中即判为“游戏局”，否则“日常” |
| `friendly_tone`     | str  | 友好接话指令               | “日常·友好”那条的语气覆盖    |
| `attack_tone`       | str  | 可刻薄                  | “日常·攻击”的语气覆盖      |
| `attack_game_tone`  | str  | 必须贴游戏骂               | “游戏·攻击”的语气覆盖      |
| `greeting_keywords` | list | 早上好/早安/晚安/你好/在吗…     | 寒暄识别词             |
| `greeting_tone`     | str  | 热情、不攻击               | 寒暄时的语气覆盖（优先级高于人设） |

### C. 特殊交互

| 参数                 | 类型   | 默认                    | 作用              |
| ------------------ | ---- | --------------------- | --------------- |
| `mute_keywords`    | list | `["肥鱼闭嘴"]`            | 永久忽略这个人，直到他下次 @ |
| `mute_reply`       | list | `["呜呜呜，接下来我再也不会理你了"]` | 收到闭嘴指令时的回执      |
| `fixed_replies`    | dict | 华为 → 两句随机             | 命中即回固定句，不走模型    |
| `sleep_wake_hours` | int  | `5`                   | 多久没互动算“睡着了”     |
| `wake_replies`     | list | 7 句赖床文案               | 睡醒被 @ 时的随机回复    |
| `quiet_keywords`   | list | 破绷/绷不住/哈哈哈/笑死/破防…     | 命中就不接话（不触发任何回复） |

### D. 节奏与冷却

| 参数                     | 类型    | 默认    | 作用               |
| ---------------------- | ----- | ----- | ---------------- |
| `min_gap`              | float | `1.5` | 全局最小发言间隔（秒）      |
| `group_cooldown`       | int   | `25`  | 同群两次**主动**发言间隔   |
| `direct_cooldown`      | int   | `3`   | 被 @ / 关键词回复的群内冷却 |
| `user_cooldown`        | int   | `60`  | 同一个人（非直接对话）冷却    |
| `direct_user_cooldown` | int   | `3`   | 被 @ / 对线时的同一人冷却  |
| `per_group_per_minute` | int   | `6`   | 单群每分钟上限          |
| `global_per_minute`    | int   | `25`  | 所有群合计上限          |
| `max_chain`            | int   | `3`   | 自己连说 3 句没人接就闭嘴   |

### E. 上下文

| 参数             | 类型  | 默认    | 作用                 |
| -------------- | --- | ----- | ------------------ |
| `history_size` | int | `14`  | 送入模型的最近消息条数（JSONL） |
| `history_ttl`  | int | `900` | 消息在内存里保留的秒数        |

### F. 输出

| 参数                      | 类型   | 默认      | 作用                          |
| ----------------------- | ---- | ------- | --------------------------- |
| `max_chars`             | int  | `120`   | 回复硬截断长度                     |
| `prefix_at`             | bool | `False` | 回复时是否 @ 对方                  |
| `quote_mode`            | str  | `auto`  | `auto` / `always` / `never` |
| `quote_busy_per_minute` | int  | `12`    | auto 模式下，群消息超过这个数才引用        |
| `quote_reply`           | bool | `True`  | **已废弃**，代码里不再读取             |

### G. 杂项

| 参数                | 类型    | 默认            | 作用             |
| ----------------- | ----- | ------------- | -------------- |
| `reply_poke`      | bool  | `True`        | 被戳一戳是否回        |
| `ignore_at_all`   | bool  | `True`        | 忽略 @全体成员       |
| `followup_window` | int   | `90`          | 刚聊完后多久内算“还在对线” |
| `followup_rate`   | float | `0.85`        | 对线期内再接话概率      |
| `taunt_keywords`  | list  | 你妈/傻逼/废物/滚/草… | 对线期内命中必回       |

## 五、范围控制 groups / users

| 参数                 | 类型        | 默认   | 作用         |
| ------------------ | --------- | ---- | ---------- |
| `groups.whitelist` | list[int] | `[]` | 非空则只在这些群工作 |
| `groups.blacklist` | list[int] | `[]` | 永不发言的群     |
| `users.blacklist`  | list[int] | `[]` | 永不回复的人     |

## ==六、安全 safety==

| 参数               | 类型        | 默认                             | 作用               |
| ---------------- | --------- | ------------------------------ | ---------------- |
| `blocked_input`  | list[str] | 裸聊 / 约炮                        | 消息命中直接吞掉，不回复、不花钱 |
| `blocked_output` | list[str] | 习近平 / 共产党 / 六四 / 台独 / 法轮功 / 自杀 | 模型输出命中则改发兜底句     |
| `fallback_reply` | str       | `君子谨言慎行`                       | 兜底句              |

==匹配方式是**纯子串匹配**，不分词、不区分大小写感知（英文区分大小写）。
「开 盒」这种中间加空格的写法挡不住。==

## 七、运行时 runtime

| 参数          | 类型   | 默认             | 作用                     |
| ----------- | ---- | -------------- | ---------------------- |
| `dry_run`   | bool | `False`        | 只打印不发送，调人设时用           |
| `log_file`  | str  | `logs/bot.log` | 日志路径（5MB × 3 轮转）       |
| `log_level` | str  | `INFO`         | DEBUG / INFO / WARNING |

## 八、失效与休眠参数

| 参数              | 状态  | 说明                                  |
| --------------- | --- | ----------------------------------- |
| `quote_reply`   | 死参数 | 只剩定义，代码不读（被 `quote_mode` 取代）        |
| `random_rate`   | 休眠  | 配了 `topic_rates` 就用不到它              |
| `question_rate` | 休眠  | `question_reply=False` 时不生效         |
| `focus_size`    | 死参数 | 只存在于 config.json，`bot.py` 无此默认值也不读取 |

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
