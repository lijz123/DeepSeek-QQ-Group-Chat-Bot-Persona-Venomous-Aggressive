# Chat Bot

基于 Python、OneBot v11 和 DeepSeek 兼容接口的 QQ 群聊天机器人。
提供 @ / 关键词触发、冷却限流、群聊上下文、人设与离线测试。

## 安装和配置

需要 Python 3.10+，以及已配置好的 OneBot v11 服务（例如 NapCat）。

```powershell
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
$env:DEEPSEEK_API_KEY = "在这里填入你自己的密钥"#想换gpt的也行，不过deepseek更可爱些，反差感更强
python bot.py --config config.json
```

Windows 双击 `start.bat` 会在本地缺少配置时复制示例配置；仍需先向启动进程提供密钥环境变量。
示例中的 `llm.api_key` 必须保留为 `env:DEEPSEEK_API_KEY`，不要把真实密钥写入公开文件。
OneBot 地址默认是本机 `ws://127.0.0.1:3001`；如设置了 OneBot Token，写到本地 `config.json`，不要提交。
本机服务建议仅监听本机地址，并按实际部署配置认证。

`groups.whitelist` 是空列表，表示不限制群；使用前请在本地配置中填入允许的群号。
`users.blacklist`、机器人名称和触发关键词可在本地配置中自定义。
只在已获群成员知情同意的环境中启用，请避免骚扰和泄露群聊信息。

## 安装 NapCat，登录机器人 QQ

按照 NapCat 官方 Windows 安装说明安装、启动，并登录打算作为机器人的 QQ 账号。 #https://napneko.github.io/guide/boot/Shell
把这个 QQ 加入测试群，保持 NapCat 运行。
然后在 NapCat 的网络配置里，新建：

| 设置    | 内容                               |
| ----- | -------------------------------- |
| 类型    | **WebSocket 服务端／正向 WS**，不要选反向 WS |
| 监听地址  | `127.0.0.1`                      |
| 端口    | `3001`                           |
| 消息格式  | `array`                          |
| Token | 自己设置一个随机密钥                       |
| 启用    | 开启                               |

具体入口见 NapCat 官方网络配置说明。这里只供本机连接，不要把接口开放到公网。#https://napneko.github.io/config/basic





## 离线自检(可以忽略，后期调参可以使用)

```powershell
python selftest.py
```

测试在本机启动模拟 OneBot 和模型服务，不连接真实 QQ、不调用真实模型、不需要真实密钥。
覆盖消息解析、@、去重、静默、关键词触发、输出清理，以及昵称与正文分离。

```powershell
python preview.py "你好" "测试用户"
```

预览会调用你配置的真实模型，可能消耗 API 额度。

## 文件说明

- `bot.py`：OneBot 客户端、消息处理、触发与发送。
- `persona.py`：人设规则和输出清洗。
- `preview.py`：模型回复预览。
- `selftest.py`：模拟服务自检。
- `config.example.json`：不含真实账号信息的示例配置。
- `start.bat`：Windows 启动入口。
- `requirements.txt`：依赖。
- `.gitignore`：忽略本地配置、凭据、日志、缓存与备份。

## 数据与能力边界

消息上下文使用 JSONL，把 `sender` 身份信息和 `content` 实际发言分开。
昵称分离适用于所有成员，不按某个昵称写特例；它降低误读，但不保证模型永不出错。
运行时发送者昵称、账号标识和消息内容可能随上下文传给模型服务，请先了解并告知群成员。
图片等媒体目前只识别为标记，不进行实际图像理解。
群聊上下文保存在内存中，重启或超时后清除；本地日志可能包含群号和聊天输出，不要公开。

## **==首次使用时执行请阅读==**

# QQ 登录与机器人启动指南

适用场景：从 GitHub 下载本项目，在 Windows 上使用，NapCat 与本项目运行在同一台电脑。

## 一、先弄清楚：由谁登录 QQ？

**启动顺序：启动 NapCat → 扫码登录机器人 QQ → 启动 Python 机器人 → 群里测试。**

| 程序             | 作用                         |
| -------------- | -------------------------- |
| NapCat         | 负责 QQ 登录、接收 QQ 消息、发送 QQ 消息 |
| 本项目的 Python 程序 | 负责处理消息、调用模型、生成回复           |

**不是运行 `bot.py` 来登录 QQ。`bot.py` 没有 QQ 扫码登录功能。**

如果配置 NapCat 时已经扫码登录成功，就不需要再登录一次。

## 二、通过 NapCat 登录机器人 QQ

### 1. 启动 NapCat

进入安装 NapCat 的文件夹，按照所使用的安装包运行对应入口：

| 安装方式          | Windows 启动入口                                              |
| ------------- | --------------------------------------------------------- |
| Windows 一键安装包 | 安装完成后，进入生成的目录，双击 `napcat.bat`                             |
| Shell 手动安装包   | 双击 `launcher.bat`；Windows 10 按官方说明使用 `launcher-win10.bat` |

不同版本的入口可能变化，以 [NapCat 官方 Windows 安装说明](https://napneko.github.io/guide/boot/Shell)为准。上述文件属于 NapCat，不在本项目文件夹里。

### 2. 手机扫码登录

1. 按照 NapCat 控制台提示，打开本机 WebUI 管理页面。
2. 使用控制台给出的 WebUI 登录信息进入管理页面，不要公开包含 Token 的地址。
3. 进入 **QQ 登录 → 扫码登录**。
4. 用手机 QQ 扫描页面上的二维码，并在手机上确认登录。
5. 等待 NapCat 显示登录成功，让这个 QQ 加入需要使用机器人的群。

**扫码确认登录的 QQ，就是之后发送机器人回复的账号。** 二维码过期时，在页面刷新二维码后重新扫描。如果 QQ 要求额外安全验证，请按官方提示完成。

登录后保持 NapCat 运行，不要关闭它的进程。

### 3. 确认正向 WebSocket 配置

在 NapCat 的网络配置中，确认已经创建并启用了 **WebSocket 服务端（正向 WS）**，不是反向 WS：

| 设置    | 本机部署示例                  |
| ----- | ----------------------- |
| 监听地址  | `127.0.0.1`             |
| 端口    | `3001`                  |
| 消息格式  | `array`                 |
| Token | 自己设置一个随机密钥，稍后填入本项目的本地配置 |
| 启用    | 开启并保存                   |

**WS Token 与 WebUI 网页登录密码不是同一个用途，不要混填。** 不要把本机 WS 接口暴露到公网。

详情见 [NapCat 官方网络配置说明](https://napneko.github.io/config/basic)。如果已经正确配置，直接进入下一步。

## 三、首次准备 Python 机器人

### 1. 安装 Python

本项目需要 Python 3.10 或以上版本。可以从 [Python 官网](https://www.python.org/downloads/windows/)安装，确保终端能使用 `python` 命令。

### 2. 在项目文件夹打开 PowerShell

解压 GitHub 下载的代码，进入能看到 `bot.py` 和 `requirements.txt` 的文件夹。在文件夹地址栏输入 `powershell`，按回车。

**下面的命令都在这个代码文件夹中执行，不是在 NapCat 文件夹执行。**

检查 Python，并安装项目依赖：

```powershell
python --version
python -m pip install -r requirements.txt
```

### 3. 生成本地配置

==仅在没有 `config.json` 时执行：==

```powershell
if (!(Test-Path -LiteralPath config.json)) {
    Copy-Item config.example.json config.json
}
notepad config.json
```

如果已经有自己的配置，不要重新复制覆盖。

主要核对下面这些字段，其他设置可以暂时保留：

| 字段                    | 应该怎么填                                           |
| --------------------- | ----------------------------------------------- |
| `onebot.ws_url`       | 与 NapCat 正向 WS 地址一致；本机示例为 `ws://127.0.0.1:3001` |
| `onebot.access_token` | 与 NapCat 正向 WS 中设置的 Token 完全一致                  |
| `groups.whitelist`    | 填入允许机器人工作的群号，建议首次只填一个测试群                        |
| `llm.base_url`        | 所使用模型服务的兼容接口地址，不要填聊天网页地址                        |
| `llm.model`           | 服务商实际支持的模型名称                                    |
| `llm.api_key`         | 保留为 `env:DEEPSEEK_API_KEY`，真实密钥通过后面的环境变量提供      |
| `persona.name`        | 自己喜欢的机器人名称                                      |

**注意：群白名单 `[]` 表示不限制群，不是禁止所有群。** 编辑后保存；JSON 不支持随意添加注释或多余的逗号。

真实密钥、群号和本地 Token 只放在自己的运行环境中，不要上传 `config.json`，也不要公开相关截图。

### 4. 运行离线自检

```powershell
python selftest.py
```

正常情况下，最后应显示 `18/18 通过`。该测试使用本机模拟服务，不连接真实 QQ，也不调用真实模型。

**离线测试通过不代表真实模型密钥、账户额度或接口兼容性已经验证成功。**

## 四、登录 QQ 后，如何启动机器人？

确保 NapCat 已登录、正向 WS 已启用，然后在项目文件夹的 PowerShell 中依次执行：

```powershell
$env:DEEPSEEK_API_KEY = "替换为自己的真实API密钥"
python bot.py --config config.json
```

密钥由使用者自行向模型服务商申请，并确认有可用额度。不要把包含真实密钥的命令或截图发到公开仓库。

也可以在设置密钥后，在**同一个 PowerShell 窗口**运行：

```powershell
.\start.bat
```

**上述两种启动方式二选一，不要同时运行两份机器人。**

`$env:DEEPSEEK_API_KEY` 只对当前 PowerShell 及其子进程生效。不要在这个窗口设置完临时变量后，又从桌面另外双击启动。重新打开 PowerShell 后，需要重新设置密钥。

本项目的 `start.bat` 只负责启动 Python 机器人，**不会替你启动 NapCat，也不会弹出 QQ 登录二维码**。

## 五、怎样确认机器人真的能工作？

1. 机器人控制台出现“连接 OneBot”“已登录”等日志，说明连接到了 QQ 接入程序；这还不等于模型调用成功。
2. 用另一个 QQ 账号进入白名单里的测试群。
3. 在群里真正 @ 机器人账号，再发送“你好”。不要仅输入名称代替 QQ 的 @ 操作。
4. 收到机器人回复，才说明 QQ 登录、消息接收、模型调用和消息发送整条链路已经跑通。

运行期间，**NapCat 与 Python 机器人两个程序都需要保持运行**。只关闭网页管理页面，不等于停止 NapCat；关闭 NapCat 进程则会断开机器人连接。

默认存在冷却和限流，连续测试时请间隔几秒，不要密集刷消息。

## 六、以后每次使用的最短流程

1. 启动 NapCat，确认机器人 QQ 在线；如果提示重新扫码，就在 NapCat 页面重新扫码确认。
2. 在项目文件夹打开 PowerShell。
3. 设置自己的 `DEEPSEEK_API_KEY`。
4. 执行 `python bot.py --config config.json`。

依赖安装和本地配置通常只需要首次完成，修改配置后应重启 Python 机器人使其生效。

停止 Python 机器人：在它的窗口按 `Ctrl+C`。如果也不需要 QQ 接入服务，再通过 NapCat 自己的方式停止它。

## 七、常见问题

| 问题                  | 优先检查                                                         |
| ------------------- | ------------------------------------------------------------ |
| 找不到 `python`        | 是否安装了 Python，以及当前终端能否找到它；安装后重开终端                             |
| 提示缺少 `aiohttp`      | 在同一个 Python 环境执行 `python -m pip install -r requirements.txt` |
| 不断显示连接异常或重连         | NapCat 是否仍在运行，正向 WS 是否启用，地址和端口是否一致                           |
| OneBot 连接鉴权失败       | `onebot.access_token` 是否与 WS Token 一致；不要填成 WebUI 登录密码        |
| 提示缺少 API Key        | 是否在启动机器人的同一个 PowerShell 中设置了环境变量                             |
| 模型返回鉴权失败、余额不足、模型不存在 | 按模型服务商的错误提示检查密钥、可用额度、接口地址和模型名称                               |
| QQ 在线但群里不回复         | 检查群白名单、是否真正 @ 了机器人、冷却时间，以及机器人窗口里的错误日志                        |

## 八、当前发布副本的接口兼容性提醒

这份指南说明启动与登录顺序，不代表已经用真实账号或真实模型完成联调。

请以所用服务商的当前文档为准。参考：[DeepSeek 官方 API 文档](https://api-docs.deepseek.com/)、[更新记录](https://api-docs.deepseek.com/updates/)、[思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)。

仅在群成员知情同意的环境使用。模型服务可能接收上下文中的昵称、账号标识和消息内容；不要公开登录二维码、密钥、本地配置或运行日志。
