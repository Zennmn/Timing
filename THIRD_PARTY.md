# 第三方来源与许可

Timing 自有源码的 [MIT](LICENSE) 不替代第三方软件或模型权重的授权。源码仓库只提供适配代码和显式下载脚本，**不分发模型、上游运行库副本、歌曲或任务结果**。模型下载到被 Git 忽略的 `models/`；安装时保留上游许可文件。下面是来源记录，不将“可下载”解释为任意再分发 / 商用许可。

| 组件 | 固定来源 | 许可说明 |
| --- | --- | --- |
| TIFA 运行库 | [openvpi/TIFA](https://github.com/openvpi/TIFA/tree/d246be1cc2e202613213ca60a4e5d35e71a2fb6a)，`d246be1cc2e202613213ca60a4e5d35e71a2fb6a` | [源码 MIT](https://github.com/openvpi/TIFA/blob/d246be1cc2e202613213ca60a4e5d35e71a2fb6a/LICENSE)，2026 Team OpenVPI |
| TIFA-1.0-ST 模型与资源 | [v1.0.0 发布包](https://github.com/openvpi/TIFA/releases/tag/v1.0.0)，下载脚本校验 SHA-256 | 按发布包内模型 / 字典各自条款使用；不从源码 MIT 推断全部资产许可 |
| 中文拼音音素表 | 内置 `timing/resources/mandarin_pinyin.txt` 与 `LICENSE-TIFA.txt`，复制自 [openvpi/TIFA](https://github.com/openvpi/TIFA/tree/d246be1cc2e202613213ca60a4e5d35e71a2fb6a)，`d246be1cc2e202613213ca60a4e5d35e71a2fb6a` | MIT（随附 `LICENSE-TIFA.txt`）；用于普通话读音与音素对应 |
| 日语分词与读音词典 | [NINJAL UniDic CWJ 202512 / UniDic ver.2025.12 常规分析包](https://clrd.ninjal.ac.jp/unidic_archive/2512/unidic-cwj-202512.zip)，下载包约 696 MB，保存到 `models/unidic-cwj-202512/` | NINJAL 归档采用 BSD / GPL / LGPL 多许可；下载器保留归档内许可文件。仓库不包含词典正文，也不能把源码 MIT 当作词典许可。安装不需要 Full 训练材料 |
| TIFA 日语音素映射 | TIFA-1.0-ST 发布包内 `dictionaries/japanese_dict_full.txt`，随 `models/tifa/` 安装 | 按 TIFA 发布包条款使用；它是假名到音素的映射表，不是 UniDic 分词词典，也不替代上述常规分析包 |
| MMS-300M | [固定模型卡](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner/blob/49402e9577b1158620820667c218cd494cc44486/README.md)，`49402e9577b1158620820667c218cd494cc44486` | 模型卡标注 **CC-BY-NC-4.0**，含非商业限制 |
| BS-Roformer 运行源码 | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/6fdad13686bb7e84d029ec66f9dbe76e65c93ebf)，`6fdad13686bb7e84d029ec66f9dbe76e65c93ebf` | 下载源码附带 MIT，2024 Roman Solovyev |
| frazer/becruily 主唱权重 | [becruily/bs-roformer-karaoke](https://huggingface.co/becruily/bs-roformer-karaoke/tree/f7849ae934209184dc288d1018cd8a76a7fc8b3c)，`f7849ae934209184dc288d1018cd8a76a7fc8b3c` | 来源没有模型卡，权重许可未明确确认；不包含在 Timing MIT 中 |
| SOFA 运行库（历史缓存） | [qiuqiao/SOFA](https://github.com/qiuqiao/SOFA/tree/584d6b9a57927843f85decebf4cbf03b9598125f)，`584d6b9a57927843f85decebf4cbf03b9598125f` | 源码 MIT，2023 suco；0.17 起不再下载或要求，仅保留旧安装的许可记录 |
| SOFA 日中英权重 / 字典（历史缓存） | [日语 akm_ja_v001](https://github.com/ariikamusic/SOFA_Models/releases/tag/akm_ja_v001)、[英语 tgm_en_v100](https://github.com/spicytigermeat/SOFA-Models/releases/tag/v1.0.0_en)、[官方普通话歌声包](https://github.com/qiuqiao/SOFA/releases/tag/v1.0.1)与[固定普通话字典](https://github.com/qiuqiao/SOFA/blob/584d6b9a57927843f85decebf4cbf03b9598125f/dictionary/opencpop-extension.txt) | 早期安装曾校验各资源 SHA-256；当前不再安装或读取。既有 `models/sofa/` 缓存保留，权重 / 词典仍按各自条款使用，不推断与 SOFA 源码相同 |
| MelBand 推理库 | [openmirlab/melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer/tree/55e4217935da91aee0f2f7ac44cbad512a3485ac)，`55e4217935da91aee0f2f7ac44cbad512a3485ac` | 通过依赖安装，使用上游自带许可；不是 Timing 自有代码 |
| hanziconv | [berniey/hanziconv](https://github.com/berniey/hanziconv)，PyPI `hanziconv` 0.3.2（随 `uv.lock` 固定） | Apache-2.0；仅用于读音前的繁→简只读转换，原文与显示保持不变 |
| Kim 人声权重 | [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) | 权重哈希 `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`；授权以权重作者条款为准，未确认可商用 |
| Feather 图标 | [v4.29.2](https://github.com/feathericons/feather/tree/v4.29.2) | MIT；完整声明见 [web/THIRD_PARTY_NOTICES.md](web/THIRD_PARTY_NOTICES.md) |

其余 Python 依赖固定在 `uv.lock`，保留各分发包中的许可；根环境使用 Fugashi 读取 CWJ 202512 常规分析包，`unidic==1.1.0` 保留为 TIFA 的兼容依赖，词典正文由上述显式下载提供。旧 UniDic CWJ 202302 目录即使仍在缓存中，也不属于当前运行依赖。TIFA 的隔离依赖版本列于 `scripts/tifa-requirements.txt`。本地共享读音 / 音素辅助使用 TIFA 资源；SOFA 缓存不参与当前运行。

旧 Qwen 对照与历史 HubertFA / Anvuew 试验不在安装配置内，相关本机资源不随仓库分发。远端 LLM 由使用者配置，需遵守该服务条款。

若要分发包含权重的安装包或提供商业服务，需另行完成相关权重 / 字典授权核对；当前整理仅覆盖源码仓库范围。
