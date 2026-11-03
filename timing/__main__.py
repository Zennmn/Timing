import argparse
import shutil
import sys
from pathlib import Path
from .config import initialize
from .request_plan import ALIGNERS, AUDIO_MODES, LANGUAGES, plan_request


def main():
    parser = argparse.ArgumentParser(description="Timing 本地中日英歌词打轴 → TTML")
    parser.add_argument("audio", nargs="?", type=Path)
    parser.add_argument("--lyrics", type=Path, help="必需，UTF-8 TXT/LRC/JSON/逐行 TTML 正确歌词")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vocals", type=Path, help="可选，同起点同时长的人声轨")
    parser.add_argument("--language", choices=LANGUAGES, default="mixed", help='本地路线：zh 普通话；zh-en 中英；mixed 日英。LLM 路线忽略此设置，自行判断语言')
    parser.add_argument("--word-level", action="store_true", help="英语按整词打轴，不拆音节")
    parser.add_argument("--english-splitter", choices=["syllabify", "pyphen", "llm"], default="syllabify", help="英语音节划分；llm 使用 OpenAI-compatible Chat Completions 接口")
    parser.add_argument('--llm-base-url',help='含 /v1 的兼容接口地址；也可用 TIMING_LLM_BASE_URL')
    parser.add_argument('--llm-model',help='接口的模型名；也可用 TIMING_LLM_MODEL')
    parser.add_argument('--llm-result',type=Path,help='网页 LLM 返回的 UTF-8 JSON，配合 --aligner llm_import，无需接口配置')
    parser.add_argument('--export-llm-prompt',type=Path,help='把完整提示词写入 TXT 后退出，只需 --lyrics，不需要音频或 Key')
    parser.add_argument('--llm-prompt-profile',choices=['fine','original'],default='fine',help='LLM 日语拆分：fine 细拆（默认）；original 适度合并（旧版）')
    parser.add_argument('--llm-api-key-header',help='API Key 请求头名称，默认 Authorization；也可用 TIMING_LLM_API_KEY_HEADER')
    parser.add_argument('--llm-api-key-format',choices=['auto','bearer','raw'],help='自动：Authorization 加 Bearer，其余原样；也可显式选 bearer/raw')
    parser.add_argument('--llm-json-mode',action='store_true',default=None,help='接口支持时启用 JSON object 模式')
    parser.add_argument('--llm-thinking-mode',choices=['default','on','off'],default=None,help='思考模式；默认跟随平台')
    parser.add_argument('--llm-thinking-format',choices=['reasoning_effort','enable_thinking','thinking'],default=None,help='兼容接口的思考参数格式')
    parser.add_argument('--llm-reasoning-effort',default=None,help='思考强度，如 low/medium/high/xhigh 或平台自定义值')
    parser.add_argument('--audio-mode', choices=AUDIO_MODES, default='lead', help='lead 只保留主唱；vocals 去伴奏并保留和声；source 直接使用原曲或导入轨')
    parser.add_argument("--aligner", choices=ALIGNERS, default="auto", help="自动使用 TIFA；MMS 支持纯文本和逐句歌词。llm_text 只发送歌词文字；llm_direct 发送完整音轨；LLM 需逐句范围，使用 TIFA 计算时间")
    parser.add_argument('--trust-line-timestamps', action='store_true', help='确信逐句 LRC／TTML 时间准确时，跳过 TIFA 前的 MMS 粗定位')
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    initialize()
    if args.serve:
        import uvicorn
        uvicorn.run("timing.server:app", host="127.0.0.1", port=args.port, log_level="info")
        return
    if args.export_llm_prompt:
        try:
            from .llm_import import export_prompt
            from .request_plan import MAX_LYRICS_BYTES
            if not args.lyrics or not args.lyrics.is_file() or args.lyrics.stat().st_size > MAX_LYRICS_BYTES:
                parser.error('请用 --lyrics 指定最大 2 MB 的 UTF-8 LRC / JSON / TTML。')
            inputs = {p.resolve() for p in (args.audio, args.lyrics, args.vocals, args.llm_result) if p}
            if args.export_llm_prompt.resolve() in inputs:
                parser.error('提示词输出不能覆盖输入文件。')
            prompt = export_prompt(args.lyrics.read_text(encoding='utf-8-sig'),args.audio.name if args.audio else '',
                                   prompt_profile=args.llm_prompt_profile)
            args.export_llm_prompt.write_text(prompt['prompt'],encoding='utf-8')
            print(f'提示词: {args.export_llm_prompt.resolve()}')
            return
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if not args.audio or not args.audio.is_file():
        parser.error("请指定音频文件，或使用 --serve 打开本地界面。")
    if not args.lyrics or not args.lyrics.is_file():
        parser.error('请用 --lyrics 指定 TXT、LRC、JSON 或逐行 TTML 歌词文件。')
    if args.vocals and not args.vocals.is_file():
        parser.error('找不到导入的人声轨，请检查 --vocals 路径。')
    from .pipeline import run
    output = args.output or args.audio.with_suffix(".ttml")
    if output.resolve() in {path.resolve() for path in (args.audio, args.lyrics, args.vocals, args.llm_result) if path}:
        parser.error("输出路径不能覆盖输入文件。")
    folder = output.parent / (output.stem + ".timing")
    try:
        from .request_plan import MAX_LYRICS_BYTES
        if args.lyrics.stat().st_size > MAX_LYRICS_BYTES:
            parser.error('歌词最大 2 MB。')
        lyrics = args.lyrics.read_text(encoding='utf-8-sig')
        llm_result = None
        if args.aligner == 'llm_import':
            from .llm_import import MAX_RESULT_BYTES, import_response
            if not args.llm_result or not args.llm_result.is_file():
                parser.error('网页 LLM 模式请用 --llm-result 指定返回的 JSON 文件。')
            if args.llm_result.stat().st_size > MAX_RESULT_BYTES:
                parser.error('网页 LLM 结果最大 16 MiB。')
            llm_result = args.llm_result.read_text(encoding='utf-8-sig')
            import_response(lyrics, llm_result)
        plan = plan_request(lyrics, language=args.language, aligner=args.aligner, audio_mode=args.audio_mode,
                            syllables=not args.word_level, english_splitter=args.english_splitter,
                            trust_line_timestamps=args.trust_line_timestamps)
        llm_config = None
        if plan.needs_llm:
            import os
            from .llm_splitter import LlmConfig
            llm_config = LlmConfig(args.llm_base_url or os.environ.get('TIMING_LLM_BASE_URL',''),
                                   args.llm_model or os.environ.get('TIMING_LLM_MODEL',''),
                                   os.environ.get('TIMING_LLM_API_KEY',''),
                                   args.llm_json_mode if args.llm_json_mode is not None else os.environ.get('TIMING_LLM_JSON_MODE','0')=='1',
                                   args.llm_thinking_mode if args.llm_thinking_mode is not None else os.environ.get('TIMING_LLM_THINKING_MODE','default'),
                                   args.llm_thinking_format if args.llm_thinking_format is not None else os.environ.get('TIMING_LLM_THINKING_FORMAT','reasoning_effort'),
                                   args.llm_reasoning_effort if args.llm_reasoning_effort is not None else os.environ.get('TIMING_LLM_REASONING_EFFORT','high'),
                                   args.llm_api_key_header if args.llm_api_key_header is not None else os.environ.get('TIMING_LLM_API_KEY_HEADER','Authorization'),
                                   args.llm_api_key_format if args.llm_api_key_format is not None else os.environ.get('TIMING_LLM_API_KEY_FORMAT','auto'))
        extra = {'llm_result':llm_result} if args.aligner == 'llm_import' else {}
        if args.aligner == 'llm_direct':
            extra['llm_prompt_profile'] = args.llm_prompt_profile
        extra['trust_line_timestamps'] = plan.trust_line_timestamps
        run(args.audio, lyrics, folder, args.language, plan.syllables, args.vocals,
                     audio_mode=plan.audio_mode, aligner=args.aligner, english_splitter=plan.english_splitter,
                     llm_config=llm_config, **extra)
        shutil.copyfile(folder / "lyrics.ttml", output)
        print(f"TTML: {output.resolve()}")
        print(f"对齐记录: {(folder / 'alignment.json').resolve()}")
    except Exception as exc:
        print("失败：" + str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
