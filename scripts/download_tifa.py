"""Explicit setup only. Pinned native TIFA and isolated non-Torch dependencies."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_support import atomic_write_json, error_tail, fetch, file_matches, sha256_file
from download_unidic import install as install_dictionary

ROOT=Path(__file__).resolve().parent.parent
FOLDER=ROOT/'models/tifa'
REVISION='d246be1cc2e202613213ca60a4e5d35e71a2fb6a'
MODEL_SHA='6da6832cd2cb981aae1ccc2d9330f8f7c085ce7977d1eb13bff63531d038a88b'


def unpack_strip(archive,target):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():continue
            parts=Path(item.filename).parts[1:]
            if not parts:continue
            dest=target.joinpath(*parts)
            if not dest.resolve().is_relative_to(target.resolve()):raise ValueError('unsafe archive')
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_name(dest.name+'.partial')
            temporary.write_bytes(z.read(item))
            temporary.replace(dest)


def _zip_valid(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('压缩包内部校验失败。')


def _manifest_ok():
    manifest_path=FOLDER/'manifest.json'
    if not manifest_path.is_file():return False
    try:manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return False
    if manifest.get('runtime_revision')!=REVISION or manifest.get('release_sha256')!=MODEL_SHA:
        return False
    assets=manifest.get('assets')
    if not isinstance(assets,dict) or not assets:return False
    for name,asset in assets.items():
        if not isinstance(asset,dict):return False
        expected=asset.get('sha256')
        if not isinstance(expected,str) or len(expected)!=64:return False
        if not file_matches(FOLDER/name,sha256=expected,size=asset.get('bytes')):
            return False
    return all((FOLDER/name).is_file() for name in (
        'model.pt','config.yaml','vocabulary.json','runtime/inference/api.py',
        'deps/g2pflow/__init__.py','dictionaries/japanese_dict_full.txt',
        'dictionaries/ds_cmudict-07b.txt','dictionaries/ds-zh-pinyin-lite.txt'))


def _package_ok():
    code=("import sys; from pathlib import Path; "
          "root=Path(sys.argv[1]); sys.path[:0]=[str(root/'deps'),str(root/'runtime')]; "
          "import g2pflow, onnxruntime; from inference.api import load_inference_model, load_g2p_config")
    env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    result=subprocess.run([sys.executable,'-X','utf8','-c',code,str(FOLDER)],capture_output=True,
                          text=True,encoding='utf-8',errors='replace',timeout=120,env=env)
    return result.returncode==0,result.stderr[-800:]


def main():
    if _manifest_ok():
        print('TIFA 本地文件校验通过，继续检查运行依赖。',flush=True)
    else:
        cache=ROOT/'models/.downloads';cache.mkdir(parents=True,exist_ok=True)
        cached=ROOT/'.runtime/research-assets/TIFA-1.0-ST.zip'
        archive=cached if cached.exists() else cache/'TIFA-1.0-ST.zip'
        fetch('https://github.com/openvpi/TIFA/releases/download/v1.0.0/TIFA-1.0-ST.zip',archive,
              expected_sha256=MODEL_SHA,label='TIFA 模型包')
        unpack_strip(archive,FOLDER)
        source=fetch(f'https://codeload.github.com/openvpi/TIFA/zip/{REVISION}',
                     cache/f'TIFA-{REVISION}.zip',label='TIFA 运行库源码',validate=_zip_valid)
        unpack_strip(source,FOLDER/'runtime')
        assets={str(p.relative_to(FOLDER)).replace('\\','/'):
                {'sha256':sha256_file(p),'bytes':p.stat().st_size}
                for p in FOLDER.rglob('*') if p.is_file() and not p.is_relative_to(FOLDER/'deps')
                and p.name!='manifest.json'}
        atomic_write_json(FOLDER/'manifest.json',{'source':'https://github.com/openvpi/TIFA',
            'runtime_revision':REVISION,'model':'TIFA-1.0-ST','release_sha256':MODEL_SHA,'assets':assets})

    compatible,detail=_package_ok()
    if not compatible:
        print('TIFA 运行依赖不兼容或缺少包；正在按固定清单修复。',flush=True)
        subprocess.run(['uv','pip','install','--python',sys.executable,'--target',str(FOLDER/'deps'),
                        '--no-deps','--upgrade','-r',str(ROOT/'scripts/tifa-requirements.txt')],check=True)
        compatible,detail=_package_ok()
    if not compatible:
        raise RuntimeError('TIFA 运行依赖导入失败：'+error_tail(detail))
    if not _manifest_ok():
        raise RuntimeError('TIFA 文件完整性校验失败，请重新运行 scripts/download_tifa.py。')
    install_dictionary()
    print('TIFA 本地文件与运行依赖校验通过。',flush=True)


if __name__=='__main__':main()
