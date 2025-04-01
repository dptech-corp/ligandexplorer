from dp.launching.typing import BaseModel, Field
from dp.launching.typing import InputFilePath, OutputDirectory
#from dp.launching.typing import Int, Float, List, Enum, String
from dp.launching.cli import to_runner

from pathlib import Path
import os

class IOOptions(BaseModel):
    input_zip: InputFilePath = Field(..., ftypes=['zip','rar','tar','tar.gz','tar.bz2','tar.xz','tgz','tbz2','7z','7zip'],description="input file")


def runner(opts: IOOptions):
    """
    当app真正运行时的流程。
    opts里面保存了之前各个Model定义的参数
    """
    #activate_env = "mamba activate app"
    cmd = f'ligandexplorer -i {opts.input_zip.get_full_path()} -o output'
    zip_cmd = "zip -r output.zip output"
    # 运行命令行
    #os.system(activate_env)
    os.system(cmd)
    os.system(zip_cmd)

if __name__ == "__main__":
    import sys
    to_runner(IOOptions, runner)(sys.argv[1:])
