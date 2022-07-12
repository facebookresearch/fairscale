import os
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Callable, IO, Union

DEFAULT_PROTOCOL: int = 2

def save(obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
         pickle_module: Any=pickle, pickle_protocol: int=DEFAULT_PROTOCOL, _use_new_zipfile_serialization: bool=True) -> None: ...

def load(f: Union[str, BinaryIO, Path], map_location=None) -> Any: ...
