import pickle
import os
from typing import Any, BinaryIO, IO, Union

DEFAULT_PROTOCOL = 2

def save(obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
         pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True) -> None: ...
def load(f: Union[str, os.PathLike, BinaryIO, IO[bytes]], map_location) -> Any: ...
