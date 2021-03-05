# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
#
#
# We need to have __init__.py in tests dir due to a pytest issue.
#
# if you have:
#   tests/
#     aa/test_name.py
#     bb/test_name.py
#
# running `pytest tests` will give an error like "import file mismatch"
# because it can't distinguish between the file in `aa` and `bb` with
# the same file name. Add __init__.py file fixes it.
#
# However, `pytest tests/__init__.py` triggers running tests that's
# not related. So we just don't include any __init__.py in the test
# list files.
