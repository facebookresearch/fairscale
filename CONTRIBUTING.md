# Contributing to FairScale

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

Minor changes and improvements will be released on an ongoing basis. Larger
changes (e.g., changesets implementing a new paper) will be released on a
more periodic basis.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code passes static analysis (see below).
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Environment setup

```
~$ python3 -m venv venv
~$ source venv2/bin/activate
(venv2) ~$ cd git/fairscale/
(venv2) ~/git/fairscale $ pip3 install -r requirements-dev.txt
```

## Coding Style

* We follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
* In your editor, install the [editorconfig](https://editorconfig.org/) extension
  which should ensure that you are following the same standards as us.
* Please read the [editorconfig](.editorconfig) file to understand the exact coding style preferences.
* Please place Python code related to models in fairscale/nn. Place Python code related to optimizers
  in fairscale/optim. Place C++ extensions in fairscale/clib.
* Please put `__all__:List[str] = []` in new `__init__.py` files for consistent importing behavior
  and less development overhead in maintaining an importing list.
* Please setup pre-commit before opening up your PR.

### Pre-Commit (Recommended)

We use pre-commit to maintain the coding style. Pre-Commit checks are run via Github Actions on every
commit. To install all the relevant libraries and run the pre-commit tests locally, execute the following
commands:

```
pip install -r requirements-dev.txt
pre-commit install
```

After the above, your `git commit` command will automatically trigger pre-commit checks.

### Running static code analysis manually (Deprecated)

Note that, trailing spaces are not checked by the manual commands below, but they are checked by the
pre-commit hooks we use above.

```
black .
isort .
flake8
mypy --ignore-missing-imports --scripts-are-modules --pretty .
```

## Testing

FairScale code is tested on Python 3.9.7, CUDA 11.2 and the following three PyTorch versions:
- the latest stable version
- the latest LTS version
- a recent nightly release

See the [README](https://github.com/facebookresearch/fairscale/blob/main/README.md#testing) for the exact version numbers.

### Unit tests

```
pytest
# single test
python -m pytest tests/nn/data_parallel/test_oss_ddp.py::test_on_cpu
```

### Check test coverage

```
python -m pytest --cov-report term --cov=fairscale/nn/data_parallel \
   tests/nn/data_parallel/test_oss_ddp.py::test_on_cpu
```

### CircleCI status

From your PR page, you can expand on the CircleCI results. For GPU test, you should see
what CI has run, like:

```
...
----- generated xml file: /home/circleci/fairscale/test-results/junit.xml ------
================== 217 passed, 2 xfailed in 218.74s (0:03:38) ==================
CircleCI received exit code 0
```

The number of passed and failed should give you an idea on whether your local
test was the same or not.

## Commit Guidelines

We follow the same guidelines as AngularJS. Each commit message consists of a **header**,
a **body** and a **footer**.  The header has a special format that includes a **type**,
and a **subject**:

```
[<type>] <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on github as well as in various git tools.

### Type

Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **cleanup**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, dead code removal etc.)
* **refactor**: A code change that neither fixes a bug or adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests or fixing them
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation
generation
* **docs**: Documentation only changes

## License

By contributing to fairscale, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
