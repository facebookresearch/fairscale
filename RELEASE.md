## Steps to do a release

### New Approach
- Go to the [fairscale release workflow](https://github.com/facebookresearch/fairscale/actions/workflows/release.yml) in Github actions.
- In the __Run Workflow__ dropdown, select the branch from which you wish to release. The default value is __main__ and should be used in almost all cases.
- In adherence to [Semantic Versioning]((https://semver.org/spec/v2.0.0.html)) enter one of the following three values for _Release Type_:
  - _patch_
  - _minor_
  - _major_
- Click __Run Workflow__.
- Verify [fairscale/version.py](https://github.com/facebookresearch/fairscale/blob/main/fairscale/version.py) has been updated.
- Verify a new [PyPI package](https://pypi.org/project/fairscale/) has been published.
- Verify a new [Github release](https://github.com/facebookresearch/fairscale/releases) has been created.

---
### Old Approach

- Update the CHANGELOG.md
- Update "what's new" in README.md
- If needed, update the PyTorch versions in README.md in the Testing section.
- Update `fairscale/__init__.py` and `docs/source/conf.py` for the new version number
- git commit the change with title like "[chore] 0.3.1 release"
- make a tag, like `git tag v0.3.1`
- git push --tags origin [your/branch]
- `python3 setup.py sdist` to build a new package (will be in dist/)
- `python3 -m twine upload --repository pypi dist/*` to upload to pypi
- visit [this page](https://github.com/facebookresearch/fairscale/tags) and create the newly
  tagged release.
