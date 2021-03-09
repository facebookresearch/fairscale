## Steps to do a release

- Update the CHANGELOG.md
- Update `fairscale/__init__.py` and `docs/source/conf.py` for the new version number
- git commit the change with title like "[chore] 0.3.1 release"
- make a tag, like `git tag v0.3.1`
- git push --tags origin [your/branch]
- `python3 setup.py sdist` to build a new package (will be in dist/)
- `python3 -m twine upload --repository pypi dist/*` to upload to pypi
