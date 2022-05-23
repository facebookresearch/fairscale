import wgit.cli as cli


def test_cli_init(capsys):
    cli.main(["--init"])
    out, err = capsys.readouterr()
    assert out == "Hello World, Wgit has been initialized!\n"
    assert err == ""
