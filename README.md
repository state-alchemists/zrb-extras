# Zrb extras

zrb-extras is a [pypi](https://pypi.org) package.

You can install zrb-extras by invoking the following command:

```bash
pip install zrb-extras
```

Once zrb-extras is installed, you can then run it by invoking the following command:

```bash
zrb-extras
```

You can also import `zrb-extras` into your Python program:

```python
from zrb_extras import hello

print(hello())
```


# For maintainers

## Publish to pypi

To publish zrb-extras, you need to have a `Pypi` account:

- Log in or register to [https://pypi.org/](https://pypi.org/)
- Create an API token

You can also create a `TestPypi` account:

- Log in or register to [https://test.pypi.org/](https://test.pypi.org/)
- Create an API token

Once you have your API token, you need to configure poetry:

```
poetry config pypi-token.pypi <your-api-token>
```

To publish zrb-extras, you can do the following command:

```bash
poetry publish --build
```

## Updating version

You can update zrb-extras version by modifying the following section in `pyproject.toml`:

```toml
[project]
version = "0.0.2"
```

## Adding dependencies

To add zrb-extras dependencies, you can edit the following section in `pyproject.toml`:

```toml
[project]
dependencies = [
    "Jinja2==3.1.2",
    "jsons==1.6.3"
]
```

## Adding script

To make zrb-extras executable, you can edit the following section in `pyproject.toml`:

```toml
[project-scripts]
zrb-extras-hello = "zrb_extras.__main__:hello"
```

Now, whenever you run `zrb-extras-hello`, the `main` function on your `__main__.py` will be executed.
