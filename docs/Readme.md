# TorchMX Documentation

## Building the viewing TorchMX documentation

```bash

# Navigate to this directory as the script only works from this directory
cd torchmx/docs

# Build and serve the docs in localhost:8000
bash ./build_and_serve.sh

# This will automatically create a build directory.
```

In your browser navigate to `localhost:8000` to see the documentation.

## Extending the documentation

We use `mkdocs` as our documentation engine. There are 2 ways to create documentation:

1. If you want to auto-generate python code docs, we use `pydoc-markdown` as the tool.
Using it is very simple, go inside the [pydoc-markdown.yaml](./pydoc-markdown.yaml) and
add the corresponding glob pattern in the `Developer API`. Also add it to the `User API`
if it is intended for the user to use it. You can think of `User API` as a subset of `Developer API` containing only the necessary documentation to use `TorchMX` without the additional fluff. This will automatically create them inside the `./build/` directory which
is not tracked by `git`.
2. If you want to manually write them, add them as a markdown in the [sources](./sources/) directory and link them within the `pages` section in the same [pydoc-markdown.yaml](./pydoc-markdown.yaml).
3. If you want to add an example, add them as a markdown in the [sources/examples](./sources/examples/) and link them within the `pages` section in the same [pydoc-markdown.yaml](./pydoc-markdown.yaml).
4. Then build and serve as above
