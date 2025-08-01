(contributions)=

# Contribution Guide

This page contains pointers and links to help you contribute to this project.

(forking_v_main)=
## Forking vs. Working in the Main Repository

If you are a member of the https://github.com/radionets-project/ GitHub organization,
the maintainers can provide you access to the main repository at https://github.com/radionets-project/pyvisgen.
Working on the main repository has the advantage, that there is no need to synchronize between a
personal fork and the main repository, and collaboration is easier on the same branch with other
developers.

If you are an external contributor, you will have to create a fork of `pyvisgen`.


(cloning_repo)=
## Cloning the Repository
The following examples assume you have setup an ssh key with your GitHub account. [See the docs][github-ssh]
if you haven't already.

::::{tab-set}
:sync-group: category

:::{tab-item} Working in the main repository
:sync: main

Clone the repository:
```shell-session
$ git clone git@github.com:radionets-project/pyvisgen.git
```
And enter the directory of the repository you just cloned:
```shell-session
$ cd pyvisgen
```
:::

:::{tab-item} Working in a fork
:sync: fork

In order for you to contribute to `pyvisgen` without write access in the main repository,
you will need to [fork the repository][github-fork] first. After you have created the fork, clone
it:
```shell-session
$ git clone https://github.com/[YOUR-GITHUB-USERNAME]/pyvisgen.git
```
Enter the directory of the repository you just cloned:
```shell-session
$ cd pyvisgen
```
Then set the main repository as a second remote called `upstream`.
This allows your fork to be synchronized with the main repository.
```shell-session
$ git remote add upstream https://github.com/radionets-project/pyvisgen.git
```
```{seealso}
The GitHub docs provide useful information on
- [Forking a repository][github-fork]
- [Syncing a fork][github-sync]
```
:::
::::


(pre_commit)=
## Further Setting Up the Development Environment

 We are using [`pre-commit`][pre-commit] with [Ruff][ruff] as linter and formatter for automatic code adherence
to the {ref}`coding-style`. Install the `pre-commit` hooks:
```shell-session
$ pre-commit install
```
The pre-commit hooks will then run every time you commit something. If any of the tools
reports a problem, the commit will be aborted and you will have to fix the issues first.
Usually, a failing `pre-commit` hook indicates code not complying with the style guide.
Once all problems are fixed, you can try committing again, and the changes will be accepted.

To run `pre-commit` manually, call:
```shell-session
$ pre-commit run
```
Or, to run it on all files:
```shell-session
$ pre-commit run --all-files
```
The [Ruff][ruff] hook uses the configuration in [`pyproject.toml`][pyvisgen-pyproject] for linting and formatting.


(testing)=
## Testing

We're using [`pytest`][pytest] to test the functionality of `pyvisgen`. To test all
components, run:
```shell-session
$ pytest -vv
```
The `-vv` option shows verbose output and can be omitted. [`pytest`][pytest] disables
any output sent `stdout` and `stderr` by default. To enable the output,
run [`pytest`][pytest] with the `-s` option (or `-vvs` for verbose).

:::{attention}
Make sure that all tests run successfully, before you commit changes.
:::


(updating_repo)=
## Updating the Repository

To update the codebase to the latest development version (i.e. merging remote changes into your
local repository):

::::{tab-set}
:sync-group: category

:::{tab-item} Working in the main repository
:sync: main

```shell-session
$ git pull
```

:::

:::{tab-item} Working in a fork
:sync: fork

First, fetch the upstream of the main repository:
```shell-session
$ git fetch upstream
```
Then merge the `main` branch of the upstream into your local repository:
```shell-session
$ git merge upstream/main --ff-only
```
And push everything to your fork's remote:
```shell-session
$ git push
```

````{note}
Alternatively, you can also press the "{octicon}`sync` Sync fork" button on GitHub
and then use
```shell-session
$ git pull
```
````

:::
::::

(contributing_to_codebase)=
## Contributing to the Codebase

Please create a new feature branch whenever you want to contribute to the codebase.
Best practice is to create one branch per new feature, so that you do not mix code from
each. This also makes the reviewing process much easier.

:::{admonition} If you're working on a fork...
:class: warning
...you should **never** add commits to the `main` branch of your fork. This will create
issues due to diverging histories of the main repository and your fork, whenever
the `main` branch is updated from the upstream.
:::


### 1. Creating a Feature Branch

To create a new feature branch
::::{tab-set}
:sync-group: category

:::{tab-item} Working in the main repository
:sync: main

Fetch the latest changes from the main repository:
```shell-session
$ git fetch
```
And then create a new branch from `main`:
```shell-session
$ git switch -c <new feature branch name> origin/main
```

:::

:::{tab-item} Working in a fork
:sync: fork

Fetch the latest changes from the upstream of the main repository:
```shell-session
$ git fetch upstream
```
And then create a new branch from `upstream/main`:
```shell-session
$ git switch -c <new feature branch name> upstream/main
```
:::
::::


### 2. Edit the Code or Implement a New Feature

Edit the code and add as many commits as you like:
```shell-session
$ git add a-changed-file.py a-folder/another-changed-file.py
$ git commit
<short descriptive message in the editor window that opens>
```
You can also use the `-m <msg>` option when committing to use `<msg>` as the message
(i.e. without having to use the editor that otherwise opens):
```shell-session
$ git commit -m <short descriptive message>
```
Please follow the *Git conventions* when writing commit messages:
use the imperative, a short description as the first line, followed by a blank line,
and then followed by details if needed, e.g. as a bullet list.

```{seealso}
[Convetional Commits][conventionalcommits] for examples and information
on how to write good commit messages.
```
Make sure you frequently test the code during development (see {ref}`testing`).
Also ensure that your commits do not contain changes that are logically different.
Best practice is to stick with one commit per feature change. You can also commit *parts*
of a changed file using
```shell-session
$ git add -p <files>
```

### 3. Pushing Changes

The first time you push a new feature branch, you will need to specify which remote branch should
be pushed to. Usually this is `origin`:
```shell-session
$ git push -u origin <feature branch name>
```
::::{admonition} Automatically setting up remote tracking
:class: tip dropdown

As of Git version `2.37` you can set the behaviour of git so that a simple `git push` will also work
for the first push:
:::{code-block}shell-session
:class: no-copybutton
$ git config --global branch.autoSetupMerge simple
$ git config --global push.autoSetupRemote true
:::
::::

Whenever you push to that branch after that, you can just use
```shell-session
$ git push
```

### 4. (Optional) Merging `main` Into Your Feature Branch

Sometimes it is necessary to update your feature branch with updates from `main`.
That is usually the case if the `main` branch has received new updates that would
result in conflicts between the two branches.

::::{tab-set}
:sync-group: category

:::{tab-item} Working in the main repository
:sync: main

Fetch the latest changes from the main repository:
```shell-session
$ git fetch
```
And then either rebase your branch on `main`
```shell-session
$ git rebase origin/main
```
or merge `main` into your feature branch:
```shell-session
$ git merge origin/main
```
:::

:::{tab-item} Working in a fork
:sync: fork

Fetch the latest changes from the upstream of the main repository:
```shell-session
$ git fetch upstream
```
And then either rebase your branch on `upstream/main`
```shell-session
$ git rebase upstream/main
```
or merge `upstream/main` into your feature branch:
```shell-session
$ git merge upstream/main
```
:::
::::

:::{seealso}
[This tutorial][atlassian-merge] for differences between merging and rebasing.
:::


### 5. Create A Pull Request

Once you have implemented your new feature, feel free to open a [PR][github-pr] on the main page
of the repository. In the {octicon}`git-branch` "Branch" menu, choose your feature branch.
Click on "Compare & pull request" to create pull request. Write a summary of the features
you have implemented and describe all the changes.

#### Wait For a Code Review

At least one review must look at your code and review your changes. They may request changes before
accepting your pull request. In addition, we have checks that also have to pass:

- All unit tests must pass.
- The documentation must build without errors.
- All `pre-commit` hooks have to run without failing.

All these checks are automatically run by our [Continuous Integration][ci] via GitHub Actions.
If a reviewer asks for changes, implement them, then commit and push them.
Once your PR is accepted, the reviewer will merge your feature branch into the `main` branch
of the main repository https://github.com/radionets-project/pyvisgen.


#### Delete Your Feature Branch

Delete your feature branch once it is merged into `main` since it is no longer needed.
You can do that either on GitHub or in your local repository:

```shell-session
$ git switch main
```

::::{tab-set}
:sync-group: category

:::{tab-item} Working in the main repository
:sync: main

Pull the latest changes from the main repository:
```shell-session
$ git pull
```

:::

:::{tab-item} Working in a fork
:sync: fork

Fetch the latest changes from the upstream of the main repository:
```shell-session
$ git fetch upstream
```
And merge `upstream/main` into your local `main`:
```shell-session
$ git merge upstream/main --ff-only
```
:::
::::

Then delete your feature branch:
```shell-session
git branch --delete --remotes <name of the feature branch>
```


(coding-style)=
## Coding Style

`pyvisgen` follows the [PEP8 style guide][pep8] for Python. This is enforced via the [Ruff][ruff]
linter and code formatter and through the `pre-commit` hook set up in {ref}`getting_started_dev`.

## Tools Used in Development

`pyvisgen` is built and developed using:

[github-ssh]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
[github-fork]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
[github-sync]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
[pre-commit]: https://pre-commit.com/
[pyvisgen-pyproject]: https://github.com/radionets-project/pyvisgen/blob/main/pyproject.toml
[pytest]: https://docs.pytest.org/en/stable/
[conventionalcommits]: https://www.conventionalcommits.org/en/v1.0.0/
[atlassian-merge]: https://www.atlassian.com/git/tutorials/merging-vs-rebasing
[github-pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[ci]: https://docs.github.com/en/actions/get-started/continuous-integration
[pep8]: https://peps.python.org/pep-0008/
[ruff]: https://docs.astral.sh/ruff/
[pytorch]: https://pytorch.org/
