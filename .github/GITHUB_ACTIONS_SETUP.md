# GitHub Actions Setup for Langvio

This document provides guidance on setting up the GitHub Actions workflows for Langvio.

## Workflows

Langvio uses two main GitHub Actions workflows:

1. **Test and Lint** (`.github/workflows/test-lint.yml`): Runs tests and linting on pull requests and pushes to main branches.
2. **Deploy to PyPI** (`.github/workflows/deploy.yml`): Deploys the package to PyPI when a new release is created.

## Setting up PyPI Deployment

To enable automatic deployments to PyPI, you need to set up a PyPI API token:

1. Create an account on [PyPI](https://pypi.org/) if you don't have one already.
2. Generate an API token from your PyPI account settings:
   - Go to https://pypi.org/manage/account/
   - Scroll to "API tokens"
   - Click "Add API token"
   - Give your token a name (e.g., "GitHub Actions")
   - Select "Entire account (all projects)" or scope it to the `langvio` project
   - Click "Create token"
   - Copy the token (you won't be able to see it again)

3. Add the token to your GitHub repository secrets:
   - Go to your GitHub repository
   - Click on "Settings" > "Secrets and variables" > "Actions"
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the PyPI token you copied
   - Click "Add secret"

## Running the Workflows

### Test and Lint Workflow

This workflow runs automatically on:
- Any push to `main`, `master`, or `develop` branches
- Any pull request targeting these branches

It will:
- Run the test suite with pytest and generate coverage reports
- Check code formatting with Black
- Check import order with isort
- Run flake8 to identify code quality issues

### Deploy Workflow

This workflow runs when you create a new release:

1. Go to your GitHub repository
2. Click on "Releases" on the right sidebar
3. Click "Create a new release"
4. Choose a tag (should match your version, e.g., `v0.3.1`)
5. Add a title and description
6. Click "Publish release"

The workflow will then:
- Build the package
- Upload it to PyPI using the stored API token

## Customizing the Workflows

You can customize the workflows by editing the YAML files in the `.github/workflows/` directory:

- **Python versions**: The test workflow runs on Python 3.8, 3.9, and 3.10. You can add or remove versions by modifying the `python-version` matrix.
- **Linting rules**: You can adjust flake8 settings in a `.flake8` file in the root of your repository.
- **Branches**: By default, workflows run on `main`, `master`, and `develop` branches. You can change this in the `on` section of each workflow.