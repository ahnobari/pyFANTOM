# Read the Docs Setup Instructions

## Quick Setup

1. **Create a Read the Docs account** at https://readthedocs.org/
   - Sign up with GitHub/GitLab/Bitbucket if your repo is hosted there

2. **Import your repository**:
   - Go to https://readthedocs.org/dashboard/
   - Click "Import a Project"
   - Select your repository (pyFANTOM)
   - Project name: `pyfantom` (lowercase, no spaces)
   - Click "Next"

3. **Configure the project**:
   - **Repository URL**: Your git repository URL
   - **Default branch**: `main` or `master` (whichever you use)
   - **Python configuration file**: `.readthedocs.yml` (already created)
   - **Requirements file**: `docs/requirements.txt`
   - **Documentation type**: Sphinx
   - **Sphinx configuration file**: `docs/source/conf.py`

4. **Advanced Settings** (optional):
   - Go to Project → Admin → Advanced Settings
   - **Install Project**: Check this if you want to install the package
   - **Python Interpreter**: Python 3.10
   - **Use system packages**: Unchecked (recommended)

5. **Build the documentation**:
   - Click "Build version" on the project dashboard
   - Wait for the build to complete
   - Your docs will be available at: `https://pyfantom.readthedocs.io/`

## Troubleshooting

- If the build fails, check the build logs in the Read the Docs dashboard
- Make sure all dependencies in `docs/requirements.txt` are installable
- The `.readthedocs.yml` file should be in the root of your repository
- Ensure your `conf.py` path is correct: `docs/source/conf.py`

## Automatic Builds

Read the Docs will automatically rebuild your documentation when you push to your repository if:
- Webhooks are properly configured (usually automatic for GitHub/GitLab)
- You have "Build on commit" enabled in project settings

