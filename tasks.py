# equivariant-center-of-mass-estimation/tasks.py
from invoke import task

@task
def preprocess(ctx, save=False, no_visualize=False):
    """
    Preprocess all .obj files.
    """
    command = "python3 scripts/preprocess.py"
    if save:
        command += " --save"
    if no_visualize:
        command += " --no-visualize"
    ctx.run(command, pty=True)
