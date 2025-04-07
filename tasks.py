from invoke import task

@task(
    help={
        "save": "Whether to save point clouds to disk",
        "no_visualize": "Disable point cloud visualization",
        "num_cameras": "Number of LiDAR cameras (1–6)",
        "h_steps": "Horizontal resolution of LiDAR scan",
        "v_steps": "Vertical resolution of LiDAR scan",
    }
)
def preprocess(ctx, save=False, no_visualize=False, num_cameras=3, h_steps=40, v_steps=40):
    """
    Run full preprocessing: generate point clouds from meshes and convert them to graph data.
    """
    cmd = (
        f"python3 scripts/preprocess.py "
        f"preprocessing.lidar.save={str(save).lower()} "
        f"preprocessing.lidar.visualize={str(not no_visualize).lower()} "
        f"preprocessing.lidar.num_cameras={num_cameras} "
        f"preprocessing.lidar.h_steps={h_steps} "
        f"preprocessing.lidar.v_steps={v_steps}"
    )
    ctx.run(cmd, pty=True)
