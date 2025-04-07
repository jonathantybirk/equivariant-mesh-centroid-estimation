from invoke import task

@task(
    help={
        "save": "Whether to save results to disk",
        "no_visualize": "Disable point cloud visualization",
        "num_cameras": "Number of virtual LiDAR cameras (1–6)",
        "h_steps": "Horizontal resolution of LiDAR scan",
        "v_steps": "Vertical resolution of LiDAR scan",
    }
)
def preprocess(ctx, save=False, no_visualize=False, num_cameras=3, h_steps=40, v_steps=40):
    """
    Preprocess .obj files into point clouds using simulated LiDAR and save results.

    Examples:
        invoke preprocess --save --num-cameras=4 --no-visualize
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
