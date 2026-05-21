set windows-shell := ["cmd", "/c"]

# GenSim2 Modal launcher

# Short horizon (articulated) task
articulated num_tasks="1":
    uv run modal run --detach launch.py --task articulated --num-tasks {{num_tasks}}

# Long horizon task
longhorizon num_tasks="1":
    uv run modal run --detach launch.py --task longhorizon --num-tasks {{num_tasks}}
