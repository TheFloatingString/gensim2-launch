set windows-shell := ["cmd", "/c"]

# GenSim2 Modal launcher

# Short horizon (articulated) task
articulated num_tasks="1":
    uv run modal run --detach launch.py --task articulated --num-tasks {{num_tasks}}

# Long horizon task
longhorizon num_tasks="1":
    uv run modal run --detach launch.py --task longhorizon --num-tasks {{num_tasks}}

# Long horizon bottom-up task
bottomup num_tasks="1":
    uv run modal run --detach launch.py --task longhorizon_bottomup --num-tasks {{num_tasks}}

# Custom task by object name (long horizon)
custom1:
    uv run modal run --detach launch.py --task longhorizon_bottomup --target-object-name "scissors" --num-tasks 1

# Generate PrepareBreakfast task (long horizon bottom-up, complex data)
prepare-breakfast:
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --target-task-name prepare-breakfast --num-tasks 1

# Generate TidyUpTable task (long horizon bottom-up, complex data)
tidy-up-table:
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --target-task-name tidy_up_table --num-tasks 1

# Generate MoveMugIntoDrawer task (long horizon bottom-up, complex data)
move-mug-into-drawer:
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --target-task-name move_mug_into_drawer --num-tasks 1