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

# Generate one complex long-horizon task (bottom-up, uses data_longhorizon_complex/)
# Pass the hyphenated task-name from tmp.json / base_tasks.json (e.g. prepare-breakfast)
run-complex name:
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --target-task-name {{name}} --num-tasks 1

# Aliases for common tasks
prepare-breakfast:
    just run-complex prepare-breakfast

tidy-up-table:
    just run-complex tidy-up-table

move-mug-into-drawer:
    just run-complex move-mug-into-drawer

# Generate all complex tasks from tmp.json / GenSim2 base_tasks.json (50 detached Modal jobs)
run-all-complex:
    just run-complex move-mug-into-drawer
    just run-complex tidy-up-table
    just run-complex prepare-breakfast
    just run-complex store-soup-in-drawer
    just run-complex store-cutlery-in-drawer
    just run-complex organize-office-drawer
    just run-complex refrigerate-fruits
    just run-complex store-condiments-in-fridge
    just run-complex refrigerate-canned-goods
    just run-complex heat-soup-in-microwave
    just run-complex microwave-meal-prep
    just run-complex store-gold-in-safe
    just run-complex secure-tool-in-safe
    just run-complex store-tools-in-box
    just run-complex fill-toy-box
    just run-complex store-pantry-items-in-box
    just run-complex pack-suitcase-with-toys
    just run-complex pack-travel-essentials
    just run-complex bake-in-oven
    just run-complex oven-prep
    just run-complex load-dishwasher
    just run-complex load-washing-machine
    just run-complex kitchen-cleanup
    just run-complex organize-kitchen-appliances
    just run-complex secure-and-organize
    just run-complex morning-routine-prep
    just run-complex tidy-workspace
    just run-complex refrigerate-and-microwave
    just run-complex store-tools-in-box-and-drawer
    just run-complex organize-pantry
    just run-complex multi-appliance-kitchen
    just run-complex store-snacks-and-press-toaster
    just run-complex stapler-and-organize
    just run-complex open-laptop-and-store
    just run-complex faucet-and-organize
    just run-complex switch-and-store
    just run-complex pack-sports-bag
    just run-complex prepare-travel-bag
    just run-complex fill-bag-with-food
    just run-complex full-kitchen-reset
    just run-complex breakfast-prep-advanced
    just run-complex office-organization
    just run-complex workshop-cleanup
    just run-complex pantry-restock
    just run-complex cooking-prep
    just run-complex travel-packing
    just run-complex grocery-storage
    just run-complex tool-organization
    just run-complex household-chores
    just run-complex meal-prep-station

# Run all primitive substep tasks from task_lib.json
run-all-articulated:
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name turn_on_faucet
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_laptop
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_box
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name turn_off_faucet
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_laptop
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_box   

    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_drawer
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name push_drawer_close
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name press_toaster_lever
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name swing_bucket_handle
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_safe 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_refrigerator_door
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_safe 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name swing_door_open 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_refrigerator_door
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name toggle_door_close 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name swing_window_left 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name rotate_microwave_door 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name lift_toilet_lid 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name relocate_suitcase 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name swing_suitcase_lid_open 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name open_washing_machine_door
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name press_toilet_lid_down  
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_suitcase_lid 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name flip_switch 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name push_toaster_forward 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name lift_bucket_upright
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name tilt_window_open 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name rotate_dishwasher_knob_open 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name lift_bag
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name move_bag_forward 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name push_box 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name sway_bag_strap 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_microwave 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name rotate_oven_knob 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name push_oven_close 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name move_window_right
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_window_by_rotating
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name close_dishwasher 
    uv run modal run --detach launch.py --task longhorizon_bottomup --prompt-data-folder data_longhorizon_complex/ --num-tasks 1 --target-task-name press_stapler_lever 
    
    
