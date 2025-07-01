source prerequisites/venv/bin/activate

mkdir robust_pomdps_logs

for entry in $(ls $1); do
    echo "Running robust RL on model $entry"
    python3 robust_pomdps_rl.py --project-path $1/$entry > robust_pomdps_logs/$entry.log 2>&1 
done