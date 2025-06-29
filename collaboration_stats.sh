#!/bin/bash

echo "=== Claude Code Collaboration Statistics ==="
echo

echo "Co-authored commits:"
git log --grep="Co-Authored-By: Claude" --oneline | wc -l

echo
echo "Lines statistics:"
git log --grep="Co-Authored-By: Claude" --pretty=format:"%H" | xargs -I {} git show --numstat --pretty=format: {} | awk 'NF==3 {plus+=$1; minus+=$2} END {printf("Lines added: %d\nLines removed: %d\nNet lines: %d\n", plus, minus, plus-minus)}'

echo
echo "Collaboration sessions:"
git log --grep="Co-Authored-By: Claude" --pretty=format:"%cd %H" --date=format:"%Y-%m-%d %H:%M:%S" | python3 -c "
import sys
import subprocess
from collections import defaultdict
from datetime import datetime

commits_by_date = defaultdict(list)
for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) >= 3:
        date, time, commit_hash = parts[0], parts[1], parts[2]
        commits_by_date[date].append((time, commit_hash))

for date in sorted(commits_by_date.keys()):
    commits = sorted(commits_by_date[date])
    first_time, first_commit = commits[0]
    last_time, last_commit = commits[-1]
    
    # Calculate duration
    first_dt = datetime.strptime(f'{date} {first_time}', '%Y-%m-%d %H:%M:%S')
    last_dt = datetime.strptime(f'{date} {last_time}', '%Y-%m-%d %H:%M:%S')
    duration = last_dt - first_dt
    
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)
    
    if duration.total_seconds() == 0:
        duration_str = '(single commit)'
    else:
        duration_str = f'({int(hours)}h {int(minutes)}m)'
    
    # Get line counts for this day's commits
    commit_hashes = [commit[1] for commit in commits]
    cmd = ['git', 'show', '--numstat', '--pretty=format:'] + commit_hashes
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    plus, minus = 0, 0
    for line in result.stdout.strip().split('\n'):
        if line and '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                plus += int(parts[0])
                minus += int(parts[1])
    
    print(f'{date}: {first_time} - {last_time} {duration_str} (+{plus}/-{minus})')
"