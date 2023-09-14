import subprocess
import os

def consensus_abpoa(cluster:list) -> list:
    fn = f'/tmp/abpoa_{os.getpid()}_{cluster[0][:19]}'
    with open(fn, 'w') as f:
        f.write(''.join([f'>{i}\n{seq}\n' for i, seq in enumerate(cluster)]))
    result = subprocess.run(['./bin/abpoa', fn, '-m', '0', '-d', '1'], capture_output=True, text=True).stdout
    consensus_list = [seq for seq in result.strip().split('\n') if not seq.startswith('>')]
    print(result)
    return consensus_list[0]