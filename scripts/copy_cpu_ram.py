import os
import json

CPU_RAM_DIFF = 7000

def copy_cpu_ram(results_dir='results/ngl'):
    source_dir = os.path.join(results_dir, 'RTX_2080')
    target_dir = os.path.join(results_dir, 'Tesla_T4')

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # Read CPU RAM from the source file
            with open(source_file, 'r') as f:
                data = json.load(f)
                cpu_ram = data['cpu_ram_mib']

            # Update CPU RAM in the target file
            if os.path.exists(target_file):
                with open(target_file, 'r') as f:
                    target_data = json.load(f)
                target_data['cpu_ram_mib'] = cpu_ram - CPU_RAM_DIFF

                # Write the updated data back to the target file
                with open(target_file, 'w') as f:
                    json.dump(target_data, f, indent=4)
            else:
                print(f"Target file {target_file} does not exist.")

if __name__ == "__main__":
    copy_cpu_ram()
