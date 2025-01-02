import itertools

# Define the ranges for each number
first_numbers = [2, 3, 4]
second_numbers = [32, 64, 128, 256]
third_numbers = [16, 32, 64]

# Generate all combinations
combinations = itertools.product(first_numbers, second_numbers, third_numbers)

# Create the list of commands
commands = []
for combo in combinations:
    command = f"python main.py train ae {combo[0]},{combo[1]},{combo[2]} --convergence --plot && \\"
    commands.append(command)

# Print all commands
for cmd in commands:
    print(cmd)
