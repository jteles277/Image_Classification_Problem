def get_1():
    input_file = 'input.txt'
    output_file = 'output.txt'

    with open(input_file, 'r') as file:
        data = file.read()

    lines = data.split('\n')

    results = []
    for line in lines:
        if "f=  " in line and 'D+05' in line:
            start_index = line.index('f=') + 3
            end_index = line.index('D+05', start_index)
            result = line[start_index:end_index]
            results.append(result)

    with open(output_file, 'w') as file:
        file.write('\n'.join(results))


def get_2():
    file_path = "input.txt"  # Replace with the actual file path

    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the values by commas
    values = content.split(", ")

    # Write each value on a separate line
    with open(file_path, 'w') as file:
        file.write('\n'.join(values))


get_2()