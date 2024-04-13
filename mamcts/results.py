def parse_log_file(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    test_cases = []
    current_test = []
    for line in lines:
        if line.strip().startswith("Contents of"):
            if current_test:
                test_cases.append(current_test)
                current_test = []
        else:
            current_test.append(line.strip())

    if current_test:
        test_cases.append(current_test)

    return test_cases

def parse_test_case(test_case):
    test_case_dict = {}
    for config in test_case:
        if config.startswith("Workers Config"):
            config = config.split(": ")
            worker_idx = config[0].split(' ')[2]
            workers = config[1].split('], ')[0]
            time = config[2].split(' ')[0]
            profit = config[3]
            
            test_case_dict[worker_idx] = {'workers': workers, 'time': time, 'profit': profit}
    
    return test_case_dict

def get_best_config(test_cases_dict):
    best_configs = {}
    
    for test_case, configurations in test_cases_dict.items():
        best_profit = float('-inf')
        best_configs[test_case] = []
        
        for config_id, config_data in configurations.items():
            profit = int(config_data['profit'])
            
            if profit > best_profit:
                best_profit = profit
                best_configs[test_case] = [(config_id, config_data)]
            elif profit == best_profit:
                best_configs[test_case].append((config_id, config_data))
                
    return best_configs

def main():
    
    parsed_test_cases = {}
    
    INPUT_LOG_FILE = "test-logs.txt"
    OUTPUT_LOG_FILE = "test-results.txt"
    
    test_cases = parse_log_file(INPUT_LOG_FILE)
    for idx, test_case in enumerate(test_cases):
        test_case_dict = parse_test_case(test_case)
        parsed_test_cases[idx+1] = test_case_dict
    
    best_configs = get_best_config(parsed_test_cases)

    with open(OUTPUT_LOG_FILE, 'w') as f:
        for test_case, configs in best_configs.items():
            for config in configs:
                _, data = config
                workers, profit = f'{data['workers']}]', data['profit']
                f.write(f"Graph {test_case} -  {workers} , {profit}\n")
                break # only write the first best config, modify if necessary

if __name__ == "__main__":
    main()
