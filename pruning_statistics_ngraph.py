import pandas as pd


def parse_pruning_transformation_log(file_path):
    lines = open(file_path, 'r').readlines()
    conv_matcher_name = 'ConvolutionInitMask'

    conv_stat = []
    for idx, line in enumerate(lines):
        if conv_matcher_name in line:
            splited_line = line.split()
            conv_name = splited_line[splited_line.index(conv_matcher_name) + 3]
            conv_stat_line = [conv_name]
            if idx + 2 >= len(lines):
                continue

            line_with_mask = lines[idx+2]
            if 'MASK' in line_with_mask:
                conv_stat_line.append(line_with_mask.split('[')[-1][:-4])
            else:
                conv_stat_line.append('-')
            conv_stat.append(conv_stat_line)

    df = pd.DataFrame(conv_stat)
    # Here you can do any queries
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)


def main():
    file_path = '/home/dlyakhov/model_export/25_11_21/log.txt'
    parse_pruning_transformation_log(file_path)


if __name__ == '__main__':
    main()
