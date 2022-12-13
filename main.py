from argparse import ArgumentParser
from mpi4py import MPI

import numpy as np
import os


def parse_args():
    parser = ArgumentParser('Process files')
    parser.add_argument('--mode', required=True, choices=['sequence', 'parallel'])

    return parser.parse_args()


def parallel_pipeline():
    world_size = MPI.COMM_WORLD.Get_size()
    world_rank = MPI.COMM_WORLD.Get_rank()
    files = os.listdir('files')

    if world_rank == 0:
        num_files = len(files)
        count_files = [0]

        # Create count files that separates all files equally by process
        for index in range(world_size):
            index += 1
            count_files.append(num_files)

        end_files = np.cumsum(count_files[1:]).astype(np.int32)
        start_files = np.cumsum(count_files[:-1]).astype(np.int32)
    else:
        end_files = None
        start_files = None
    start_local = np.empty(1, dtype=np.int32)
    end_local = np.empty(1, dtype=np.int32)

    # Send start files and end files via Scatter
    MPI.COMM_WORLD.Scatter(start_files, start_local, root=0)
    MPI.COMM_WORLD.Scatter(end_files, end_local, root=0)
    start_file = start_local[0]
    end_file = end_local[0]
    data = read_files(start_file, end_file)
    stats = calculate_statistics(data)
    received_stats = MPI.COMM_WORLD.gather(stats, root=0)

    if world_rank == 0:
        return accumulate_stats(received_stats)
    else:
        return None


def accumulate_stats(stats):
    all_count = stats[0]['count']
    all_sums = stats[0]['sum']
    all_sums_squared = sum(stats[0]['sum_squares'])

    avg = all_sums / all_count
    var = all_sums_squared / all_count - (avg**2)

    results = {
        'avg': avg,
        'std': np.sqrt(var),
        'hist': {}
    }

    for key in ['1M', '10M', '100M', '1B', '1B+']:
        # Calculate sum of values by stats
        result = sum(item[key] for item in stats)
        results['hist'][key] = result
    return results


def calculate_statistics(client_sums):
    sums = np.sum(client_sums)
    sum_squares = [number ** 2 for number in client_sums]

    below_1M = len([i for i in client_sums if i < 1000])
    below_10M = len([i for i in client_sums if i < 10000])
    below_100M = len([i for i in client_sums if i < 100000])
    below_1B = len([i for i in client_sums if i < 1000000])
    more_1B = len([i for i in client_sums if i >= 1000000])
    count = len(client_sums)

    count_1M_10M = below_10M - below_1M
    count_10M_100M = below_100M - below_10M
    count_100M_1B = below_1B - below_100M

    return {
        'count': count,
        '1M': below_1M,
        '10M': count_1M_10M,
        '100M': count_10M_100M,
        '1B': count_100M_1B,
        '1B+': more_1B,
        'sum': sums,
        'sum_squares': sum_squares
    }


def read_files(from_file: int, to_file: int):
    """
        Read data in files in interval [from_file, to_file)
        Outputs np.array of data files in every interval
    """
    batch_values = []
    for index in range(from_file, to_file):
        file = open(f"files/{index}.txt", mode='r')
        lines = file.readlines()
        for line in lines:
            balance = float(line)
            batch_values.append(balance)
    batch_values = np.array(batch_values)
    return np.concatenate([batch_values], axis=0)


def sequence_pipeline():
    files = os.listdir('files')
    num_files = len(files)
    start_file = int(str.replace(files[0], '.txt', ''))
    end_file = int(str.replace(files[-1], '.txt', ''))
    data = read_files(start_file, end_file)
    stats = calculate_statistics(data)
    return accumulate_stats([stats])


def main():
    args = parse_args()
    if args.mode == 'parallel':
        world_rank = MPI.COMM_WORLD.Get_rank()
        output = parallel_pipeline()
        if output is not None:
            print(output)
    else:
        output = sequence_pipeline()
        print(output)


if __name__ == '__main__':
    main()
