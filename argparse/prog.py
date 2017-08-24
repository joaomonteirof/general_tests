import argparse

parser = argparse.ArgumentParser(description='Test')
#parser.add_argument('integers', metavar='N', type=int, nargs='+', help='integer')
parser.add_argument('integers', type=int, nargs='+', help='integer')
parser.add_argument('--to-print', type=str, default='nada', help='integer')
parser.add_argument('-sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum (default: max)')

args=parser.parse_args()
print(args.to_print)
print(args.accumulate(args.integers))
