import sys
import getopt
import numpy as np
from point_generator import PointGenerator
from constraint_reader import ConstraintReader

def print_help():
    print("""
Multi-dimensional space sampler. Run as:
    sampler.py <input_file> <output_file> <n_results>
where:
    input_file is the name of a constraint .txt filefile
    output_file is the name of the file for saving output point vectors
    n_results is the number of point vectors to output (int)

Example:
    sampler.py "mixture.txt" "out.txt" 1000
    """)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h")
        for opt, _ in opts:
            if opt.lower() == '-h':
                print_help()
                return
        
        if len(args) != 3:
            print('Error: expected 3 inputs but found {}'.format(len(args)))
            print_help()
            return
        
        input_file = args[0]
        output_file = args[1]
        n_results = int(args[2])
        print('input_file: ' + input_file)
        print('output_file: ' + output_file)
        print('n_results: {}'.format(n_results))

    except:
        print('An error occurred parsing input parameters.')
        print_help()
        return

    constraints = ConstraintReader(input_file)
    n_dim = constraints.get_ndim()
    consts = constraints.get_constraints()
    known_bounds = constraints.get_bounds()

    pg = PointGenerator(n_dim, n_results, consts, known_bounds)
    x = pg.generate_points()
    np.savetxt(output_file, x)

if __name__ == "__main__":
    main(sys.argv[1:])
