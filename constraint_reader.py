import re

class ConstraintReader():
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a ConstraintReader object from a constraints file

        :param fname: Name of the file to read the ConstraintReader from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])

        # Read in the constraints as strings
        self.constraints = []
        self.cmin_dict = {}
        self.cmax_dict = {}
        for i in range(2, len(lines)):
            # Support comments in the first line
            if lines[i][0] == "#":
                continue
            
            line = lines[i]
            self._check_min_max(line)

            # Only keep the portion before the ">= 0.0"
            line = line.split(">")[0]
            # Upgrade [#] to [:,#] to support testing entire vectors of points
            line = line.replace("[", "[:,")
            self.constraints.append(line)
        return

    def _check_min_max(self, line):
        """
        Search for min/max constraints on variables, assuming that the form
        of the constraint files will not significantly vary from the provided
        file.
        
        :param line: One line of the file containing a constraint expression (string)
        """

        # Pull out any constant -0.0004 in expressions like 'x[0] - 0.0004':
        cmin = re.match('x\[([0-9]+)\]\s*(-\s*[-+]?[0-9]*\.?[0-9]+)\s*>', line)
        if cmin is not None:
            i = int(cmin.group(1)) # the variable/dimension number
            val = -float(cmin.group(2).replace(' ', '')) # the constant
            self.cmin_dict[i] = val

        # Pull out any constant 0.014 in expressions like '0.014 - x[2] >= 0.0':
        cmax = re.match('([-+]?[0-9]*\.?[0-9]+)\s*-\s*x\[([0-9]+)\]\s*>', line)
        if cmax is not None:
            i = int(cmax.group(2)) # the variable/dimension number
            val = float(cmax.group(1).replace(' ', '')) # the constant
            self.cmax_dict[i] = val

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim

    def get_constraints(self):
        """Get the list of constraint strings"""
        return self.constraints

    def get_bounds(self):
        """Get any min/max bounds, if detected"""
        return (self.cmin_dict, self.cmax_dict)